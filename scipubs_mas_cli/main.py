from scipubs_mas_cli.app.graph_app import run_pipeline
from scipubs_mas_cli.app.memory import ensure_memory_table, load_recent
from pathlib import Path
import base64
from scipubs_mas_cli.app.db import run_sql
from  datetime import datetime
import os
import sys
import csv


def export_rows_csv(rows: list[dict], out_path: Path) -> None:
    """Сохраняет результат raw_table_view в CSV (UTF-8)."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # создадим пустой файл с заголовком, если можем восстановить колонки
        out_path.write_text("", encoding="utf-8")
        return

    # Предсказуемый порядок колонок (как в Collector.raw_table_view)
    preferred = ["doi", "title", "publication_year", "cited_by_count", "journal", "topic"]
    # Если в данных есть неожиданные колонки — добавим их в конец
    extra_cols = [k for k in rows[0].keys() if k not in preferred]
    fieldnames = [c for c in preferred if c in rows[0]] + extra_cols

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # приводим значения к простым типам
            writer.writerow({k: r.get(k) for k in fieldnames})

def export_selected_publications_txt(sql_params: dict, out_path: Path, batch_size: int = 5000) -> int:
    """
    Сохраняет в TXT все публикации, выбранные системой, по тем же фильтрам что и агрегирующий запрос.
    Возвращает количество сохранённых публикаций.
    """
    haystack = (
        "lower(concat_ws(' ', "
        "coalesce(topic,''), coalesce(subfield,''), coalesce(field,''), "
        "coalesce(domain,''), coalesce(title,''), coalesce(abstract,'')"
        "))"
    )

    where_parts = []
    params = dict(sql_params or {})

    if "year_from" in params and "year_to" in params:
        where_parts.append("publication_year BETWEEN %(year_from)s AND %(year_to)s")

    # Collector может использовать либо `patterns` (обычный поиск),
    # либо `grp_patterns` (compare_groups). Для compare_groups используем
    # именно grp_patterns, чтобы экспорт совпадал с агрегирующим запросом.
    if params.get("grp_patterns"):
        patterns = params.get("grp_patterns")
        params["patterns"] = patterns
    else:
        patterns = params.get("patterns")

    if patterns:
        where_parts.append(f"{haystack} LIKE ANY(%(patterns)s)")

    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

    export_sql = (
        "SELECT doi, title, publication_year, journal, cited_by_count, domain, field, subfield, topic\n"
        "FROM articles_cast"
        f"{where_sql}\n"
        "ORDER BY publication_year, doi\n"
        "LIMIT %(export_limit)s OFFSET %(export_offset)s"
    )

    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Selected publications export\n")
        f.write(f"Created: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Filters: {sql_params}\n")
        f.write("=" * 80 + "\n\n")

        export_offset = 0
        while True:
            params["export_limit"] = batch_size
            params["export_offset"] = export_offset

            batch = run_sql(export_sql, params=params)
            if not batch:
                break

            for row in batch:
                total += 1
                f.write(f"#{total}\n")
                f.write(f"DOI: {row.get('doi')}\n")
                f.write(f"Year: {row.get('publication_year')}\n")
                f.write(f"Title: {row.get('title')}\n")
                f.write(f"Journal: {row.get('journal')}\n")
                f.write(f"Cited by: {row.get('cited_by_count')}\n")
                f.write(f"Domain: {row.get('domain')}\n")
                f.write(f"Field: {row.get('field')}\n")
                f.write(f"Subfield: {row.get('subfield')}\n")
                f.write(f"Topic: {row.get('topic')}\n")
                f.write("-" * 80 + "\n\n")

            export_offset += len(batch)

    return total



def print_memory(limit: int = 5) -> None:
    """Печатает последние записи из long-term memory (agent_memory)."""
    try:
        items = load_recent(limit=limit)
    except Exception as e:
        print(f"\nНе удалось прочитать память: {e}")
        return

    print("\n" + "=" * 80)
    print(f"Последние записи agent_memory (top {limit}):")
    if not items:
        print("Память пуста.")
    for r in items:
        ts = str(r.get("created_at"))
        q = (r.get("user_query") or "").strip().replace("\n", " ")
        rc = r.get("row_count")
        print(f"- {ts} | rows={rc} | {q[:140]}")
    print("=" * 80 + "\n")


def run_one_query(
    user_query: str,
    interactive: bool,
) -> tuple[dict, str] | None:
    """Запускает пайплайн; при необходимости уточнения запрашивает переформулировку."""
    while True:
        state = run_pipeline(user_query)

        if state.get("needs_clarification"):
            print("=" * 80)
            print("Запрос пользователя:")
            print(user_query)
            print("=" * 80)
            print("\nНужно уточнение:\n")
            print(state.get("clarification_prompt") or "Пожалуйста, переформулируйте запрос более конкретно.")

            if not interactive:
                return state, user_query
            else:
                return state, user_query
            new_q = input("\nПереформулируйте запрос (или /exit): ").strip()
            if not new_q or new_q.lower() in {"/exit", "exit", "quit", "q"}:
                return None

            # Дополнительные команды даже внутри уточнения
            if new_q.lower().startswith("/memory"):
                parts = new_q.split()
                lim = 5
                if len(parts) >= 2 and parts[1].isdigit():
                    lim = int(parts[1])
                print_memory(limit=lim)
                continue
            user_query = new_q
            continue

        return state, user_query


def render_result(state: dict, user_query: str) -> None:
    attempt = state.get("attempt", 1)

    classifier_result = state.get("classifier_result")
    analyst_result = state.get("analyst_result")
    rows = state.get("rows", [])
    sql = state.get("sql")
    sql_params = state.get("sql_params")
    collector_src = (state.get("collector_data_source") or "sql").lower()
    collector_meta = state.get("collector_meta")
    tool_routing = state.get("tool_routing")
    memory_saved = state.get("memory_saved")
    memory_error = state.get("memory_error")
    plan_obj = state.get("planner_plan")
    intent = getattr(plan_obj, "user_intent", None) if plan_obj is not None else None

    print("=" * 80)
    print(f"\nИсточник данных Collector: {collector_src}")
    if isinstance(tool_routing, dict) and tool_routing:
        # This comes from the in-graph Tool Router node.
        mode = tool_routing.get("collector_mode")
        style = tool_routing.get("request_style")
        reason = tool_routing.get("reason")
        print(f"Tool Router decision: mode={mode}, style={style}, reason={reason}")
    if collector_meta is not None:
        print("Collector meta:")
        print(collector_meta)
    print("Запрос пользователя:")
    print(f"Попытка (итоговая): {attempt}")
    print(user_query)
    print("=" * 80)

    if classifier_result is not None:
        try:
            def _get(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            used = bool(_get(classifier_result, "openalex_used", False))
            ok = bool(_get(classifier_result, "openalex_ok", True))
            reqs = int(_get(classifier_result, "openalex_requests", 0) or 0)
            total = int(_get(classifier_result, "openalex_total_results", 0) or 0)

            if not used:
                print("\nOpenAlex: SKIPPED (пользователь не запрашивал обращение к OpenAlex)")
            elif ok:
                print(f"\nOpenAlex fetch: OK (requests={reqs}, results={total})")
            else:
                print(f"\nOpenAlex fetch: FAILED (requests={reqs}, results={total})")
                errors = _get(classifier_result, "openalex_errors") or []
                for e in errors[:3]:
                    print(f"  - {e}")
                if len(errors) > 3:
                    print(f"  ... and {len(errors) - 3} more")

            sel_log = _get(classifier_result, "openalex_selection_log") or []
            if used and sel_log:
                print("\nOpenAlex selection (anchor → selected):")
                for item in sel_log:
                    base_q = item.get("base_query")
                    if item.get("error"):
                        print(f"- {base_q}: ERROR: {item['error']}")
                        continue
                    anchor = item.get("anchor") or {}
                    anc_name = anchor.get("topic")
                    print(f"- {base_q} | anchor: {anc_name}")
                    for s in (item.get("selected") or []):
                        sim = s.get("sim")
                        sim_txt = f"{sim:.2f}" if isinstance(sim, (int, float)) else "?"
                        print(f"    • {s.get('topic')} (field={s.get('field')}, sim={sim_txt})")

        except Exception:
            pass

    if sql:
        print("\nСгенерированный SQL-запрос:\n")
        print(sql)
        if sql_params:
            print("\nПараметры SQL:")
            print(sql_params)

    print(f"\nЧисло строк результата: {len(rows)}")

    if memory_saved is True:
        print("\nLong-term memory: запись сохранена в agent_memory")
    elif memory_saved is False:
        print(f"\nLong-term memory: НЕ удалось сохранить запись ({memory_error})")

    # Печатаем данные, чтобы аналитический текст был проверяемым
    if rows:
        row0 = rows[0]
        if "publication_year" in row0 and "n_papers" in row0 and "group_topic" not in row0:
            if "source" in row0:
                print("\nДанные (год, источник → число публикаций):")
                for r in rows:
                    print(f"  {r.get('publication_year')}, {r.get('source')}: {r.get('n_papers')}")
            else:
                print("\nДанные (год → число публикаций):")
                for r in rows:
                    print(f"  {r.get('publication_year')}: {r.get('n_papers')}")
        elif "publication_year" in row0 and "group_topic" in row0 and "n_papers" in row0:
            print("\nДанные (год, группа → число публикаций):")
            for r in rows:
                print(f"  {r.get('publication_year')}, {r.get('group_topic')}: {r.get('n_papers')}")

    if analyst_result:
        print("\nАналитический комментарий:\n")
        print(analyst_result.summary)
    else:
        print("\nАналитический агент не вернул результат.")

    if analyst_result and analyst_result.plot_type and analyst_result.plot_base64:
        print(f"\nПостроен график типа: {analyst_result.plot_type}")
        print("График закодирован в base64 (первые 200 символов):")
        print(analyst_result.plot_base64[:200] + "...")

        # --- SAVE & OPEN PNG ---
        out_path = Path("plot.png").resolve()
        png_bytes = base64.b64decode(analyst_result.plot_base64)
        out_path.write_bytes(png_bytes)
        print(f"\nГрафик сохранён в файл: {out_path}")

        # Windows: открыть стандартным просмотрщиком
        try:
            os.startfile(str(out_path))  # type: ignore[attr-defined]
        except Exception as e:
            print(f"Не удалось автоматически открыть файл: {e}")
            print("Откройте plot.png вручную.")

    # --- SAVE RAW TABLE TO CSV (raw_table_view) ---
    # Для raw_table_view экспортируем CSV независимо от источника (SQL/OpenAlex/both),
    # если есть строки.
    if intent == "raw_table_view" and rows:
        csv_name = f"raw_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = Path("exports") / csv_name
        export_rows_csv(rows, csv_path)
        print(f"\nТаблица сохранена в CSV: {csv_path.resolve()}")

    # --- EXPORT ALL SELECTED PUBLICATIONS TO TXT ---
    sql_params = state.get("sql_params") or {}
    export_path = Path("exports") / "selected_publications.txt"

    # Защита от случайного экспорта *всей* таблицы при пустых фильтрах.
    if collector_src not in {"sql", "both"}:
        print("\nЭкспорт в TXT пропущен: источник данных не SQL.")
    elif not (sql_params.get("patterns") or sql_params.get("grp_patterns")):
        print("\nЭкспорт в TXT пропущен: нет текстовых фильтров (patterns/grp_patterns).")
    else:
        if collector_src == "both":
            print("\nЭкспорт в TXT будет выполнен только по SQL-выборке (локальная БД); OpenAlex не участвует.")
        n = export_selected_publications_txt(sql_params, export_path)
        print(f"\nЭкспортировано публикаций в TXT: {n}")
        print(f"Файл: {export_path.resolve()}")

    print("\nГотово.")



def main() -> None:
    # Гарантируем, что таблица долговременной памяти существует
    ensure_memory_table()

    interactive = sys.stdin.isatty()
    default_query = "Построй динамику числа публикаций по теме deep learning с 2010 по 2024 год"

    # Если передан запрос аргументами командной строки — используем его как первый запрос.
    initial_query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""

    # Неинтерактивный запуск (например, пайплайн/редирект stdout): один прогон и выход.
    if not interactive:
        user_query = initial_query or default_query
        out = run_one_query(
            user_query,
            interactive=False,
        )
        if out is None:
            return
        state, final_q = out
        render_result(state, final_q)
        return

    # Интерактивный REPL-режим: непрерывный диалог.
    print("CLI-диалог запущен.")
    print("Команды: /exit, /memory, /memory N")
    print("По умолчанию система использует только локальную БД (SQL).")
    print("OpenAlex используется ТОЛЬКО если вы явно попросили об этом в текущем запросе.")
    print("Примеры:")
    print("  - @openalex Построй динамику публикаций по теме deep learning 2010–2024")
    print("  - @both Сравни динамику публикаций по теме NLP 2015–2024 в SQL и OpenAlex")
    print("  - Используй OpenAlex: динамика публикаций по graph neural networks 2018–2024")

    # Выполним первый запрос, если он был передан в аргументах.
    if initial_query:
        out = run_one_query(
            initial_query,
            interactive=True,
        )
        if out is None:
            print("Выход.")
            return
        state, final_q = out
        render_result(state, final_q)

    while True:
        try:
            user_query = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return

        if not user_query:
            continue

        low = user_query.lower()
        if low in {"/exit", "exit", "quit", "q"}:
            print("Выход.")
            return

        if low.startswith("/memory"):
            parts = user_query.split()
            limit = 5
            if len(parts) >= 2 and parts[1].isdigit():
                limit = int(parts[1])
            print_memory(limit=limit)
            continue

        out = run_one_query(
            user_query,
            interactive=True,
        )
        if out is None:
            print("Выход.")
            return

        state, final_q = out
        render_result(state, final_q)




if __name__ == '__main__':
    # main()
    o = run_one_query("Построй динамику числа публикаций по теме deep learning с 2020 по 2024", interactive=False)
    print(o)

    state, user_query = o

    print(state.keys())
