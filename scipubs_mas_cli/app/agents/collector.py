from __future__ import annotations

"""Collector agent.

Раньше Collector просил LLM сгенерировать SQL целиком. Это приводило к двум
проблемам:
1) модель «раздувала» WHERE, размножая ключевые слова по многим колонкам;
2) генерация занимала заметное время.

В этой версии Collector строит SQL ДЕТЕРМИНИРОВАННО:
 - принимает структурированный результат Classifier (search_terms);
 - извлекает диапазон лет из plan.question_rewrite;
 - компилирует короткий параметризованный SELECT.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field, ConfigDict

from ..db import run_sql
from ..config import settings
from ..tools.openalex import group_works, list_works
from ..safety import is_safe_sql
from .planner import PlannerPlan
from .classifier import ClassifierResult


# -------------------- Models --------------------


class CollectorOutput(BaseModel):
    """Что возвращает Collector в пайплайне."""

    model_config = ConfigDict(extra="ignore")

    sql: str | None = Field(None, description="Параметризованный SELECT-запрос (если источник данных — SQL)")
    params: Dict[str, Any] | None = Field(
        default=None, description="Параметры для psycopg2 (если используются)"
    )

    # Откуда взяты итоговые строки: локальная БД (SQL) или OpenAlex API
    data_source: Literal["sql", "openalex", "both"] = "sql"

    # Debug: какие темы/таксономия были задействованы (из Classifier/OpenAlex Topics)
    used_openalex_topics: bool = False
    openalex_topics_payload: Any | None = None

    # Debug: какие параметры были использованы при запросе данных из OpenAlex
    openalex_data_meta: Any | None = None


class QuerySpec(BaseModel):
    """Внутренняя спецификация запроса, из которой компилируется SQL."""

    model_config = ConfigDict(extra="ignore")

    intent: str = "other"
    search_terms: List[str] = Field(default_factory=list)
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    limit: Optional[int] = None


# -------------------- Helpers --------------------


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _parse_year_range(text: str) -> tuple[Optional[int], Optional[int]]:
    """Извлекает годы из текста и возвращает (min, max).

    Если найден один год — считаем (year_from == year_to).
    """
    years = [int(y) for y in YEAR_RE.findall(text or "")]
    if not years:
        return None, None
    if len(years) == 1:
        return years[0], years[0]
    return min(years), max(years)


def _normalize_terms(terms: List[str], max_terms: int = 8) -> List[str]:
    """Нормализует и ограничивает число терминов (защита от разрастания).

    Важно: Collector НЕ делает alias/расширений терминов.
    Все нормализации/перевод должны происходить в Classifier.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in terms or []:
        t = " ".join(str(t).strip().split())
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _haystack_expr() -> str:
    """Единое текстовое поле для поиска, чтобы не размножать условия по колонкам."""
    return (
        "lower(concat_ws(' ', "
        "coalesce(topic,''), coalesce(subfield,''), coalesce(field,''), "
        "coalesce(domain,''), coalesce(title,''), coalesce(abstract,'')))"
    )


def _build_where(
    search_terms: List[str],
    year_from: Optional[int],
    year_to: Optional[int],
) -> tuple[str, Dict[str, Any]]:
    """Строит WHERE и параметры."""
    where_parts: list[str] = []
    params: Dict[str, Any] = {}

    terms = _normalize_terms(search_terms, max_terms=8)
    if terms:
        params["patterns"] = [f"%{t.lower()}%" for t in terms]
        where_parts.append(f"{_haystack_expr()} LIKE ANY(%(patterns)s)")

    if year_from is not None and year_to is not None:
        params["year_from"] = int(year_from)
        params["year_to"] = int(year_to)
        if year_from == year_to:
            where_parts.append("publication_year = %(year_from)s")
        else:
            where_parts.append("publication_year BETWEEN %(year_from)s AND %(year_to)s")

    if not where_parts:
        return "", {}
    return " WHERE " + " AND ".join(where_parts), params


# -------------------- SQL compilation --------------------


def build_query_spec(plan: PlannerPlan, clf: ClassifierResult | None) -> QuerySpec:
    year_from, year_to = _parse_year_range(plan.question_rewrite or "")

    terms: list[str] = []
    if clf is not None:
        # search_terms уже ограничены Classifier'ом и строятся на базе OpenAlex
        # (с безопасной деградацией, если OpenAlex вернул 0 результатов).
        terms = list(clf.search_terms or [])

    limit = None
    if plan.user_intent == "raw_table_view":
        limit = 100
    elif plan.user_intent == "top_journals":
        limit = 20

    return QuerySpec(
        intent=plan.user_intent,
        search_terms=terms,
        year_from=year_from,
        year_to=year_to,
        limit=limit,
    )


def compile_sql(spec: QuerySpec, clf: ClassifierResult | None = None) -> CollectorOutput:
    intent = spec.intent or "other"
    where_sql, params = _build_where(spec.search_terms, spec.year_from, spec.year_to)

    # ----- intent-specific SELECT -----
    if intent == "describe_dataset":
        sql = (
            "SELECT publication_year, COUNT(*) AS n_papers "
            "FROM articles_cast "
            "GROUP BY publication_year "
            "ORDER BY publication_year"
        )

    elif intent == "trend_over_time":
        sql = (
            "SELECT publication_year, COUNT(*) AS n_papers "
            "FROM articles_cast"
            f"{where_sql} "
            "GROUP BY publication_year "
            "ORDER BY publication_year"
        )

    elif intent == "compare_groups":
        # Для сравнения групп используем только исходные research_topics,
        # чтобы группы были интерпретируемы, и не было привязки к длинным
        # каноническим названиям.
        base_topics: list[str] = []
        if clf is not None:
            base_topics = _normalize_terms(list(clf.research_topics or []), max_terms=4)

        # Если тем < 2 — деградируем до trend_over_time.
        if len(base_topics) < 2:
            sql = (
                "SELECT publication_year, COUNT(*) AS n_papers "
                "FROM articles_cast"
                f"{where_sql} "
                "GROUP BY publication_year "
                "ORDER BY publication_year"
            )
        else:
            # For compare_groups we fully control the text filter via grp_patterns.
            # The generic `patterns` from _build_where (built from clf.search_terms)
            # may not match the group patterns and would be misleading if printed
            # or reused for exports.
            params.pop("patterns", None)

            # CASE-метка группы
            # Важно: для каждой группы используем НЕ один шаблон, а небольшое множество
            # вариантов (например, nlp -> nlp OR natural language processing), чтобы
            # не терять попадания в базе.
            hay = _haystack_expr()
            case_lines: list[str] = ["CASE"]
            all_grp_patterns: list[str] = []
            for i, t in enumerate(base_topics):
                params[f"grp_lbl_{i}"] = t

                variants = _normalize_terms([t], max_terms=6)
                like_patterns = [f"%{v.lower()}%" for v in variants]
                params[f"grp_pats_{i}"] = like_patterns
                all_grp_patterns.extend(like_patterns)

                case_lines.append(
                    f"  WHEN {hay} LIKE ANY(%(grp_pats_{i})s) THEN %(grp_lbl_{i})s"
                )
            case_lines.append("  ELSE 'other'")
            case_lines.append("END AS group_topic")
            case_expr = "\n".join(case_lines)

            # Фильтруем по объединению шаблонов всех групп (иначе CASE даёт 'other').
            # (Дедуп — чтобы не раздувать массив параметров.)
            seen_p: set[str] = set()
            grp_patterns: list[str] = []
            for p in all_grp_patterns:
                if p in seen_p:
                    continue
                seen_p.add(p)
                grp_patterns.append(p)
            params["grp_patterns"] = grp_patterns

            # Важно: year-фильтр оставляем из where_sql (если был).
            # А текстовый фильтр заменяем на базовый (grp_patterns), чтобы совпадало с CASE.
            where_parts: list[str] = []
            if spec.year_from is not None and spec.year_to is not None:
                if spec.year_from == spec.year_to:
                    where_parts.append("publication_year = %(year_from)s")
                else:
                    where_parts.append(
                        "publication_year BETWEEN %(year_from)s AND %(year_to)s"
                    )
            where_parts.append(f"{hay} LIKE ANY(%(grp_patterns)s)")
            where_cmp = " WHERE " + " AND ".join(where_parts)

            sql = (
                "SELECT publication_year,\n"
                f"{case_expr},\n"
                "COUNT(*) AS n_papers\n"
                "FROM articles_cast"
                f"{where_cmp}\n"
                "GROUP BY publication_year, group_topic\n"
                "ORDER BY publication_year, group_topic"
            )

    elif intent == "correlation_or_relationship":
        sql = (
            "SELECT publication_year, cited_by_count "
            "FROM articles_cast"
            f"{where_sql} "
            "AND cited_by_count IS NOT NULL"
            if where_sql
            else "SELECT publication_year, cited_by_count FROM articles_cast WHERE cited_by_count IS NOT NULL"
        )

    elif intent == "top_journals":
        # Top journals (sources) by number of matched works.
        # Note: journal may be NULL/empty in local snapshot.
        limit = int(spec.limit or 20)
        if where_sql:
            where_j = f"{where_sql} AND journal IS NOT NULL AND journal <> ''"
        else:
            where_j = " WHERE journal IS NOT NULL AND journal <> ''"

        sql = (
            "SELECT journal, COUNT(*) AS n_papers "
            "FROM articles_cast"
            f"{where_j} "
            "GROUP BY journal "
            "ORDER BY n_papers DESC, journal "
            "LIMIT %(limit)s"
        )
        params["limit"] = limit

    elif intent == "raw_table_view":
        sql = (
            "SELECT doi, title, publication_year, cited_by_count, journal, topic "
            "FROM articles_cast"
            f"{where_sql} "
            "ORDER BY cited_by_count DESC NULLS LAST, publication_year DESC NULLS LAST "
            "LIMIT %(limit)s"
        )
        params["limit"] = int(spec.limit or 100)

    else:
        # Безопасный дефолт
        sql = (
            "SELECT publication_year, COUNT(*) AS n_papers "
            "FROM articles_cast"
            f"{where_sql} "
            "GROUP BY publication_year "
            "ORDER BY publication_year"
        )

    # финальная проверка безопасности
    if not is_safe_sql(sql):
        raise ValueError(f"Unsafe SQL compiled: {sql}")

    used_openalex_topics = bool(clf and getattr(clf, "categories", None))
    openalex_topics_payload = None
    if used_openalex_topics:
        # Храним компактный «слепок» привязки, полезный для отладки.
        openalex_topics_payload = [
            {
                "openalex_id": getattr(c, "openalex_id", ""),
                "domain": c.domain,
                "field": c.field,
                "subfield": c.subfield,
                "topic": c.topic,
            }
            for c in (clf.categories or [])
        ]

    return CollectorOutput(
        sql=sql,
        params=params or None,
        data_source="sql",
        used_openalex_topics=used_openalex_topics,
        openalex_topics_payload=openalex_topics_payload,
        openalex_data_meta=None,
    )


# -------------------- Public API --------------------


def generate_sql(plan: PlannerPlan, clf: ClassifierResult | None = None) -> CollectorOutput:
    """Собирает QuerySpec и компилирует безопасный SQL."""
    spec = build_query_spec(plan, clf)
    return compile_sql(spec, clf)


def execute_sql(sql: str, params: Dict[str, Any] | None = None) -> list[Dict[str, Any]]:
    """Выполняет SQL против PostgreSQL."""
    return run_sql(sql, params)


# -------------------- OpenAlex data collection (optional) --------------------


def _openalex_supported(intent: str) -> bool:
    """Какие intent мы умеем собирать напрямую из OpenAlex."""
    return intent in {
        "describe_dataset",
        "trend_over_time",
        "compare_groups",
        "top_journals",
        "raw_table_view",
    }


def _topic_id_for_base_query(clf: ClassifierResult | None, base_query: str) -> str | None:
    """Пытается достать OpenAlex Topic ID (URL) для конкретной базовой темы.

    В Classifier.openalex_selection_log мы храним якорь/выборку для каждой base_query.
    Берём якорь (anchor.openalex_id), иначе — первый selected.
    """
    if not clf or not base_query:
        return None

    log = getattr(clf, "openalex_selection_log", None) or []
    for item in log:
        if (item or {}).get("base_query") != base_query:
            continue
        anchor = (item or {}).get("anchor") or {}
        if anchor.get("openalex_id"):
            return str(anchor.get("openalex_id"))
        sel = (item or {}).get("selected") or []
        if sel and isinstance(sel, list) and (sel[0] or {}).get("openalex_id"):
            return str((sel[0] or {}).get("openalex_id"))
        break

    # Фолбэк: если selection_log по какой-то причине пуст — берём любой topic id.
    cats = getattr(clf, "categories", None) or []
    for c in cats:
        tid = getattr(c, "openalex_id", None)
        if tid:
            return str(tid)
    return None


def _openalex_filters(
    *,
    topic_id: str | None,
    year_from: int | None,
    year_to: int | None,
) -> str | None:
    parts: list[str] = []
    if topic_id:
        # Works have `topics` list; filtering by topics.id is usually less strict than primary_topic.
        parts.append(f"topics.id:{topic_id}")
    if year_from is not None and year_to is not None:
        if year_from == year_to:
            parts.append(f"publication_year:{int(year_from)}")
        else:
            parts.append(f"publication_year:{int(year_from)}-{int(year_to)}")
    return ",".join(parts) if parts else None


def _openalex_group_by_year(
    *,
    topic_id: str | None,
    search: str | None,
    year_from: int | None,
    year_to: int | None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Возвращает rows формата [{publication_year, n_papers}] из OpenAlex."""
    filters = _openalex_filters(topic_id=topic_id, year_from=year_from, year_to=year_to)
    meta: Dict[str, Any] = {
        "group_by": "publication_year",
        "filters": filters,
        "search": search,
    }
    payload = group_works(
        "publication_year",
        filters=filters,
        search=search,
        per_page=200,
        email=settings.openalex_mailto,
    )
    meta["meta"] = payload.get("meta")

    rows: list[Dict[str, Any]] = []
    for g in payload.get("group_by") or []:
        key = g.get("key")
        try:
            year = int(str(key))
        except Exception:
            continue
        rows.append({"publication_year": year, "n_papers": int(g.get("count") or 0)})

    rows.sort(key=lambda r: r.get("publication_year") or 0)
    return rows, meta


def _openalex_group_by_journal(
    *,
    topic_id: str | None,
    search: str | None,
    year_from: int | None,
    year_to: int | None,
    limit: int = 20,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Возвращает rows формата [{journal, journal_openalex_id, n_papers}] из OpenAlex."""
    filters = _openalex_filters(topic_id=topic_id, year_from=year_from, year_to=year_to)
    meta: Dict[str, Any] = {
        "group_by": "primary_location.source.id",
        "filters": filters,
        "search": search,
    }
    payload = group_works(
        "primary_location.source.id",
        filters=filters,
        search=search,
        per_page=200,
        email=settings.openalex_mailto,
    )
    meta["meta"] = payload.get("meta")

    groups = list(payload.get("group_by") or [])
    # Make ordering deterministic: sort by count desc, then name.
    groups.sort(key=lambda g: (-(int(g.get("count") or 0)), str(g.get("key_display_name") or "")))

    rows: list[Dict[str, Any]] = []
    for g in groups[: max(1, int(limit or 20))]:
        name = str(g.get("key_display_name") or "").strip()
        if not name:
            continue
        rows.append(
            {
                "journal": name,
                "journal_openalex_id": g.get("key"),
                "n_papers": int(g.get("count") or 0),
            }
        )
    return rows, meta


def _openalex_raw_table_view(
    *,
    topic_id: str | None,
    search: str | None,
    year_from: int | None,
    year_to: int | None,
    limit: int = 100,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Возвращает «сырую таблицу» работ из OpenAlex.

    Формат строк стараемся приблизить к SQL raw_table_view:
      - doi, title, publication_year, cited_by_count, journal, topic

    Примечание: в OpenAlex journal берём из primary_location.source.display_name,
    а topic — из primary_topic.display_name (это может отличаться от локального снапшота).
    """

    lim = max(1, min(int(limit or 100), 200))
    filters = _openalex_filters(topic_id=topic_id, year_from=year_from, year_to=year_to)
    meta: Dict[str, Any] = {
        "mode": "raw_table_view",
        "filters": filters,
        "search": search,
        "sort": "cited_by_count:desc",
        # select supports only root-level fields
        "select": "id,title,doi,publication_year,cited_by_count,primary_location,primary_topic",
        "per_page": lim,
        "page": 1,
    }

    payload = list_works(
        filters=filters,
        search=search,
        sort="cited_by_count:desc",
        select=meta["select"],
        per_page=lim,
        page=1,
        email=settings.openalex_mailto,
    )
    meta["meta"] = payload.get("meta")

    rows: list[Dict[str, Any]] = []
    for w in payload.get("results") or []:
        # DOI in OpenAlex may be a URL like "https://doi.org/..."
        doi = w.get("doi")
        if isinstance(doi, str) and doi.lower().startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/") :]

        pl = (w.get("primary_location") or {})
        src = (pl.get("source") or {}) if isinstance(pl, dict) else {}
        journal = None
        journal_id = None
        if isinstance(src, dict):
            journal = (src.get("display_name") or "").strip() or None
            journal_id = src.get("id")

        pt = (w.get("primary_topic") or {})
        topic = None
        if isinstance(pt, dict):
            topic = (pt.get("display_name") or "").strip() or None

        rows.append(
            {
                "openalex_work_id": w.get("id"),
                "doi": doi,
                "title": w.get("title") or None,
                "publication_year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count"),
                "journal": journal,
                "journal_openalex_id": journal_id,
                "topic": topic,
            }
        )

    # Ensure deterministic order
    rows.sort(
        key=lambda r: (
            -(int(r.get("cited_by_count") or 0)),
            -(int(r.get("publication_year") or 0)),
            str(r.get("title") or ""),
        )
    )
    return rows, meta


def _collect_openalex(spec: QuerySpec, clf: ClassifierResult | None) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Строит результат через OpenAlex в зависимости от intent."""
    intent = spec.intent or "other"

    if intent == "describe_dataset":
        return _openalex_group_by_year(
            topic_id=None,
            search=None,
            year_from=spec.year_from,
            year_to=spec.year_to,
        )

    if intent == "trend_over_time":
        base_q = None
        if clf and (clf.research_topics or []):
            base_q = str(clf.research_topics[0])
        topic_id = _topic_id_for_base_query(clf, base_q or "") if base_q else None
        # Если нет topic_id — используем search по базовому термину.
        search = None if topic_id else (base_q or " ".join(spec.search_terms[:2]) or None)
        return _openalex_group_by_year(
            topic_id=topic_id,
            search=search,
            year_from=spec.year_from,
            year_to=spec.year_to,
        )

    if intent == "compare_groups":
        base_topics: list[str] = []
        if clf is not None:
            base_topics = _normalize_terms(list(clf.research_topics or []), max_terms=4)

        # Если тем < 2 — деградируем до trend_over_time.
        if len(base_topics) < 2:
            return _collect_openalex(QuerySpec(**spec.model_dump(), intent="trend_over_time"), clf)

        out_rows: list[Dict[str, Any]] = []
        meta: Dict[str, Any] = {"mode": "compare_groups", "groups": []}

        for t in base_topics:
            topic_id = _topic_id_for_base_query(clf, t)
            search = None if topic_id else t
            rows, m = _openalex_group_by_year(
                topic_id=topic_id,
                search=search,
                year_from=spec.year_from,
                year_to=spec.year_to,
            )
            meta["groups"].append({"topic": t, "topic_id": topic_id, "request": m})
            for r in rows:
                out_rows.append(
                    {
                        "publication_year": r.get("publication_year"),
                        "group_topic": t,
                        "n_papers": r.get("n_papers"),
                    }
                )

        out_rows.sort(key=lambda r: (r.get("publication_year") or 0, str(r.get("group_topic") or "")))
        return out_rows, meta

    if intent == "top_journals":
        base_q = None
        if clf and (clf.research_topics or []):
            base_q = str(clf.research_topics[0])
        topic_id = _topic_id_for_base_query(clf, base_q or "") if base_q else None
        search = None if topic_id else (base_q or " ".join(spec.search_terms[:2]) or None)
        return _openalex_group_by_journal(
            topic_id=topic_id,
            search=search,
            year_from=spec.year_from,
            year_to=spec.year_to,
            limit=spec.limit or 20,
        )

    if intent == "raw_table_view":
        base_q = None
        if clf and (clf.research_topics or []):
            base_q = str(clf.research_topics[0])

        topic_id = _topic_id_for_base_query(clf, base_q or "") if base_q else None
        search = None if topic_id else (base_q or " ".join(spec.search_terms[:2]) or None)
        return _openalex_raw_table_view(
            topic_id=topic_id,
            search=search,
            year_from=spec.year_from,
            year_to=spec.year_to,
            limit=int(spec.limit or 100),
        )

    return [], {"error": f"OpenAlex collector: intent '{intent}' is not supported"}


def collect_data(
    plan: PlannerPlan,
    clf: ClassifierResult | None = None,
    *,
    mode_override: str | None = None,
) -> tuple[CollectorOutput, list[Dict[str, Any]]]:
    """Главная точка входа Collector.

    Collector умеет работать в режимах (mode_override):
      - sql: только локальная БД
      - openalex: только OpenAlex (для поддерживаемых intent)
      - both: собрать данные и из SQL, и из OpenAlex (если intent поддерживается)

    Для correlation_or_relationship мы всегда используем SQL.
    Для raw_table_view OpenAlex доступен только при явном запросе пользователя.
    """
    spec = build_query_spec(plan, clf)
    intent = spec.intent or "other"

    # По умолчанию — строго SQL. Обращение к OpenAlex допускается
    # только при явном запросе пользователя (mode_override).
    mode = (mode_override or "sql").strip().lower()
    if intent in {"correlation_or_relationship"}:
        mode = "sql"

    if mode not in {"sql", "openalex", "both"}:
        mode = "sql"

    # --- mode: both (collect from SQL and OpenAlex) ---
    if mode == "both" and _openalex_supported(intent):
        sql_out = compile_sql(spec, clf)

        rows_sql: list[Dict[str, Any]] = []
        sql_err: str | None = None
        try:
            if sql_out.sql:
                rows_sql = execute_sql(sql_out.sql, sql_out.params)
        except Exception as e:
            sql_err = f"{type(e).__name__}: {e}"
            rows_sql = []

        try:
            rows_oa, meta_oa = _collect_openalex(spec, clf)
        except Exception as e:
            rows_oa, meta_oa = [], {"error": f"{type(e).__name__}: {e}"}

        # Normalize rows so that Analyst can render them in one pass.
        combined: list[Dict[str, Any]] = []

        if intent in {"describe_dataset", "trend_over_time"}:
            for r in rows_sql:
                rr = dict(r)
                rr["source"] = "sql"
                combined.append(rr)
            for r in rows_oa:
                rr = dict(r)
                rr["source"] = "openalex"
                combined.append(rr)

            # Sort deterministically (year then source)
            combined.sort(key=lambda x: (x.get("publication_year") or 0, str(x.get("source") or "")))

        elif intent == "compare_groups":
            # compare_groups already uses `group_topic` as a series key. If we simply add
            # `source`, the plotter would merge points by group_topic. Instead, we make
            # group_topic unique per source.
            for r in rows_sql:
                rr = dict(r)
                if rr.get("group_topic") is not None:
                    rr["group_topic"] = f"{rr.get('group_topic')} [SQL]"
                rr["source"] = "sql"
                combined.append(rr)
            for r in rows_oa:
                rr = dict(r)
                if rr.get("group_topic") is not None:
                    rr["group_topic"] = f"{rr.get('group_topic')} [OpenAlex]"
                rr["source"] = "openalex"
                combined.append(rr)
            combined.sort(
                key=lambda x: (
                    x.get("publication_year") or 0,
                    str(x.get("group_topic") or ""),
                    str(x.get("source") or ""),
                )
            )

        elif intent == "top_journals":
            # Make journals unique per source to avoid merging visually and to keep comparisons explicit.
            for r in rows_sql:
                rr = dict(r)
                if rr.get("journal") is not None:
                    rr["journal"] = f"{rr.get('journal')} [SQL]"
                rr["source"] = "sql"
                combined.append(rr)
            for r in rows_oa:
                rr = dict(r)
                if rr.get("journal") is not None:
                    rr["journal"] = f"{rr.get('journal')} [OpenAlex]"
                rr["source"] = "openalex"
                combined.append(rr)
            combined.sort(key=lambda x: (-(int(x.get("n_papers") or 0)), str(x.get("journal") or "")))

        elif intent == "raw_table_view":
            # Просто объединяем две выборки (списки работ) и помечаем источник.
            # Поля могут различаться (OpenAlex добавляет openalex_work_id и т.п.).
            for r in rows_sql:
                rr = dict(r)
                rr["source"] = "sql"
                combined.append(rr)
            for r in rows_oa:
                rr = dict(r)
                rr["source"] = "openalex"
                combined.append(rr)

            combined.sort(
                key=lambda x: (
                    -(int(x.get("cited_by_count") or 0)),
                    -(int(x.get("publication_year") or 0)),
                    str(x.get("title") or ""),
                    str(x.get("source") or ""),
                )
            )

        meta: Dict[str, Any] = {
            "mode": "both",
            "sql": {
                "error": sql_err,
                "sql": sql_out.sql,
                "params": sql_out.params,
                "row_count": len(rows_sql),
            },
            "openalex": {
                "meta": meta_oa,
                "row_count": len(rows_oa),
                "warning": (
                    "OpenAlex возвращает агрегаты по внешней базе и может не совпадать "
                    "по покрытию с локальной БД. Сравнивайте источники как разные выборки."
                ),
            },
        }

        return (
            CollectorOutput(
                sql=sql_out.sql,
                params=sql_out.params,
                data_source="both",
                used_openalex_topics=sql_out.used_openalex_topics,
                openalex_topics_payload=sql_out.openalex_topics_payload,
                openalex_data_meta=meta,
            ),
            combined,
        )

    # --- mode: openalex (explicit user request) ---
    if mode == "openalex" and _openalex_supported(intent):
        try:
            rows, meta = _collect_openalex(spec, clf)
        except Exception as e:
            rows, meta = [], {"error": f"{type(e).__name__}: {e}"}

        sql_out = compile_sql(spec, clf)
        return (
            CollectorOutput(
                sql=None,
                params=None,
                data_source="openalex",
                used_openalex_topics=sql_out.used_openalex_topics,
                openalex_topics_payload=sql_out.openalex_topics_payload,
                openalex_data_meta=meta,
            ),
            rows,
        )

    # --- SQL path (default) ---
    sql_out = compile_sql(spec, clf)
    rows_sql: list[Dict[str, Any]] = []
    sql_err: str | None = None
    try:
        if sql_out.sql:
            rows_sql = execute_sql(sql_out.sql, sql_out.params)
    except Exception as e:
        sql_err = f"{type(e).__name__}: {e}"
        rows_sql = []

    # Возвращаем SQL даже если он вернул 0 строк: Analyst умеет корректно отвечать.
    if sql_err:
        sql_out.openalex_data_meta = {"sql_error": sql_err}
    return sql_out, rows_sql
