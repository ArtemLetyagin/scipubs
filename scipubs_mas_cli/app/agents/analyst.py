from __future__ import annotations

import json
from typing import Dict, Any, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from ..llm import call_llm
from ..tools.plotting import line_chart_base64, bar_chart_base64
from .planner import PlannerPlan


class AnalystResult(BaseModel):
    summary: str = Field(..., description="Текстовый аналитический комментарий.")
    plot_type: str | None = Field(
        None,
        description="Тип построенного графика: line_chart, bar_chart и т.п. (если есть).",
    )
    plot_base64: str | None = Field(
        None,
        description="PNG-график в base64, если он был построен.",
    )


ANALYST_SYSTEM_PROMPT = """Вы — агент Analyst в многоагентной системе анализа научных публикаций.

Вы получаете:
- исходный запрос пользователя;
- структурированный план от Planner'а (intent, инструкции и ожидаемый формат вывода);
- результат SQL-запроса (первые строки таблицы в текстовом виде).

Ваша задача — написать краткий, но содержательный аналитический комментарий:
- Какие тренды видны?
- Как сравниваются группы?
- Есть ли явная зависимость между переменными?
- Если это просто таблица, что в ней самое важное?

Правила:
- Не выдумывайте данные, опирайтесь только на предоставленную выборку.
- Если выборка маленькая, явно укажите, что выводы предварительные.
- Отвечайте коротко (3–7 предложений).
- Отвечайте на русском языке.
"""


analyst_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANALYST_SYSTEM_PROMPT),
        (
            "user",
            "Исходный запрос пользователя:\n{user_query}\n\n"
            "План от Planner'а:\n{planner_plan}\n\n"
            "Первые строки результата SQL-запроса:\n{rows_preview}\n\n"
            "Напиши аналитический комментарий.",
        ),
    ]
)


def _rows_preview(rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
    return json.dumps(rows[:max_rows], ensure_ascii=False, indent=2)


def analyze(
    user_query: str,
    plan: PlannerPlan,
    rows: List[Dict[str, Any]],
) -> AnalystResult:
    """Main entrypoint for Analyst agent."""
    # Deterministic fallback for empty result sets.
    # This prevents the LLM from speculating about reasons ("не проиндексировано" etc.).
    if not rows:
        hints: list[str] = []
        if plan.user_intent == "raw_table_view":
            hints.append(
                "Попробуйте уточнить тему (например, 'machine learning' вместо 'ML') или расширить ключевые слова."
            )
        else:
            hints.append(
                "Попробуйте расширить запрос: добавить синонимы на английском и/или ослабить фильтры по году."
            )
        summary = (
            "SQL-запрос не вернул строк — в локальной базе данных не найдено публикаций, "
            "удовлетворяющих указанным фильтрам (тематика/ключевые слова/год). "
            + " ".join(hints)
        )
        return AnalystResult(summary=summary, plot_type=None, plot_base64=None)

    rows_preview = _rows_preview(rows)
    planner_serialized = plan.model_dump(mode="json")

    messages = analyst_prompt.format_prompt(
        user_query=user_query,
        planner_plan=json.dumps(planner_serialized, ensure_ascii=False, indent=2),
        rows_preview=rows_preview,
    ).to_messages()

    summary = call_llm(messages, temperature=0.3)

    # Try to build a chart if requested
    plot_type: str | None = None
    plot_base64: str | None = None

    if rows and "line_chart" in plan.expected_output:
        # Heuristic: trend over time -> publication_year vs count
        row0 = rows[0]
        y_key = None
        if "n_papers" in row0:
            y_key = "n_papers"
        elif "count" in row0:
            y_key = "count"
        if "publication_year" in row0 and y_key is not None:
            plot_type = "line_chart"
            plot_base64 = line_chart_base64(rows, "publication_year", y_key)

    if rows and plot_base64 is None and "bar_chart" in plan.expected_output:
        # Simple bar chart by topic or domain
        row0 = rows[0]
        cat_key = None
        y_key = None
        for key in ("topic", "domain", "field", "journal"):
            if key in row0:
                cat_key = key
                break
        for key in ("n_papers", "count"):
            if key in row0:
                y_key = key
                break
        if cat_key and y_key:
            plot_type = "bar_chart"
            plot_base64 = bar_chart_base64(rows, cat_key, y_key)

    return AnalystResult(summary=summary.strip(), plot_type=plot_type, plot_base64=plot_base64)
