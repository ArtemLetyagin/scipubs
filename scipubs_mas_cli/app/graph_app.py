from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

from .agents.planner import PlannerPlan, plan as planner_plan
from .agents.collector import collect_data
from .agents.analyst import analyze, AnalystResult
from .agents.classifier import ClassifierResult, classify
from .agents.tool_router import ToolRoutingDecision, route_tools
from .memory import ensure_memory_table, save_interaction


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    # Raw user input (as received from CLI/API). Router may produce a cleaned variant.
    question_raw: str
    # Cleaned user query (routing directives removed); used by downstream agents.
    question: str

    # Router decision (debug/traceability)
    tool_routing: Dict[str, Any]
    planner_plan: PlannerPlan
    classifier_result: ClassifierResult
    sql: str
    sql_params: Dict[str, Any]
    collector_data_source: str
    collector_meta: Any
    rows: List[Dict[str, Any]]
    analyst_result: AnalystResult

    # Clarification flow (when Planner couldn't determine intent)
    needs_clarification: bool
    clarification_prompt: str

    # Long-term memory write status (best-effort)
    memory_saved: bool
    memory_error: str

    # Loop control
    attempt: int
    max_attempts: int
    feedback: str
    retry: bool

    # Per-request override for collector data sources (sql/openalex/both)
    collector_mode: str

    # OpenAlex usage is disabled by default and must be explicitly requested by the user.
    openalex_allowed: bool


def tool_router_node(state: AgentState) -> AgentState:
    """Router node: decide which tools/data sources may be used for this request.

    The router is part of the MAS (not the CLI). It parses explicit user requests
    for OpenAlex (directives or NL phrasing) and sets:
      - question_raw / question (cleaned)
      - openalex_allowed
      - collector_mode
      - tool_routing (debug)

    Optional overrides (for testing) may be provided via state keys
    `collector_mode` and/or `openalex_allowed`.
    """
    raw = (state.get("question_raw") or state.get("question") or "").strip()

    decision: ToolRoutingDecision = route_tools(raw)

    # Optional per-run overrides (used in tests/dev; CLI normally won't set these).
    mode = decision.collector_mode
    oa_allowed = decision.openalex_allowed

    if "collector_mode" in state and state.get("collector_mode"):
        mode = str(state.get("collector_mode") or "sql").strip().lower()  # type: ignore[assignment]

    if "openalex_allowed" in state and state.get("openalex_allowed") is not None:
        oa_allowed = bool(state.get("openalex_allowed"))

    # Keep the two flags consistent.
    if not oa_allowed:
        mode = "sql"
    if mode not in {"sql", "openalex", "both"}:
        mode = "sql"

    return {
        "question_raw": raw,
        "question": decision.cleaned_query,
        "collector_mode": mode,
        "openalex_allowed": oa_allowed,
        "tool_routing": decision.model_dump(mode="json"),
    }


def planner_node(state: AgentState) -> AgentState:
    user_query = state["question"]
    plan = planner_plan(user_query, feedback=state.get("feedback"), attempt=state.get("attempt", 1))

    if getattr(plan, "user_intent", None) == "other":
        clarification_prompt = (
            "Я не смог однозначно определить тип задачи и предмет поиска. "
            "Пожалуйста, переформулируйте запрос: укажите научную тему/область, "
            "что именно нужно посчитать/сравнить и (если важно) период.\n\n"
            "Примеры:\n"
            "- 'Построй динамику числа публикаций по теме deep learning с 2010 по 2024'\n"
            "- 'Сравни число публикаций по computer vision и NLP по годам за 2015–2024'\n"
            "- 'Покажи таблицу статей по теме graph neural networks за 2020–2024'"
        )
        return {
            "planner_plan": plan,
            "needs_clarification": True,
            "clarification_prompt": clarification_prompt,
        }

    return {"planner_plan": plan, "needs_clarification": False}


def clarify_node(state: AgentState) -> AgentState:
    """Stop the pipeline early and return a clarification question to the user."""
    # No-op: planner_node already prepared the prompt.
    return {}


def classifier_node(state: AgentState) -> AgentState:
    """Агент Classifier: выделяет предметные темы и (опционально) канонизирует их через OpenAlex.

    Важно: на этом шаге мы НЕ дописываем в plan.sql_instruction список ключевых слов.
    Collector получает структурированные search_terms и строит SQL детерминированно.
    """
    plan = state["planner_plan"]
    clf_result = classify(plan, allow_openalex=bool(state.get("openalex_allowed")))
    return {"classifier_result": clf_result}

def collector_node(state: AgentState) -> AgentState:
    plan = state["planner_plan"]
    clf = state.get("classifier_result")
    mode = state.get("collector_mode")
    # Safety net: if the user explicitly allowed OpenAlex but mode wasn't set,
    # interpret it as an explicit OpenAlex request.
    if not mode and state.get("openalex_allowed"):
        mode = "openalex"

    collector_out, rows = collect_data(plan, clf, mode_override=mode)

    out: AgentState = {
        "sql": collector_out.sql,
        "rows": rows,
        "collector_data_source": collector_out.data_source,
    }
    if collector_out.params:
        out["sql_params"] = collector_out.params
    if collector_out.openalex_data_meta is not None:
        out["collector_meta"] = collector_out.openalex_data_meta
    return out


def analyst_node(state: AgentState) -> AgentState:
    user_query = state["question"]
    plan = state["planner_plan"]
    rows = state.get("rows", [])
    result = analyze(user_query, plan, rows)
    return {"analyst_result": result}



def reflect_node(state: AgentState) -> AgentState:
    """Decide whether we should retry with a revised plan (simple reasoning loop)."""
    attempt = int(state.get("attempt", 1) or 1)
    max_attempts = int(state.get("max_attempts", 2) or 2)

    rows = state.get("rows") or []
    clf = state.get("classifier_result")
    openalex_errors = []
    try:
        if clf and getattr(clf, "openalex_errors", None):
            openalex_errors = list(clf.openalex_errors)
    except Exception:
        openalex_errors = []

    # Basic heuristic: retry once (or up to max_attempts) if we got no data
    if (not rows) and attempt < max_attempts:
        src = (state.get("collector_data_source") or "sql").upper()
        fb_parts = [
            f"Предыдущая попытка вернула 0 строк (источник данных: {src}).",
            "Сделайте запрос менее строгим: упростите формулировку, сократите/обобщите темы, "
            "уберите или расширьте временной диапазон, если он был задан.",
        ]
        if openalex_errors:
            fb_parts.append(
                "Также были ошибки при обращении к OpenAlex; при необходимости сформулируйте темы так, "
                "чтобы классификация могла работать даже без внешнего API."
            )

        feedback = " ".join(fb_parts)

        # Clear previous attempt artifacts to avoid confusion
        return {
            "retry": True,
            "attempt": attempt + 1,
            "feedback": feedback,
            "classifier_result": None,
            "sql": None,
            "sql_params": None,
            "rows": [],
            "analyst_result": None,
        }

    return {"retry": False}


def memory_node(state: AgentState) -> AgentState:
    """Persist the interaction into long-term memory (best-effort)."""
    try:
        # Store the raw user query (with directives) if available, for traceability.
        user_query = state.get("question_raw") or state.get("question", "")

        plan_obj = state.get("planner_plan")
        if plan_obj is None:
            plan_dict = None
        elif hasattr(plan_obj, "model_dump"):
            plan_dict = plan_obj.model_dump(mode="json")
        elif isinstance(plan_obj, dict):
            plan_dict = plan_obj
        else:
            plan_dict = None

        sql = state.get("sql")
        rows = state.get("rows") or []

        analyst_res = state.get("analyst_result")
        analysis = getattr(analyst_res, "summary", None) if analyst_res else None

        save_interaction(
            user_query=user_query,
            planner_plan=plan_dict,
            sql=sql,
            row_count=len(rows),
            analysis=analysis,
        )
        return {"memory_saved": True}
    except Exception as e:
        return {"memory_saved": False, "memory_error": f"{type(e).__name__}: {e}"}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("tool_router", tool_router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("collector", collector_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("memory", memory_node)

    graph.set_entry_point("tool_router")

    # Routing decision must happen before any LLM steps.
    graph.add_edge("tool_router", "planner")

    def _route_after_planner(state: AgentState) -> str:
        return "clarify" if state.get("needs_clarification") else "classifier"

    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"clarify": "clarify", "classifier": "classifier"},
    )

    graph.add_edge("clarify", "memory")
    graph.add_edge("classifier", "collector")
    graph.add_edge("collector", "analyst")
    graph.add_edge("analyst", "reflect")

    def _route_after_reflect(state: AgentState) -> str:
        return "planner" if state.get("retry") else "memory"

    graph.add_conditional_edges("reflect", _route_after_reflect, {"planner": "planner", "memory": "memory"})
    graph.add_edge("memory", END)

    return graph.compile()



def run_pipeline(
    question: str,
    *,
    collector_mode: str | None = None,
    openalex_allowed: bool | None = None,
) -> AgentState:
    
    graph_app = build_graph()
    """Convenience wrapper to run the full Planner → Collector → Analyst pipeline."""
    # Idempotent: safe to call on every run
    ensure_memory_table()
    payload: AgentState = {
        "question": question,
        "attempt": 1,
        "max_attempts": 2,
        "feedback": "",
    }
    # Optional overrides for testing/dev. In normal CLI usage, Tool Router decides.
    if openalex_allowed is not None:
        payload["openalex_allowed"] = bool(openalex_allowed)
    if collector_mode:
        payload["collector_mode"] = str(collector_mode)
    return graph_app.invoke(payload)


def run_pipeline_with_openalex(
    question: str,
    *,
    collector_mode: str | None = None,
    openalex_allowed: bool = False,
) -> AgentState:
    """Wrapper that allows explicitly enabling OpenAlex for this request only."""
    ensure_memory_table()
    payload: AgentState = {
        "question": question,
        "attempt": 1,
        "max_attempts": 2,
        "feedback": "",
        "openalex_allowed": bool(openalex_allowed),
    }
    if collector_mode:
        payload["collector_mode"] = str(collector_mode)
    return graph_app.invoke(payload)
