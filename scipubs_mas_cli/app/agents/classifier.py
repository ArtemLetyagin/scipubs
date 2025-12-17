from __future__ import annotations
import math
import re
"""Classifier agent.

Задача Classifier'а — выделить *предмет поиска* (научные темы) из
переформулированного запроса Planner'а и ПРИВЯЗАТЬ их к каноническим темам
OpenAlex.

Ключевое отличие от прежней версии:
 - мы НЕ делаем генеративное «семантическое расширение» тем, которое
   часто превращало аналитические операции ("по годам", "годовой подсчёт")
   в «ключевые слова» и раздувало WHERE.
 - OpenAlex-линковка является обязательным шагом: пайплайн всегда выполняет
   вызов OpenAlex и использует его результаты как основной источник
   search_terms (с безопасной деградацией на исходные темы, если OpenAlex
   вернул 0 результатов).
 - мы возвращаем компактный список *канонических* терминов для поиска.
"""

import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ConfigDict

from ..llm import call_llm
from ..tools.openalex import search_topics
from .planner import PlannerPlan


# -------------------- Models --------------------


class ResearchTopics(BaseModel):
    """Только предметные научные темы из запроса (без аналитических операций)."""

    topics: List[str] = Field(
        default_factory=list,
        description=(
            "Список 1–3 научных тем/областей/методов (1–6 слов каждая), "
            "напр. 'deep learning', 'graph neural networks'. "
            "Не включать операции анализа (count/trend/by year/annual и т.п.)."
        ),
    )


class CategoryCandidate(BaseModel):
    """Кандидат на пару (domain, field, subfield, topic) из OpenAlex Topics."""
    openalex_id: str = ""
    domain: str = ""
    field: str = ""
    subfield: str = ""
    topic: str = ""
    works_count: int = 0


class ClassifierResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    research_topics: List[str] = Field(default_factory=list)
    categories: List[CategoryCandidate] = Field(default_factory=list)
    search_terms: List[str] = Field(default_factory=list)

    # OpenAlex usage must be explicitly requested by the user.
    openalex_used: bool = False

    openalex_ok: bool = True
    openalex_requests: int = 0
    openalex_total_results: int = 0
    openalex_errors: List[str] = Field(default_factory=list)

    # Debug: как OpenAlex-кандидаты были отобраны в финальные search_terms
    openalex_selection_log: List[dict] = Field(default_factory=list)


# -------------------- Step 1: extract research topics --------------------


_EXTRACT_RESEARCH_TOPICS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Ты выделяешь из запроса пользователя только ПРЕДМЕТ ПОИСКА: "
                "научные темы/области/методы, по которым нужно искать публикации.\n"
                "Важно: верни темы СТРОГО НА АНГЛИЙСКОМ ЯЗЫКЕ, независимо от языка запроса. "
                "Если запрос на русском — переведи темы на английский. "
                "Если встречаются общие аббревиатуры (ML/NLP/CV/RL/LLM/GNN), раскрой их в стандартные английские термины.\n"
                "Если в запросе уже явно написана тема на английском (например, 'machine learning'), "
                "перепиши её ВЕРБАТИМ и НЕ заменяй на более широкие/соседние темы (например, не меняй на 'artificial intelligence').\n"
                "Важно: НЕ включай аналитические операции и требования к отчёту "
                "(например, 'динамика', 'по годам', 'годовой подсчёт', 'trend', "
                "'count', 'compare', 'visualize', 'annual/yearly' и т.п.), "
                "а также ограничения по периоду.\n"
                "Если в запросе несколько научных тем — верни их отдельными элементами." 
                "\nЕсли предмет поиска неочевиден, всё равно верни ОДНУ наиболее вероятную научную тему."
            ),
        ),
        (
            "user",
            (
                "Запрос:\n{query}\n\n"
                "Ответь строго JSON вида:\n"
                "{{\"topics\": [\"topic 1\", \"topic 2\"]}}\n"
                "Никакого текста до или после JSON."
            ),
        ),
    ]
)


# -------------------- Heuristics: preserve explicit English phrases --------------------

# If the user already wrote the topic in English (e.g., "machine learning"),
# we must not "generalize" it into a broader neighbor (e.g., "artificial intelligence").
# The LLM sometimes does that. To prevent it, we extract explicit multi-word
# English phrases from the original query and use them as a fallback when
# the model output does not match anything the user actually wrote.

_EN_MULTIWORD_PHRASE_RE = re.compile(r"\b[a-zA-Z]{2,}(?:[-\s]+[a-zA-Z]{2,}){1,5}\b")
_EN_STOPWORDS = {
    # reporting / analysis verbs
    "build",
    "make",
    "create",
    "plot",
    "draw",
    "show",
    "compare",
    "trend",
    "trends",
    "count",
    "counts",
    "table",
    "chart",
    "graph",
    "year",
    "years",
    "from",
    "to",
    "between",
    "per",
    "by",
    # common non-topical words often present in requests
    "most",
    "top",
    "cited",
    "citation",
    "citations",
    "work",
    "works",
    "paper",
    "papers",
    "publication",
    "publications",
}


def _norm_for_contains(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s


def _extract_explicit_english_phrases(query: str, max_items: int = 3) -> List[str]:
    q = query or ""
    out: list[str] = []
    seen: set[str] = set()
    for m in _EN_MULTIWORD_PHRASE_RE.finditer(q):
        phrase = " ".join(m.group(0).replace("-", " ").split())
        key = phrase.lower()
        if key in seen:
            continue
        tokens = key.split()
        # Filter out phrases that are obviously report/analysis artifacts.
        if any(t in _EN_STOPWORDS for t in tokens):
            continue
        seen.add(key)
        out.append(phrase)
        if len(out) >= max_items:
            break
    return out


def extract_research_topics(query: str) -> List[str]:
    messages = _EXTRACT_RESEARCH_TOPICS_PROMPT.format_prompt(query=query).to_messages()
    raw = call_llm(messages, temperature=0.1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and start < end:
            data = json.loads(raw[start : end + 1])
        else:
            raise

    obj = ResearchTopics.model_validate(data)
    topics: list[str] = []
    for t in obj.topics:
        t = " ".join(str(t).strip().split())
        if t:
            topics.append(t)
    # Убираем дубликаты, сохраняя порядок
    seen: set[str] = set()
    uniq: list[str] = []
    for t in topics:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    # Safety net: иногда модель может вернуть тему не на английском.
    # OpenAlex ожидает англоязычный запрос, поэтому нормализуем в EN.
    if any(re.search(r"[\u0400-\u04FF]", t) for t in uniq):
        uniq = translate_topics_to_english(uniq)

    # If the model returned topics that don't appear in the user's text,
    # but the user explicitly wrote a multi-word English topic, trust the user.
    # Example: query contains "machine learning" but the model outputs "artificial intelligence".
    explicit_en = _extract_explicit_english_phrases(query)
    if explicit_en:
        norm_q = _norm_for_contains(query)
        model_matches_query = any(_norm_for_contains(t) in norm_q for t in uniq)
        if not model_matches_query:
            return explicit_en

    return uniq


_TRANSLATE_TOPICS_TO_EN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Ты переводишь список научных тем на английский язык. "
                "Сохраняй смысл и область. Не добавляй новые темы. "
                "Если тема уже на английском — оставь как есть. "
                "Если тема является общепринятой аббревиатурой (например, ML/NLP), "
                "раскрой её в стандартную английскую формулировку."
            ),
        ),
        (
            "user",
            (
                "Темы (JSON):\n{payload}\n\n"
                "Ответь строго JSON вида:\n"
                "{\"topics\": [\"topic 1\", \"topic 2\"]}\n"
                "Никакого текста до или после JSON."
            ),
        ),
    ]
)


def translate_topics_to_english(topics: List[str]) -> List[str]:
    payload = json.dumps({"topics": list(topics or [])}, ensure_ascii=False)
    messages = _TRANSLATE_TOPICS_TO_EN_PROMPT.format_prompt(payload=payload).to_messages()
    raw = call_llm(messages, temperature=0.0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and start < end:
            data = json.loads(raw[start : end + 1])
        else:
            raise

    obj = ResearchTopics.model_validate(data)
    out: list[str] = []
    for t in obj.topics:
        t = " ".join(str(t).strip().split())
        if t:
            out.append(t)

    # Dedup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
    return uniq


# -------------------- Step 2: OpenAlex canonical topics --------------------

# -------------------- Step 2: OpenAlex canonical topics --------------------


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Простая токенизация для similarity (без внешних библиотек)."""
    return {
        m.group(0).lower()
        for m in _WORD_RE.finditer(text or "")
        if len(m.group(0)) >= 2
    }


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity по токенам (0..1)."""
    sa = _tokenize(a)
    sb = _tokenize(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def fetch_openalex_candidates(
    query: str,
    per_page: int = 15,
) -> tuple[List[CategoryCandidate], bool, str | None]:
    """Загружает кандидатов Topics из OpenAlex.

    Возвращает (candidates, ok, error_message).
    """
    try:
        results = search_topics(query, per_page=per_page)
    except Exception as e:
        return [], False, f"{type(e).__name__}: {e}"

    candidates: List[CategoryCandidate] = []
    for r in results or []:
        dom = r.get("domain") or {}
        fld = r.get("field") or {}
        sub = r.get("subfield") or {}
        candidates.append(
            CategoryCandidate(
                openalex_id=str(r.get("id") or ""),
                domain=str(dom.get("display_name") or ""),
                field=str(fld.get("display_name") or ""),
                subfield=str(sub.get("display_name") or ""),
                topic=str(r.get("display_name") or query),
                works_count=int(r.get("works_count") or 0),
            )
        )
    return candidates, True, None


def _select_openalex_terms(
    base_query: str,
    candidates: List[CategoryCandidate],
    max_terms: int = 6,
    sim_min: float = 0.20,
    domain_sim_min: float = 0.30,
) -> tuple[List[CategoryCandidate], dict]:
    """Отбирает компактный и согласованный с якорем набор тем.

    Логика (без стоп-листов):
    1) anchor = кандидат с максимальной близостью к base_query;
    2) допускаем кандидаты, согласованные с anchor по field/subfield
       (или хотя бы по domain при достаточно высокой similarity);
    3) ранжируем и берём top-K.
    """
    if not candidates:
        return [], {
            "base_query": base_query,
            "anchor": None,
            "selected": [],
            "rejected_sample": [],
        }

    # Broad, short queries like "machine learning" often have many plausible
    # OpenAlex topic candidates across domains/fields. Picking the single
    # closest topic as the anchor can over-narrow the taxonomy (e.g., anchoring
    # on "... in Bioinformatics" and rejecting everything else as mismatch).
    # For such cases we select a more general anchor (high works_count among the
    # top-similarity set) and relax taxonomy filtering.
    broad_query = len(_tokenize(base_query)) <= 2

    sims: list[tuple[float, int, CategoryCandidate]] = [
        (_jaccard(base_query, c.topic), int(c.works_count or 0), c) for c in candidates
    ]

    if broad_query:
        top_sim = sorted(sims, key=lambda x: (x[0], x[1]), reverse=True)[:5]
        if top_sim:
            anchor = max(top_sim, key=lambda x: x[1])[2]
        else:
            anchor = max(candidates, key=lambda c: (0.0, int(c.works_count or 0)))
    else:
        def _anchor_key(c: CategoryCandidate) -> tuple[float, int]:
            return _jaccard(base_query, c.topic), int(c.works_count or 0)

        anchor = max(candidates, key=_anchor_key)
    max_wc = max((c.works_count for c in candidates), default=1) or 1

    scored: list[tuple[float, CategoryCandidate, float, float]] = []
    rejected: list[tuple[float, CategoryCandidate, str]] = []

    for c in candidates:
        sim_q = _jaccard(base_query, c.topic)
        sim_a = _jaccard(anchor.topic, c.topic)

        same_subfield = bool(anchor.subfield and c.subfield and c.subfield == anchor.subfield)
        same_field = bool(anchor.field and c.field and c.field == anchor.field)
        same_domain = bool(anchor.domain and c.domain and c.domain == anchor.domain)

        if broad_query:
            taxonomy_ok = True
        else:
            taxonomy_ok = same_subfield or same_field or (same_domain and sim_q >= domain_sim_min)
        if sim_q < sim_min:
            rejected.append((sim_q, c, f"sim<{sim_min:.2f}"))
            continue
        if not taxonomy_ok:
            rejected.append((sim_q, c, "taxonomy_mismatch"))
            continue

        wc_norm = math.log1p(max(c.works_count, 0)) / math.log1p(max_wc)
        score = 0.65 * sim_q + 0.25 * sim_a + 0.10 * wc_norm
        scored.append((score, c, sim_q, sim_a))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[CategoryCandidate] = []
    seen: set[str] = set()
    for score, c, sim_q, sim_a in scored:
        key = c.topic.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(c)
        if len(selected) >= max_terms:
            break

    rejected.sort(key=lambda x: x[0], reverse=True)
    rejected_sample = [
        {
            "topic": r.topic,
            "domain": r.domain,
            "field": r.field,
            "subfield": r.subfield,
            "works_count": r.works_count,
            "sim": float(sim),
            "reason": reason,
        }
        for sim, r, reason in rejected[:5]
    ]

    log = {
        "base_query": base_query,
        "anchor": {
            "openalex_id": anchor.openalex_id,
            "topic": anchor.topic,
            "domain": anchor.domain,
            "field": anchor.field,
            "subfield": anchor.subfield,
            "works_count": anchor.works_count,
            "sim": float(_jaccard(base_query, anchor.topic)),
        },
        "selected": [
            {
                "openalex_id": s.openalex_id,
                "topic": s.topic,
                "domain": s.domain,
                "field": s.field,
                "subfield": s.subfield,
                "works_count": s.works_count,
                "sim": float(_jaccard(base_query, s.topic)),
            }
            for s in selected
        ],
        "rejected_sample": rejected_sample,
    }
    return selected, log


# -------------------- Main entry --------------------


def classify(plan: PlannerPlan, *, allow_openalex: bool = False) -> ClassifierResult:
    query = plan.question_rewrite or ""
    research_topics = extract_research_topics(query)

    selected_categories: List[CategoryCandidate] = []
    selection_log: List[dict] = []
    selected_terms: List[str] = []

    openalex_requests = 0
    openalex_total_results = 0
    openalex_errors: List[str] = []

    def _fetch_and_select(base_query: str, per_page: int, max_terms: int) -> None:
        nonlocal openalex_requests, openalex_total_results
        openalex_requests += 1
        cand, ok, err = fetch_openalex_candidates(base_query, per_page=per_page)
        if not ok and err:
            openalex_errors.append(f"query='{base_query}': {err}")
            selection_log.append(
                {"base_query": base_query, "error": err, "anchor": None, "selected": [], "rejected_sample": []}
            )
            return

        openalex_total_results += len(cand)
        chosen, log = _select_openalex_terms(base_query, cand, max_terms=max_terms)
        selection_log.append(log)
        selected_categories.extend(chosen)
        for c in chosen:
            if c.topic:
                selected_terms.append(c.topic)

    if allow_openalex:
        if research_topics:
            for t in research_topics:
                # For broad queries (e.g., "machine learning"), keep more anchors.
                # Otherwise we risk over-narrowing to a single niche topic.
                is_broad = len(_tokenize(t)) <= 2
                _fetch_and_select(t, per_page=20 if is_broad else 15, max_terms=5 if is_broad else 3)
        else:
            _fetch_and_select(query, per_page=20, max_terms=5)
            if selected_categories:
                research_topics = [c.topic for c in selected_categories if c.topic][:3]

    # search_terms: «база + якори».
    # Важно: НЕ заменяем исходную тему пользователя (research_topics)
    # на один узкий OpenAlex-якорь; якори лишь дополняют базовый термин.
    seen: set[str] = set()
    search_terms: List[str] = []

    # 1) Сначала добавляем базовые темы (контракт запроса)
    for t in research_topics:
        t = " ".join(str(t).strip().split())
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        search_terms.append(t)
        if len(search_terms) >= 8:
            break

    # 2) Затем добавляем OpenAlex-канонические термины/якоря
    if len(search_terms) < 8:
        for t in selected_terms:
            t = " ".join(str(t).strip().split())
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            search_terms.append(t)
            if len(search_terms) >= 8:
                break

    # fallback: если по какой-то причине не получили вообще ничего
    if not search_terms and query.strip():
        search_terms = [query.strip()]

    return ClassifierResult(
        research_topics=research_topics,
        categories=selected_categories,
        search_terms=search_terms,

        openalex_used=bool(allow_openalex),
        openalex_ok=(len(openalex_errors) == 0),
        openalex_requests=openalex_requests,
        openalex_total_results=openalex_total_results,
        openalex_errors=openalex_errors,
        openalex_selection_log=selection_log,
    )