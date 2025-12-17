from __future__ import annotations

"""Tool Router agent.

This node is responsible for *dynamic tool choice* inside the MAS.

Key policy (as requested by the user of this repo):
- Default to local SQL (Postgres) for all requests.
- Use OpenAlex ONLY when the user explicitly asks in the *current* request.

Explicit request formats supported:
- Directives (recommended):
    @openalex ...
    @both ...
    @sql ...
  and bracket form:
    [source=openalex] ...
    [source=both] ...
    [source=sql] ...

- Natural-language explicit requests that contain the token "OpenAlex" together
  with a clear action verb, e.g. "используй OpenAlex", "via OpenAlex".
  (Plain mention of the word "OpenAlex" without an action verb does NOT enable it.)
"""

import re
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict


CollectorMode = Literal["sql", "openalex", "both"]


class ToolRoutingDecision(BaseModel):
    """Structured routing decision produced by the Tool Router node."""

    model_config = ConfigDict(extra="ignore")

    cleaned_query: str = Field(default="", description="Query after stripping routing directives.")
    collector_mode: CollectorMode = Field(default="sql", description="Which data source(s) Collector should use.")
    openalex_allowed: bool = Field(default=False, description="Whether agents are allowed to call OpenAlex.")

    # Debug/traceability
    request_style: Optional[Literal["directive", "nl", "none"]] = None
    reason: str = ""


_DIRECTIVE_RE = re.compile(r"^@(?P<mode>openalex|both|sql)\b\s*", flags=re.IGNORECASE)
_BRACKET_RE = re.compile(r"^\[\s*source\s*=\s*(?P<mode>openalex|both|sql)\s*\]\s*", flags=re.IGNORECASE)


def _strip_openalex_wording(text: str) -> str:
    """Remove explicit OpenAlex instructions so they don't pollute topic extraction."""
    t = text or ""
    # Action + OpenAlex patterns (RU)
    t = re.sub(
        r"\b(используй|использовать|обратись|обращайся|сходи|зайди|получи|выгрузи|подгрузи|по\s+данным|через)\s+openalex\b",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    # Action + OpenAlex patterns (EN)
    t = re.sub(r"\b(use|using|via|from)\s+openalex\b", " ", t, flags=re.IGNORECASE)
    # Remove standalone OpenAlex tokens
    t = re.sub(r"\bopenalex\b", " ", t, flags=re.IGNORECASE)
    return " ".join(t.split())


def _parse_directive(query: str) -> Tuple[str, Optional[CollectorMode]]:
    q = (query or "").strip()
    if not q:
        return q, None

    m = _DIRECTIVE_RE.match(q)
    if m:
        mode = m.group("mode").lower()
        return q[m.end() :].lstrip(), mode  # type: ignore[return-value]

    m = _BRACKET_RE.match(q)
    if m:
        mode = m.group("mode").lower()
        return q[m.end() :].lstrip(), mode  # type: ignore[return-value]

    return q, None


# NL explicit request detection: must contain the token "openalex" + an action verb.
_HAS_OPENALEX_RE = re.compile(r"\bopenalex\b", flags=re.IGNORECASE)
_ACTION_VERB_RE = re.compile(
    r"\b(используй|использовать|обратись|обращайся|сходи|зайди|получи|выгрузи|подгрузи|через|по\s+данным|use|using|via|from)\b",
    flags=re.IGNORECASE,
)

_BOTH_HINT_RE = re.compile(
    r"\b(both|compare|vs|versus|сравн\w+|оба\w*|дв(а|ух)\s+источн\w+|и\s+в\s+базе\w*\s+и\s+в)\b",
    flags=re.IGNORECASE,
)


def _detect_openalex_nl(query: str) -> Optional[CollectorMode]:
    q = (query or "").strip()
    if not q:
        return None

    if not _HAS_OPENALEX_RE.search(q):
        return None

    # Require a clear action verb, otherwise treat as a mere mention.
    if not _ACTION_VERB_RE.search(q):
        return None

    # If the user explicitly asks to compare or use both sources.
    if _BOTH_HINT_RE.search(q):
        return "both"
    return "openalex"


def route_tools(user_query: str) -> ToolRoutingDecision:
    """Compute routing decision for a single user request."""

    raw = (user_query or "").strip()
    if not raw:
        return ToolRoutingDecision(cleaned_query="", collector_mode="sql", openalex_allowed=False, request_style="none")

    cleaned, directive_mode = _parse_directive(raw)
    if directive_mode:
        mode: CollectorMode = directive_mode
        if mode in {"openalex", "both"}:
            return ToolRoutingDecision(
                cleaned_query=_strip_openalex_wording(cleaned),
                collector_mode=mode,
                openalex_allowed=True,
                request_style="directive",
                reason=f"explicit directive: {mode}",
            )
        return ToolRoutingDecision(
            cleaned_query=cleaned,
            collector_mode="sql",
            openalex_allowed=False,
            request_style="directive",
            reason="explicit directive: sql",
        )

    nl_mode = _detect_openalex_nl(raw)
    if nl_mode:
        return ToolRoutingDecision(
            cleaned_query=_strip_openalex_wording(raw),
            collector_mode=nl_mode,
            openalex_allowed=True,
            request_style="nl",
            reason=f"explicit NL request: {nl_mode}",
        )

    return ToolRoutingDecision(
        cleaned_query=raw,
        collector_mode="sql",
        openalex_allowed=False,
        request_style="none",
        reason="default: sql",
    )
