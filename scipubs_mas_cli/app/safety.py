from __future__ import annotations

import re


DANGEROUS_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def is_safe_sql(sql: str) -> bool:
    """Very simple SQL safety check.

    Allows only SELECT queries and forbids obvious DDL/DML statements.
    """
    sql_strip = sql.strip()
    if not sql_strip.lower().startswith("select"):
        return False
    if DANGEROUS_SQL_PATTERN.search(sql_strip):
        return False
    # Disallow multiple statements separated by semicolon, except optional trailing ;.
    if ";" in sql_strip[:-1]:
        return False
    return True


def sanitize_user_query_for_prompt(query: str) -> str:
    """Remove the most obvious prompt-injection patterns.

    This is intentionally simple: the main protection is in system prompts,
    where we explicitly ask the model to ignore user attempts to change rules.
    """
    # Drop lines that look like meta-instructions.
    lines = []
    for line in query.splitlines():
        if any(
            marker in line.lower()
            for marker in [
                "ignore previous instructions",
                "you are chatgpt",
                "disregard the system prompt",
                "act as",
            ]
        ):
            continue
        lines.append(line)
    return "\n".join(lines)
