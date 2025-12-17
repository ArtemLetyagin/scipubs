from __future__ import annotations

from typing import Any, List, Dict

from psycopg2.extras import Json

from .db import get_connection


def ensure_memory_table() -> None:
    """Create simple long-term memory table in PostgreSQL if it does not exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                '''
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    user_query TEXT,
                    planner_plan JSONB,
                    sql TEXT,
                    row_count INT,
                    analysis TEXT
                );
                '''
            )

            # Helpful index for fast retrieval of recent items
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_memory_created_at
                ON agent_memory (created_at DESC);
                """
            )
        conn.commit()
    finally:
        conn.close()


def save_interaction(
    user_query: str,
    planner_plan: Dict[str, Any] | None,
    sql: str | None,
    row_count: int | None,
    analysis: str | None,
) -> None:
    """Persist one interaction into long-term memory."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                '''
                INSERT INTO agent_memory (user_query, planner_plan, sql, row_count, analysis)
                VALUES (%s, %s, %s, %s, %s);
                ''',
                (
                    user_query,
                    Json(planner_plan) if planner_plan is not None else None,
                    sql,
                    row_count,
                    analysis,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def load_recent(limit: int = 5) -> List[Dict[str, Any]]:
    """Load N most recent interactions from memory."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT id, created_at, user_query, planner_plan, sql, row_count, analysis
                FROM agent_memory
                ORDER BY created_at DESC
                LIMIT %s;
                ''',
                (limit,),
            )
            rows = cur.fetchall()
        conn.commit()
        result: List[Dict[str, Any]] = []
        for r in rows:
            result.append(
                {
                    "id": r[0],
                    "created_at": r[1],
                    "user_query": r[2],
                    "planner_plan": r[3],
                    "sql": r[4],
                    "row_count": r[5],
                    "analysis": r[6],
                }
            )
        return result
    finally:
        conn.close()


def load_relevant(user_query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Naive keyword-based retrieval (no vector DB) from long-term memory.

    This keeps the project reproducible without extra infra, while still providing
    meaningful retrieval beyond "last N".
    """
    q = (user_query or "").strip().lower()
    # Very simple tokenization: keep reasonably informative tokens
    tokens = [t for t in q.replace("/", " ").replace("-", " ").split() if len(t) >= 4]
    tokens = tokens[:8]
    if not tokens:
        return load_recent(limit=limit)

    patterns = [f"%{t}%" for t in tokens]

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, user_query, planner_plan, sql, row_count, analysis
                FROM agent_memory
                WHERE lower(coalesce(user_query, '')) LIKE ANY(%s)
                   OR lower(coalesce(analysis, '')) LIKE ANY(%s)
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (patterns, patterns, limit),
            )
            rows = cur.fetchall()
        conn.commit()

        result: List[Dict[str, Any]] = []
        for r in rows:
            result.append(
                {
                    "id": r[0],
                    "created_at": r[1],
                    "user_query": r[2],
                    "planner_plan": r[3],
                    "sql": r[4],
                    "row_count": r[5],
                    "analysis": r[6],
                }
            )
        return result
    finally:
        conn.close()
