from typing import Any, Dict, List
import psycopg2
from psycopg2.extras import RealDictCursor

from .config import settings


def get_connection():
    """Create a new PostgreSQL connection.

    Uses environment / config parameters, compatible with docker-compose.
    """
    return psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


def run_sql(query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Execute a read-only SQL query and return result rows as list of dicts.

    A simple safety check ensures that only SELECT statements are allowed.
    """
    sql_clean = query.strip().lower()

    # Allow optional trailing semicolon but forbid multiple statements
    if ";" in sql_clean[:-1]:
        raise ValueError("Only single-statement SELECT queries are allowed.")

    if not sql_clean.startswith("select"):
        raise ValueError("Only SELECT queries are allowed in Collector.")

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # если параметры реально есть — передаём их,
            # если None или пустой dict/список — выполняем без параметров
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            rows = cur.fetchall()
        conn.commit()
        return [dict(r) for r in rows]
    finally:
        conn.close()
