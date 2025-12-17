from __future__ import annotations

from typing import List, Dict, Any, Optional
import requests


BASE_URL = "https://api.openalex.org/works"
TOPICS_BASE_URL = "https://api.openalex.org/topics"


def list_works(
    *,
    filters: Optional[str] = None,
    search: Optional[str] = None,
    sort: Optional[str] = None,
    select: Optional[str] = None,
    per_page: int = 25,
    page: int = 1,
    email: str | None = None,
) -> Dict[str, Any]:
    """Get a list of works from OpenAlex.

    This is the low-level helper used for "raw_table_view" in Collector.

    Supports standard list params: filter/search/sort/per-page/page/select.
    - select only supports root-level fields (per OpenAlex docs)
    - select does NOT work with group_by (that's why it's separate from group_works)
    """

    params: Dict[str, Any] = {
        "per-page": int(per_page),
        "page": int(page),
    }
    if filters:
        params["filter"] = filters
    if search:
        params["search"] = search
    if sort:
        params["sort"] = sort
    if select:
        params["select"] = select
    if email:
        params["mailto"] = email

    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def group_works(
    group_by: str,
    *,
    filters: Optional[str] = None,
    search: Optional[str] = None,
    per_page: int = 200,
    email: str | None = None,
) -> Dict[str, Any]:
    """Group (facet) OpenAlex works.

    This calls the /works endpoint with the `group_by` parameter and optional
    `filter`/`search` arguments.

    Response format (docs): returns `meta` and `group_by` fields.
    We return the full decoded JSON so the caller can extract both.
    """

    params: Dict[str, Any] = {
        "group_by": group_by,
        "per-page": int(per_page),
    }
    if filters:
        params["filter"] = filters
    if search:
        params["search"] = search
    if email:
        params["mailto"] = email

    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def search_works(
    query: str,
    per_page: int = 10,
    email: str | None = None,
) -> List[Dict[str, Any]]:
    """Search OpenAlex works by fulltext query in title/abstract/fulltext.

    Uses the /works endpoint with the `search` parameter.
    Docs: https://docs.openalex.org/api-entities/works/search-works
    """
    params = {
        "search": query,
        "per-page": per_page,
    }
    if email:
        params["mailto"] = email

    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def search_topics(
    query: str,
    per_page: int = 10,
    email: str | None = None,
) -> List[Dict[str, Any]]:
    """Search OpenAlex topics by name.

    Uses the /topics endpoint with the search parameter.
    Docs: https://docs.openalex.org/api-entities/topics/search-topics
    """
    params = {
        "search": query,
        "per-page": per_page,
    }
    if email:
        params["mailto"] = email

    resp = requests.get(TOPICS_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])
