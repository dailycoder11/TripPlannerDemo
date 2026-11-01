"""Tavily search adapter and strict normalization for stored news excerpts."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _safe_article(item: Any) -> dict[str, str | None] | None:
    if not isinstance(item, dict):
        return None
    title = item.get("title")
    excerpt = item.get("content")
    source_url = item.get("url")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(excerpt, str) or not excerpt.strip():
        return None
    if not isinstance(source_url, str):
        return None
    parsed = urlparse(source_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    published_date = item.get("published_date")
    if published_date is not None and not isinstance(published_date, str):
        published_date = None
    return {
        "title": title.strip()[:300],
        "excerpt": excerpt.strip()[:2000],
        "source_url": source_url.strip()[:2000],
        "source_name": parsed.hostname.lower()[:255],
        "published_date": published_date.strip()[:80] if published_date else None,
    }


def search_tourism_news(
    *,
    api_key: str,
    city: str,
    start_date: str,
    end_date: str,
    max_results: int = 10,
    search_depth: str = "basic",
    timeout: float = 25.0,
    session: Any = requests,
) -> list[dict[str, str | None]]:
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not configured")
    response = session.post(
        TAVILY_SEARCH_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "query": f"{city} tourism",
            "topic": "news",
            "search_depth": search_depth,
            "max_results": max(1, min(int(max_results), 10)),
            "start_date": start_date,
            "end_date": end_date,
            "filter_by_published_date": True,
            "include_published_date": True,
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError("Tavily returned an unexpected search response")

    articles: list[dict[str, str | None]] = []
    seen_urls: set[str] = set()
    for item in payload["results"]:
        article = _safe_article(item)
        if not article or article["source_url"] in seen_urls:
            continue
        articles.append(article)
        seen_urls.add(str(article["source_url"]))
        if len(articles) >= max_results:
            break
    return articles
