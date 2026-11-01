"""Build a complete rolling tourism-news snapshot from Tavily."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable

from news_service.periods import NewsPeriod, build_news_periods
from news_service.storage import NewsStore
from news_service.tavily import search_tourism_news


def collect_news_snapshot(
    *,
    cities: list[str],
    api_key: str,
    max_results: int = 10,
    search_depth: str = "basic",
    today: date | None = None,
    search: Callable[..., list[dict[str, Any]]] = search_tourism_news,
) -> list[dict[str, Any]]:
    periods = build_news_periods(today)
    batches: list[dict[str, Any]] = []
    for city in cities:
        for period in periods:
            articles = search(
                api_key=api_key,
                city=city,
                start_date=period.start_date.isoformat(),
                end_date=period.end_date.isoformat(),
                max_results=max_results,
                search_depth=search_depth,
            )
            batches.append({
                "city": city,
                "period_kind": period.kind,
                "start_date": period.start_date.isoformat(),
                "end_date": period.end_date.isoformat(),
                "articles": articles,
            })
    return batches


def refresh_news_store(
    store: NewsStore,
    *,
    cities: list[str],
    api_key: str,
    max_results: int = 10,
    search_depth: str = "basic",
    today: date | None = None,
    search: Callable[..., list[dict[str, Any]]] = search_tourism_news,
) -> dict[str, Any]:
    """Fetch everything before replacing the prior DB snapshot atomically."""
    batches = collect_news_snapshot(
        cities=cities,
        api_key=api_key,
        max_results=max_results,
        search_depth=search_depth,
        today=today,
        search=search,
    )
    fetched_at = datetime.now().astimezone().isoformat()
    store.replace_snapshot(batches, fetched_at)
    return {
        "cities": len(cities),
        "windows": len(batches),
        "articles": sum(len(batch["articles"]) for batch in batches),
        "fetched_at": fetched_at,
    }
