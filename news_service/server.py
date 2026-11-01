"""Local REST service for city tourism news search and comparison."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request

from news_service.ingestion import refresh_news_store
from news_service.storage import NewsStore

SERVICE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SERVICE_DIR.parent
load_dotenv(ROOT_DIR / ".env")
CONFIG = json.loads((SERVICE_DIR / "config.json").read_text(encoding="utf-8"))
DATABASE_PATH = SERVICE_DIR / CONFIG.get("databaseFile", "data/news.sqlite3")
LOGGER = logging.getLogger("news_service")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _canonical_city(value: str, configured_cities: list[str]) -> str | None:
    requested = value.strip().casefold()
    return next((city for city in configured_cities if city.casefold() == requested), None)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    store: NewsStore = app.state.news_store
    await asyncio.to_thread(store.initialize)
    app.state.refresh_status = "failed"
    app.state.refresh_error = None
    app.state.refresh_summary = None
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key or api_key == "your_tavily_api_key":
        app.state.refresh_error = "TAVILY_API_KEY is not configured."
    else:
        try:
            app.state.refresh_summary = await asyncio.to_thread(
                refresh_news_store,
                store,
                cities=app.state.cities,
                api_key=api_key,
                max_results=int(CONFIG.get("maxResultsPerSearch", 10)),
                search_depth=str(CONFIG.get("searchDepth", "basic")),
            )
            app.state.refresh_status = "succeeded"
        except Exception as exc:
            app.state.refresh_error = type(exc).__name__
            LOGGER.exception("News snapshot refresh failed (%s)", type(exc).__name__)
    LOGGER.info(json.dumps({
        "event": "news_service_ready" if store.has_snapshot() else "news_service_unavailable",
        "refresh_status": app.state.refresh_status,
        "refresh_error": app.state.refresh_error,
        "summary": app.state.refresh_summary,
        "serving_previous_snapshot": store.has_snapshot() and app.state.refresh_status != "succeeded",
    }, ensure_ascii=False))
    yield


def create_app(*, database_path: Path | str | None = None, cities: list[str] | None = None) -> FastAPI:
    configured_cities = cities if cities is not None else CONFIG.get("cities", [])
    store = NewsStore(database_path or DATABASE_PATH)
    app = FastAPI(
        title="Daytrip City Tourism News",
        description="Local news excerpts and source links for configured Indian tourism cities.",
        version="1.0.0",
        lifespan=_lifespan,
    )
    app.state.news_store = store
    app.state.cities = configured_cities

    @app.get("/health")
    async def health(request: Request):
        ready = request.app.state.news_store.has_snapshot()
        if not ready:
            raise HTTPException(status_code=503, detail={
                "status": "unavailable",
                "refresh_status": request.app.state.refresh_status,
                "refresh_error": request.app.state.refresh_error,
            })
        return {
            "status": "ready",
            "configured_cities": request.app.state.cities,
            "refresh_status": request.app.state.refresh_status,
            "refresh_error": request.app.state.refresh_error,
            "refresh": request.app.state.refresh_summary,
            "serving_previous_snapshot": ready and request.app.state.refresh_status != "succeeded",
        }

    @app.get("/api/v1/news")
    async def tourism_news(request: Request, city: str = Query(min_length=1, max_length=100)):
        canonical_city = _canonical_city(city, request.app.state.cities)
        if canonical_city is None:
            return {"city": city, "available": False, "current": None, "last_year_same_time": None}
        if not request.app.state.news_store.has_snapshot():
            raise HTTPException(status_code=503, detail="Tourism news is not ready yet.")
        result = request.app.state.news_store.latest_comparison(canonical_city)
        if result is None:
            raise HTTPException(status_code=503, detail="Tourism news is not available for this city yet.")
        return {
            "available": True,
            "serving_previous_snapshot": request.app.state.refresh_status != "succeeded",
            **result,
        }

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=str(CONFIG.get("host", "127.0.0.1")),
        port=int(CONFIG.get("port", 8003)),
    )
