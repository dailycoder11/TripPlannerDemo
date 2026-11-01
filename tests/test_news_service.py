"""Tests for tourism-news windows, Tavily requests, and SQLite snapshot safety."""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from news_service.ingestion import collect_news_snapshot, refresh_news_store
from news_service.periods import build_news_periods
from news_service.server import create_app
from news_service.storage import NewsStore
from news_service.tavily import search_tourism_news


class NewsPeriodTests(unittest.TestCase):
    def test_november_first_refresh_has_four_rolling_windows_and_yoy_comparison(self):
        periods = build_news_periods(date(2026, 11, 1))
        self.assertEqual(
            [(p.kind, p.start_date.isoformat(), p.end_date.isoformat()) for p in periods],
            [
                ("history", "2025-11-01", "2026-01-31"),
                ("history", "2026-02-01", "2026-04-30"),
                ("history", "2026-05-01", "2026-07-31"),
                ("history", "2026-08-01", "2026-10-31"),
                ("comparison", "2025-08-01", "2025-10-31"),
            ],
        )

    def test_five_cities_use_five_searches_each_with_max_ten_results(self):
        calls = []

        def search(**kwargs):
            calls.append(kwargs)
            return []

        batches = collect_news_snapshot(
            cities=["Shimla", "Manali", "Jaipur", "Agra", "Udaipur"],
            api_key="test-token",
            today=date(2026, 11, 1),
            search=search,
        )
        self.assertEqual(len(calls), 25)
        self.assertEqual(len(batches), 25)
        self.assertTrue(all(call["max_results"] == 10 for call in calls))
        self.assertTrue(all(call["search_depth"] == "basic" for call in calls))


class TavilyAdapterTests(unittest.TestCase):
    def test_search_sends_news_date_filters_and_returns_safe_unique_articles(self):
        class Response:
            def raise_for_status(self):
                pass

            @staticmethod
            def json():
                return {"results": [
                    {"title": "Shimla festival", "content": "A news excerpt.",
                     "url": "https://example.org/story", "published_date": "2026-10-03"},
                    {"title": "duplicate", "content": "duplicate excerpt",
                     "url": "https://example.org/story", "published_date": "2026-10-03"},
                    {"title": "unsafe link", "content": "skip me", "url": "javascript:alert(1)"},
                ]}

        class Session:
            def __init__(self):
                self.call = None

            def post(self, *args, **kwargs):
                self.call = (args, kwargs)
                return Response()

        session = Session()
        results = search_tourism_news(
            api_key="test-key", city="Shimla", start_date="2026-08-01", end_date="2026-10-31",
            max_results=10, session=session,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_name"], "example.org")
        payload = session.call[1]["json"]
        self.assertEqual(payload["query"], "Shimla tourism")
        self.assertEqual(payload["topic"], "news")
        self.assertEqual(payload["start_date"], "2026-08-01")
        self.assertEqual(payload["end_date"], "2026-10-31")
        self.assertTrue(payload["filter_by_published_date"])
        self.assertFalse(payload["include_raw_content"])


class NewsStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = NewsStore(Path(self.temp_dir.name) / "news.sqlite3")
        self.store.initialize()

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def article(title):
        slug = title.replace(" ", "-")
        return {"title": title, "excerpt": f"{title} excerpt", "source_url": f"https://news.example/{slug}",
                "source_name": "news.example", "published_date": "2026-10-01"}

    def seed(self, title="previous snapshot"):
        self.store.replace_snapshot([
            {"city": "Shimla", "period_kind": "history", "start_date": "2026-08-01",
             "end_date": "2026-10-31", "articles": [self.article(title)]},
            {"city": "Shimla", "period_kind": "comparison", "start_date": "2025-08-01",
             "end_date": "2025-10-31", "articles": [self.article("last year")]},
        ], "2026-11-01T00:00:00+00:00")

    def test_latest_comparison_returns_periods_and_source_links(self):
        self.seed()
        result = self.store.latest_comparison("Shimla")
        self.assertEqual(result["current"]["start_date"], "2026-08-01")
        self.assertEqual(result["last_year_same_time"]["start_date"], "2025-08-01")
        self.assertEqual(result["current"]["articles"][0]["source_url"], "https://news.example/previous-snapshot")

    def test_failed_remote_refresh_does_not_replace_last_good_data(self):
        self.seed()
        count = 0

        def failing_search(**kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("upstream failure")
            return [self.article("new data")]

        with self.assertRaises(RuntimeError):
            refresh_news_store(
                self.store, cities=["Shimla"], api_key="test-key", today=date(2026, 11, 1),
                search=failing_search,
            )
        result = self.store.latest_comparison("Shimla")
        self.assertEqual(result["current"]["articles"][0]["title"], "previous snapshot")

    def test_service_api_returns_comparison_and_unknown_cities_as_unavailable(self):
        self.seed()
        app = create_app(database_path=self.store.database_path, cities=["Shimla"])
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}), TestClient(app) as client:
            response = client.get("/api/v1/news", params={"city": "shimla"})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["available"])
            self.assertEqual(response.json()["current"]["articles"][0]["title"], "previous snapshot")
            unknown = client.get("/api/v1/news", params={"city": "Mumbai"})
            self.assertEqual(unknown.status_code, 200)
            self.assertFalse(unknown.json()["available"])

    def test_health_is_unavailable_when_no_complete_snapshot_exists(self):
        empty_store_path = Path(self.temp_dir.name) / "empty.sqlite3"
        app = create_app(database_path=empty_store_path, cities=["Shimla"])
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}), TestClient(app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
