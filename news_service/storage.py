"""SQLite persistence for collected city news."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class NewsStore:
    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)

    def connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS news_windows (
                    city TEXT NOT NULL,
                    period_kind TEXT NOT NULL CHECK (period_kind IN ('history', 'comparison')),
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    result_count INTEGER NOT NULL,
                    PRIMARY KEY (city, period_kind, start_date, end_date)
                );
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY,
                    city TEXT NOT NULL,
                    period_kind TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    excerpt TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    published_date TEXT,
                    FOREIGN KEY (city, period_kind, start_date, end_date)
                        REFERENCES news_windows(city, period_kind, start_date, end_date)
                        ON DELETE CASCADE,
                    UNIQUE (city, period_kind, start_date, end_date, source_url)
                );
                CREATE INDEX IF NOT EXISTS news_articles_city_window
                    ON news_articles(city, period_kind, start_date, end_date, rank);
                """
            )
            connection.commit()

    def has_snapshot(self) -> bool:
        try:
            with self._connection() as connection:
                row = connection.execute("SELECT COUNT(*) AS total FROM news_windows").fetchone()
                return bool(row and row["total"])
        except sqlite3.Error:
            return False

    def replace_snapshot(self, batches: list[dict[str, Any]], fetched_at: str) -> None:
        """Atomically replace the full snapshot after all remote calls succeed."""
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM news_articles")
                connection.execute("DELETE FROM news_windows")
                for batch in batches:
                    connection.execute(
                        """INSERT INTO news_windows
                           (city, period_kind, start_date, end_date, fetched_at, result_count)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (batch["city"], batch["period_kind"], batch["start_date"],
                         batch["end_date"], fetched_at, len(batch["articles"])),
                    )
                    for rank, article in enumerate(batch["articles"], start=1):
                        connection.execute(
                            """INSERT OR IGNORE INTO news_articles
                               (city, period_kind, start_date, end_date, rank, title,
                                excerpt, source_url, source_name, published_date)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (batch["city"], batch["period_kind"], batch["start_date"],
                             batch["end_date"], rank, article["title"], article["excerpt"],
                             article["source_url"], article["source_name"], article.get("published_date")),
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def latest_comparison(self, city: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            current = connection.execute(
                """SELECT * FROM news_windows WHERE city = ? AND period_kind = 'history'
                   ORDER BY end_date DESC LIMIT 1""",
                (city,),
            ).fetchone()
            comparison = connection.execute(
                """SELECT * FROM news_windows WHERE city = ? AND period_kind = 'comparison'
                   ORDER BY end_date DESC LIMIT 1""",
                (city,),
            ).fetchone()
            if current is None or comparison is None:
                return None

            return {
                "city": city,
                "fetched_at": current["fetched_at"],
                "current": self._window_with_articles(connection, current),
                "last_year_same_time": self._window_with_articles(connection, comparison),
            }

    @staticmethod
    def _window_with_articles(connection: sqlite3.Connection, window: sqlite3.Row) -> dict[str, Any]:
        articles = connection.execute(
            """SELECT title, excerpt, source_url, source_name, published_date
               FROM news_articles
               WHERE city = ? AND period_kind = ? AND start_date = ? AND end_date = ?
               ORDER BY rank, id""",
            (window["city"], window["period_kind"], window["start_date"], window["end_date"]),
        ).fetchall()
        return {
            "start_date": window["start_date"],
            "end_date": window["end_date"],
            "articles": [dict(article) for article in articles],
        }
