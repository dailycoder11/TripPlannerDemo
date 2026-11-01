"""Calendar-month windows used to refresh and compare tourism news."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class NewsPeriod:
    kind: str
    start_date: date
    end_date: date

    @property
    def period_id(self) -> str:
        return f"{self.kind}:{self.start_date.isoformat()}:{self.end_date.isoformat()}"


def _month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def _shift_month(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    return _month_start(index // 12, index % 12 + 1)


def _month_end(value: date) -> date:
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def build_news_periods(today: date | None = None) -> list[NewsPeriod]:
    """Return four complete 3-month history windows and the latest YoY window.

    The history covers the last twelve completed calendar months, grouped into
    four consecutive three-month windows. The comparison is the same latest
    three-month window one year earlier.
    """
    today = today or date.today()
    last_completed_month = _shift_month(today.replace(day=1), -1)
    latest_end = _month_end(last_completed_month)
    latest_start = _shift_month(last_completed_month, -2)
    history_start = _shift_month(latest_start, -9)

    periods: list[NewsPeriod] = []
    window_start = history_start
    for _ in range(4):
        window_end_month = _shift_month(window_start, 2)
        periods.append(NewsPeriod("history", window_start, _month_end(window_end_month)))
        window_start = _shift_month(window_start, 3)

    periods.append(NewsPeriod(
        "comparison",
        _shift_month(latest_start, -12),
        _month_end(_shift_month(last_completed_month, -12)),
    ))
    assert periods[-2].start_date == latest_start
    assert periods[-2].end_date == latest_end
    return periods
