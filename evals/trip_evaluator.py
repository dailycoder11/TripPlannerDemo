"""Deterministic checks for structured trip itineraries."""

from __future__ import annotations

import json
import math
import re
from datetime import date, time
from typing import Any


MAX_REVISIONS = 1


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _as_time(value: time | str) -> time:
    return value if isinstance(value, time) else time.fromisoformat(value)


def _parse_itinerary(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        parsed = value
    else:
        content = value.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            content = fenced.group(1)
        parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("The itinerary JSON must be an object.")
    return parsed


def evaluate_itinerary(
    itinerary: str | dict[str, Any],
    *,
    budget_per_person: float,
    requested_start_time: time | str,
    requested_end_time: time | str,
    trip_start_date: date | str,
    trip_end_date: date | str,
) -> dict[str, Any]:
    """Evaluate itinerary data and return a PASS/FAIL result with check details."""
    checks: dict[str, str] = {}
    failures: list[str] = []

    def record(key: str, passed: bool, failure: str) -> None:
        checks[key] = "PASS" if passed else "FAIL"
        if not passed:
            failures.append(failure)

    try:
        plan = _parse_itinerary(itinerary)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        checks["format"] = "FAIL"
        failures.append(f"Itinerary is not valid structured JSON: {exc}")
        return {"status": "FAIL", "checks": checks, "failures": failures}

    expected_plan_keys = {"cost_per_person", "activities"}
    schema_valid = set(plan) == expected_plan_keys
    raw_activities = plan.get("activities")
    expected_activity_keys = {
        "date", "start_time", "end_time", "title", "category", "description"
    }
    if not isinstance(raw_activities, list):
        schema_valid = False
    else:
        for activity in raw_activities:
            if not isinstance(activity, dict) or set(activity) != expected_activity_keys:
                schema_valid = False
                break
            if any(not isinstance(activity.get(key), str) for key in expected_activity_keys):
                schema_valid = False
                break
            if activity.get("category", "").strip().casefold() not in {"activity", "meal"}:
                schema_valid = False
                break
    record("format", schema_valid, "Itinerary does not match the required JSON structure")
    raw_cost = plan.get("cost_per_person")
    cost_valid = (
        isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool)
        and math.isfinite(raw_cost) and raw_cost > 0
    )
    record("budget", bool(cost_valid and raw_cost <= budget_per_person), "Budget exceeded or per-person cost is missing/invalid")

    activities = plan.get("activities")
    if not isinstance(activities, list):
        activities = []
    valid_activity_objects = [item for item in activities if isinstance(item, dict)]
    record("activity_count", len(valid_activity_objects) >= 2, "At least 2 activities are required")

    try:
        start_bound = _as_date(trip_start_date)
        end_bound = _as_date(trip_end_date)
        activity_dates = [_as_date(item["date"]) for item in valid_activity_objects]
        dates_valid = len(activity_dates) == len(activities) and all(
            start_bound <= activity_date <= end_bound for activity_date in activity_dates
        )
    except (KeyError, TypeError, ValueError):
        activity_dates = []
        dates_valid = False
    record("activity_dates", dates_valid, "All activities must fall within the trip start and end dates")

    scheduled: list[tuple[date, time, time]] = []
    try:
        for item in valid_activity_objects:
            scheduled.append((
                _as_date(item["date"]),
                _as_time(item["start_time"]),
                _as_time(item["end_time"]),
            ))
        scheduled.sort(key=lambda row: (row[0], row[1]))
        first_start = scheduled[0][1]
        last_end = max(scheduled, key=lambda row: (row[0], row[2]))[2]
        requested_start = _as_time(requested_start_time)
        requested_end = _as_time(requested_end_time)
        start_valid = first_start >= requested_start
        end_valid = last_end <= requested_end
    except (IndexError, KeyError, TypeError, ValueError):
        start_valid = end_valid = False
    record("start_time", start_valid, "Trip starts before the requested start time or has invalid activity times")
    record("end_time", end_valid, "Trip ends after the requested end time or has invalid activity times")

    meal_terms = re.compile(r"\b(lunch|meal|breakfast|dinner)\b", re.IGNORECASE)
    has_meal = any(
        str(item.get("category", "")).strip().casefold() in {"meal", "food", "lunch", "dining"}
        or bool(meal_terms.search(str(item.get("title", ""))))
        for item in valid_activity_objects
    )
    record("meal", has_meal, "Include a lunch or meal activity")

    return {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
    }
