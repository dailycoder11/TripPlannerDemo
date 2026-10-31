"""Deterministic validation and normalization for external tool results."""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime
from typing import Any


class ToolResultRejected(ValueError):
    """Raised when provider output is not safe to add to model context."""


WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Freezing drizzle",
    61: "Slight rain", 63: "Rain", 65: "Heavy rain", 66: "Light freezing rain",
    67: "Heavy freezing rain", 71: "Slight snow", 73: "Snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Slight rain showers", 81: "Rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

_INSTRUCTION_PATTERNS = (
    re.compile(r"\b(ignore|disregard|forget|override|bypass)\b.{0,80}\b(previous|prior|above|system|developer|all)\b", re.I),
    re.compile(r"\b(reveal|show|print|output|expose|share)\b.{0,80}\b(prompt|secret|api[_ -]?key|credential|another user|other session)\b", re.I),
    re.compile(r"\b(access|read|retrieve|send|exfiltrate)\b.{0,80}\b(another user|other session|private memory|credentials|secrets)\b", re.I),
    re.compile(r"\b(call|invoke|use)\b.{0,60}\b(tool|function|booking|external action)\b", re.I),
)
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.I)
_MARKUP_PATTERN = re.compile(r"</?[a-z][^>]*>|\[[^\]]+\]\([^)]*\)", re.I)


def _reject(code: str) -> ToolResultRejected:
    return ToolResultRejected(code)


def _plain_text(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise _reject("invalid_tool_result_text")
    text = unicodedata.normalize("NFKC", value).strip()
    if not text or len(text) > maximum:
        raise _reject("invalid_tool_result_text")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in text):
        raise _reject("unsafe_tool_result_text")
    if "<" in text or ">" in text or _MARKUP_PATTERN.search(text) or _URL_PATTERN.search(text):
        raise _reject("unsafe_tool_result_text")
    normalized = " ".join(text.split())
    if any(pattern.search(normalized) for pattern in _INSTRUCTION_PATTERNS):
        raise _reject("unsafe_tool_result_text")
    return normalized


def _number(value: Any, *, minimum: float, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _reject("invalid_tool_result_number")
    number = float(value)
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        raise _reject("invalid_tool_result_number")
    return number


def _coordinate_pair(latitude: Any, longitude: Any) -> tuple[float, float]:
    lat = _number(latitude, minimum=-90, maximum=90)
    lon = _number(longitude, minimum=-180, maximum=180)
    return lat, lon


def _exact_keys(value: Any, required: set[str], *, optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _reject("invalid_tool_result_shape")
    allowed = required | (optional or set())
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise _reject("invalid_tool_result_shape")
    return value


def _validate_attractions(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, list) or len(result) > 10:
        raise _reject("invalid_attraction_results")
    cleaned: list[dict[str, Any]] = []
    seen_names: dict[str, tuple[float, float]] = {}
    for item in result:
        item = _exact_keys(
            item,
            {"name", "category", "latitude", "longitude", "source"},
            optional={"description"},
        )
        name = _plain_text(item["name"], field="name", maximum=120)
        category = _plain_text(item["category"], field="category", maximum=240)
        if item["source"] != "OpenTripMap":
            raise _reject("invalid_attraction_source")
        latitude, longitude = _coordinate_pair(item["latitude"], item["longitude"])
        identity = " ".join(name.casefold().split())
        prior = seen_names.get(identity)
        if prior and (abs(prior[0] - latitude) > 0.1 or abs(prior[1] - longitude) > 0.1):
            raise _reject("conflicting_attraction_coordinates")
        seen_names[identity] = (latitude, longitude)
        # Provider descriptions are deliberately omitted, even if present.
        cleaned.append({
            "name": name,
            "category": category,
            "latitude": latitude,
            "longitude": longitude,
            "source": "OpenTripMap",
        })
    return cleaned


def _optional_number(value: Any, *, minimum: float, maximum: float | None = None) -> float | None:
    return None if value is None else _number(value, minimum=minimum, maximum=maximum)


def _weather_code(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in WEATHER_CODES:
        raise _reject("invalid_weather_code")
    return value


def _validate_weather(result: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    required = {
        "date", "location", "timezone", "temperatureMinC", "temperatureMaxC",
        "precipitationProbabilityMaxPct", "precipitationMm", "weatherCode", "condition", "hourly",
    }
    result = _exact_keys(result, required)
    requested_date = arguments["trip_date"]
    if result["date"] != requested_date:
        raise _reject("weather_date_mismatch")

    location = result["location"]
    if not isinstance(location, dict) or not location or not set(location).issubset(
        {"name", "admin1", "country", "latitude", "longitude"}
    ):
        raise _reject("invalid_weather_location")
    clean_location: dict[str, Any] = {}
    for key in ("name", "admin1", "country"):
        if key in location and location[key] is not None:
            clean_location[key] = _plain_text(location[key], field=key, maximum=120)
    if "latitude" in location or "longitude" in location:
        if "latitude" not in location or "longitude" not in location:
            raise _reject("invalid_weather_location")
        clean_location["latitude"], clean_location["longitude"] = _coordinate_pair(
            location["latitude"], location["longitude"]
        )
    if not clean_location:
        raise _reject("invalid_weather_location")

    timezone = result["timezone"]
    if timezone is not None:
        timezone = _plain_text(timezone, field="timezone", maximum=64)
    weather_code = _weather_code(result["weatherCode"])
    if result["condition"] != WEATHER_CODES[weather_code]:
        raise _reject("weather_condition_mismatch")
    clean = {
        "date": requested_date,
        "location": clean_location,
        "timezone": timezone,
        "temperatureMinC": _optional_number(result["temperatureMinC"], minimum=-100, maximum=70),
        "temperatureMaxC": _optional_number(result["temperatureMaxC"], minimum=-100, maximum=70),
        "precipitationProbabilityMaxPct": _optional_number(
            result["precipitationProbabilityMaxPct"], minimum=0, maximum=100
        ),
        "precipitationMm": _optional_number(result["precipitationMm"], minimum=0, maximum=1000),
        "weatherCode": weather_code,
        "condition": WEATHER_CODES[weather_code],
    }
    low, high = clean["temperatureMinC"], clean["temperatureMaxC"]
    if low is not None and high is not None and low > high:
        raise _reject("weather_temperature_conflict")

    hourly = result["hourly"]
    if not isinstance(hourly, list) or len(hourly) > 24:
        raise _reject("invalid_hourly_weather")
    clean_hourly: list[dict[str, Any]] = []
    seen_times: set[str] = set()
    hourly_keys = {"time", "temperatureC", "precipitationProbabilityPct", "precipitationMm", "weatherCode", "condition"}
    for entry in hourly:
        entry = _exact_keys(entry, hourly_keys)
        timestamp = entry["time"]
        if not isinstance(timestamp, str):
            raise _reject("invalid_hourly_weather")
        try:
            parsed_time = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise _reject("invalid_hourly_weather") from exc
        if parsed_time.date().isoformat() != requested_date or timestamp in seen_times:
            raise _reject("invalid_hourly_weather")
        seen_times.add(timestamp)
        code = _weather_code(entry["weatherCode"])
        if entry["condition"] != WEATHER_CODES[code]:
            raise _reject("weather_condition_mismatch")
        clean_hourly.append({
            "time": timestamp,
            "temperatureC": _optional_number(entry["temperatureC"], minimum=-100, maximum=70),
            "precipitationProbabilityPct": _optional_number(
                entry["precipitationProbabilityPct"], minimum=0, maximum=100
            ),
            "precipitationMm": _optional_number(entry["precipitationMm"], minimum=0, maximum=1000),
            "weatherCode": code,
            "condition": WEATHER_CODES[code],
        })
    clean["hourly"] = clean_hourly
    return clean


def _endpoint_label(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return "coordinates"
    raise _reject("invalid_route_arguments")


def _validate_endpoint_argument(value: Any) -> Any:
    if isinstance(value, str):
        text = _plain_text(value, field="place", maximum=160)
        parts = text.split(",")
        if len(parts) == 2:
            try:
                latitude, longitude = float(parts[0]), float(parts[1])
            except ValueError:
                return text
            _coordinate_pair(latitude, longitude)
        return text
    if not isinstance(value, dict) or not set(value).issubset(
        {"name", "latitude", "longitude", "lat", "lon"}
    ):
        raise _reject("invalid_route_arguments")
    if "latitude" in value and "lat" in value or "longitude" in value and "lon" in value:
        raise _reject("invalid_route_arguments")
    latitude = value.get("latitude", value.get("lat"))
    longitude = value.get("longitude", value.get("lon"))
    if (latitude is None) != (longitude is None):
        raise _reject("invalid_route_arguments")
    clean: dict[str, Any] = {}
    if latitude is not None:
        clean["latitude"], clean["longitude"] = _coordinate_pair(latitude, longitude)
    if "name" in value:
        clean["name"] = _plain_text(value["name"], field="name", maximum=160)
    if not clean:
        raise _reject("invalid_route_arguments")
    return clean


def validate_tool_arguments(name: str, arguments: Any) -> dict[str, Any]:
    """Validate exact tool argument shapes, text bounds, and coordinates."""
    if not isinstance(arguments, dict):
        raise _reject("invalid_tool_arguments")
    if name == "search_attractions":
        if not {"place"}.issubset(arguments) or not set(arguments).issubset({"place", "category", "limit"}):
            raise _reject("invalid_tool_arguments")
        clean: dict[str, Any] = {"place": _plain_text(arguments["place"], field="place", maximum=100)}
        if "category" in arguments and arguments["category"] is not None:
            clean["category"] = _plain_text(arguments["category"], field="category", maximum=80)
        if "limit" in arguments:
            limit = arguments["limit"]
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
                raise _reject("invalid_tool_arguments")
            clean["limit"] = limit
        return clean

    if name == "get_weather":
        if "trip_date" not in arguments or not set(arguments).issubset(
            {"trip_date", "city", "latitude", "longitude"}
        ):
            raise _reject("invalid_tool_arguments")
        trip_date = arguments["trip_date"]
        if not isinstance(trip_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", trip_date):
            raise _reject("invalid_tool_arguments")
        try:
            date.fromisoformat(trip_date)
        except ValueError as exc:
            raise _reject("invalid_tool_arguments") from exc
        city = arguments.get("city")
        latitude = arguments.get("latitude")
        longitude = arguments.get("longitude")
        if (latitude is None) != (longitude is None):
            raise _reject("invalid_tool_arguments")
        if latitude is not None:
            latitude, longitude = _coordinate_pair(latitude, longitude)
        if city is not None:
            city = _plain_text(city, field="city", maximum=100)
        if city is None and latitude is None:
            raise _reject("invalid_tool_arguments")
        clean = {"trip_date": trip_date}
        if city is not None:
            clean["city"] = city
        if latitude is not None:
            clean["latitude"] = latitude
            clean["longitude"] = longitude
        return clean

    if name == "get_travel_time":
        if set(arguments) != {"origin", "destination"}:
            raise _reject("invalid_tool_arguments")
        return {
            "origin": _validate_endpoint_argument(arguments["origin"]),
            "destination": _validate_endpoint_argument(arguments["destination"]),
        }
    raise _reject("unknown_tool")


def _validate_route(result: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    result = _exact_keys(result, {"origin", "destination", "distanceKm", "durationMinutes", "mode", "source"})
    origin = _plain_text(result["origin"], field="origin", maximum=160)
    destination = _plain_text(result["destination"], field="destination", maximum=160)
    if origin != _endpoint_label(arguments["origin"]) or destination != _endpoint_label(arguments["destination"]):
        raise _reject("route_endpoint_mismatch")
    if result["mode"] != "driving" or result["source"] != "OSRM":
        raise _reject("invalid_route_source")
    return {
        "origin": origin,
        "destination": destination,
        "distanceKm": _number(result["distanceKm"], minimum=0),
        "durationMinutes": _number(result["durationMinutes"], minimum=0),
        "mode": "driving",
        "source": "OSRM",
    }


def validate_tool_result(name: str, result: Any, arguments: dict[str, Any]) -> Any:
    """Validate and return a minimal normalized result safe for model context."""
    if name == "search_attractions":
        return _validate_attractions(result)
    if name == "get_weather":
        return _validate_weather(result, arguments)
    if name == "get_travel_time":
        return _validate_route(result, arguments)
    raise _reject("unknown_tool_result")
