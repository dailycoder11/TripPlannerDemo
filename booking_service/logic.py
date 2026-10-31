"""Booking policy and LLM proposal logic, owned by the booking service."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from datetime import date
from typing import Any

from groq import Groq

from booking_service.storage import BookingStore


class BookingError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise BookingError("invalid_request", "The booking request must be a JSON object.")
    request_id = request.get("request_id")
    event_name = request.get("event_name")
    event_date = request.get("event_date")
    attendees = request.get("attendee_count")
    amount = request.get("amount_per_person_inr")
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id):
        raise BookingError("invalid_request_id", "request_id must contain 1–100 letters, numbers, underscores, or hyphens.")
    if not isinstance(event_name, str) or not event_name.strip() or len(event_name.strip()) > 160:
        raise BookingError("invalid_event_name", "event_name must contain 1–160 characters.")
    if not isinstance(event_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", event_date):
        raise BookingError("invalid_event_date", "event_date must be a valid YYYY-MM-DD date.")
    try:
        date.fromisoformat(event_date)
    except ValueError as exc:
        raise BookingError("invalid_event_date", "event_date must be a valid YYYY-MM-DD date.") from exc
    if (isinstance(attendees, bool) or not isinstance(attendees, (int, float))
            or not float(attendees).is_integer() or attendees < 1 or attendees > 500):
        raise BookingError("invalid_attendee_count", "attendee_count must be an integer from 1 to 500.")
    attendees = int(attendees)
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
        raise BookingError("invalid_amount", "amount_per_person_inr must be a positive number.")
    return {
        "request_id": request_id,
        "event_name": event_name.strip(),
        "event_date": event_date,
        "attendee_count": attendees,
        "amount_per_person_inr": round(float(amount), 2),
    }


def _proposal(request: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key":
        raise BookingError("llm_unavailable", "The booking service is missing GROQ_API_KEY and cannot suggest a revision.")
    minimum_attendees = int(config["minimumAttendees"])
    minimum_amount = float(config["minimumAmountPerPersonInr"])
    client = Groq(api_key=api_key, timeout=30.0, max_retries=0)
    prompt = {
        "request": request,
        "failed_rules": {
            "minimum_attendees": minimum_attendees,
            "minimum_amount_per_person_inr": minimum_amount,
        },
        "instruction": "Suggest the smallest changes needed so the request satisfies all failed rules. Keep compliant fields unchanged. Return JSON with attendee_count, amount_per_person_inr, explanation.",
    }
    try:
        response = client.chat.completions.create(
            model=config.get("model", "openai/gpt-oss-20b"),
            temperature=0.1,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the ticket-booking agent used by Daytrip after its day-trip planning flow. "
                        "You only suggest revisions for a demo request; you do not make real bookings or payments. "
                        "Never change or reinterpret the supplied booking rules. Output only the requested JSON object."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        suggestion = json.loads(response.choices[0].message.content or "")
    except Exception as exc:
        raise BookingError("proposal_generation_failed", "The booking service could not generate a valid suggestion. Please try again later.") from exc
    attendees = suggestion.get("attendee_count") if isinstance(suggestion, dict) else None
    amount = suggestion.get("amount_per_person_inr") if isinstance(suggestion, dict) else None
    explanation = suggestion.get("explanation") if isinstance(suggestion, dict) else None
    if (isinstance(attendees, bool) or not isinstance(attendees, (int, float)) or not float(attendees).is_integer()
            or attendees < minimum_attendees or attendees > 500
            or isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < minimum_amount
            or not isinstance(explanation, str) or not explanation.strip()):
        raise BookingError("invalid_llm_proposal", "The booking service received an invalid suggestion and did not create a booking.")
    suggested = {
        "attendee_count": attendees,
        "amount_per_person_inr": round(float(amount), 2),
    }
    # Do not allow an LLM to inflate fields that already satisfy the rules.
    if request["attendee_count"] >= minimum_attendees and attendees != request["attendee_count"]:
        raise BookingError("invalid_llm_proposal", "The suggestion changed a value that already met the booking rules.")
    if request["amount_per_person_inr"] >= minimum_amount and suggested["amount_per_person_inr"] != request["amount_per_person_inr"]:
        raise BookingError("invalid_llm_proposal", "The suggestion changed a value that already met the booking rules.")
    return {
        "status": "proposal",
        "original_request": request,
        "suggested_values": suggested,
        "explanation": explanation.strip()[:500],
    }


def process_booking(request_data: Any, config: dict[str, Any], store: BookingStore) -> dict[str, Any]:
    try:
        request = validate_request(request_data)
    except BookingError as exc:
        return {"status": "failed", "error_code": exc.code, "message": str(exc)}
    with store.transaction():
        existing = store.get(request["request_id"])
        if existing:
            if existing.get("request_fingerprint") != {k: v for k, v in request.items() if k != "request_id"}:
                return {"status": "failed", "error_code": "request_id_conflict", "message": "This request_id was already used for different booking details."}
            return existing["result"]
        min_attendees = int(config["minimumAttendees"])
        min_amount = float(config["minimumAmountPerPersonInr"])
        if request["attendee_count"] < min_attendees or request["amount_per_person_inr"] < min_amount:
            try:
                result = _proposal(request, config)
            except BookingError as exc:
                result = {"status": "failed", "error_code": exc.code, "message": str(exc)}
            except Exception:
                result = {
                    "status": "failed",
                    "error_code": "proposal_generation_failed",
                    "message": "The booking service could not generate a valid suggestion. Please try again later.",
                }
        else:
            result = {
                "status": "confirmed",
                "confirmation_id": "DUMMY-" + uuid.uuid4().hex[:12].upper(),
                "event_name": request["event_name"],
                "event_date": request["event_date"],
                "attendee_count": request["attendee_count"],
                "amount_per_person_inr": request["amount_per_person_inr"],
                "total_amount_inr": round(request["attendee_count"] * request["amount_per_person_inr"], 2),
                "booking_status": "confirmed",
            }
        store.put(request["request_id"], {
            "request_fingerprint": {k: v for k, v in request.items() if k != "request_id"},
            "result": result,
        })
        return result
