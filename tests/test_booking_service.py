"""Focused tests for the booking policy, proposal validation, and idempotency."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from booking_service.logic import process_booking
from booking_service.storage import BookingStore


CONFIG = {"minimumAttendees": 2, "minimumAmountPerPersonInr": 500, "model": "unit-test"}


class BookingServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = BookingStore(Path(self.temp_dir.name) / "bookings.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def request(request_id="request-1", attendees=2, amount=750):
        return {
            "request_id": request_id,
            "event_name": "Demo concert",
            "event_date": "2026-11-15",
            "attendee_count": attendees,
            "amount_per_person_inr": amount,
        }

    def test_valid_request_confirms_and_retries_idempotently(self):
        request = self.request()
        first = process_booking(request, CONFIG, self.store)
        second = process_booking(request, CONFIG, self.store)
        self.assertEqual(first["status"], "confirmed")
        self.assertEqual(first["total_amount_inr"], 1500)
        self.assertEqual(first["confirmation_id"], second["confirmation_id"])

    def test_same_request_id_with_different_values_is_rejected(self):
        process_booking(self.request(), CONFIG, self.store)
        response = process_booking(self.request(attendees=3), CONFIG, self.store)
        self.assertEqual(response["error_code"], "request_id_conflict")

    def test_policy_violation_returns_validated_proposal(self):
        proposal = {
            "status": "proposal",
            "original_request": self.request(attendees=1, amount=250),
            "suggested_values": {"attendee_count": 2, "amount_per_person_inr": 500},
            "explanation": "Adjusted to meet minimum requirements.",
        }
        with patch("booking_service.logic._proposal", return_value=proposal):
            response = process_booking(self.request(attendees=1, amount=250), CONFIG, self.store)
        self.assertEqual(response["status"], "proposal")
        self.assertEqual(response["suggested_values"]["attendee_count"], 2)

    def test_still_invalid_llm_suggestion_fails_without_booking(self):
        with patch("booking_service.logic.Groq") as fake, patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            fake.return_value.chat.completions.create.return_value.choices = [
                type("Choice", (), {"message": type("Message", (), {"content": json.dumps({
                    "attendee_count": 1,
                    "amount_per_person_inr": 250,
                    "explanation": "Still under the configured minimums.",
                })})()})()
            ]
            response = process_booking(self.request(attendees=1, amount=250), CONFIG, self.store)
        self.assertEqual(response["status"], "failed")
        self.assertNotIn("confirmation_id", self.store.get("request-1")["result"])

    def test_accepted_revised_request_is_checked_and_confirmed(self):
        revised = self.request(request_id="accepted-1", attendees=2, amount=500)
        response = process_booking(revised, CONFIG, self.store)
        self.assertEqual(response["status"], "confirmed")
        self.assertEqual(response["total_amount_inr"], 1000)


if __name__ == "__main__":
    unittest.main()
