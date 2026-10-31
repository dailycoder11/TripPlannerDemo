"""Tests for the preflight trip screen and deterministic extraction checks."""

import json
import unittest
from datetime import date, time
from types import SimpleNamespace
from unittest.mock import patch

from security.preflight import _validate_screening_result, screen_trip_request


class PreflightScreeningTests(unittest.TestCase):
    def setUp(self):
        self.submitted = {
            "destination": "Mumbai",
            "trip_date": date(2026, 10, 31),
            "people": 2,
            "budget_per_person": 2500.0,
            "currency": "INR",
            "start_time": time(9, 0),
            "end_time": time(19, 0),
        }
        self.response = {
            "status": "pass",
            "reason_code": None,
            "message": None,
            "trip": {
                "destination": "Mumbai",
                "trip_date": "2026-10-31",
                "people": 2,
                "budget_per_person": 2500.0,
                "currency": "INR",
                "start_time": "09:00",
                "end_time": "19:00",
            },
        }

    def test_valid_extraction_returns_canonical_form_values(self):
        canonical, error = _validate_screening_result(self.response, self.submitted)
        self.assertIsNone(error)
        self.assertEqual(canonical, self.submitted)

    def test_changed_destination_and_malformed_response_fail(self):
        changed = json.loads(json.dumps(self.response))
        changed["trip"]["destination"] = "Pune"
        canonical, error = _validate_screening_result(changed, self.submitted)
        self.assertIsNone(canonical)
        self.assertEqual(error, "screening_changed_destination")
        canonical, error = _validate_screening_result({"status": "pass"}, self.submitted)
        self.assertIsNone(canonical)
        self.assertEqual(error, "invalid_screening_response")

    def test_flagged_input_returns_only_a_safe_reason_code(self):
        flagged = dict(self.response, status="flag", reason_code="instruction_override")
        canonical, error = _validate_screening_result(flagged, self.submitted)
        self.assertIsNone(canonical)
        self.assertEqual(error, "instruction_override")

    def test_screening_has_no_tools_and_provider_failure_fails_closed(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps(self.response)
        ))])
        with patch("security.preflight.Groq") as fake_groq:
            fake_groq.return_value.chat.completions.create.return_value = completion
            canonical, error = screen_trip_request(
                api_key="test-key",
                model="unit-test-20b",
                **self.submitted,
            )
        self.assertIsNone(error)
        self.assertEqual(canonical, self.submitted)
        request = fake_groq.return_value.chat.completions.create.call_args.kwargs
        self.assertNotIn("tools", request)

        canonical, error = screen_trip_request(
            api_key="",
            model="unit-test-20b",
            **self.submitted,
        )
        self.assertIsNone(canonical)
        self.assertEqual(error, "screening_service_unavailable")


if __name__ == "__main__":
    unittest.main()
