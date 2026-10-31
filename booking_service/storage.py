"""Atomic JSON persistence for booking outcomes and idempotency records."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class BookingStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self):
        """Serialize lookup, booking creation, and persistence for idempotency."""
        with self._lock:
            yield

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._read()
            result = data.get(request_id)
            return result if isinstance(result, dict) else None

    def put(self, request_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            data = self._read()
            data[request_id] = result
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Booking storage at {self.path} is not valid JSON.") from exc
