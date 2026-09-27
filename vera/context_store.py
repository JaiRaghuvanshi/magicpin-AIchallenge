"""
In-memory context store.

Implements the contract from challenge-testing-brief.md §2.1:
- Keyed by (scope, context_id)
- Idempotent on (context_id, version): re-posting the same version is a no-op
- A higher version for the same context_id replaces the prior version atomically
- Persists until teardown (no restarts assumed mid-test)

Swap this for Redis/SQLite in production; the interface is intentionally small.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal

Scope = Literal["category", "merchant", "customer", "trigger"]


@dataclass
class StoredContext:
    version: int
    payload: dict[str, Any]


class ContextStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[tuple[str, str], StoredContext] = {}

    def put(self, scope: Scope, context_id: str, version: int, payload: dict[str, Any]) -> tuple[bool, dict]:
        """
        Returns (accepted, response_extra).
        response_extra carries 'current_version' on rejection.
        """
        key = (scope, context_id)
        with self._lock:
            current = self._data.get(key)
            if current is not None and current.version >= version:
                return False, {"current_version": current.version}
            self._data[key] = StoredContext(version=version, payload=payload)
            return True, {}

    def get(self, scope: Scope, context_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._data.get((scope, context_id))
            return entry.payload if entry else None

    def get_version(self, scope: Scope, context_id: str) -> int | None:
        with self._lock:
            entry = self._data.get((scope, context_id))
            return entry.version if entry else None

    def counts(self) -> dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._lock:
            for (scope, _key) in self._data.keys():
                counts[scope] = counts.get(scope, 0) + 1
        return counts

    def all_ids(self, scope: Scope) -> list[str]:
        with self._lock:
            return [cid for (s, cid) in self._data.keys() if s == scope]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
