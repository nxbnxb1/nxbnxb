"""Persistent result cache (SQLite) keyed by method, engine and the exact pixels of the crop."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


class ResultCache:
    def __init__(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path / "results.sqlite3", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT, created REAL)")
        self._conn.commit()
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM results WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO results (key, value, created) VALUES (?, ?, ?)", (key, payload, time.time())
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
