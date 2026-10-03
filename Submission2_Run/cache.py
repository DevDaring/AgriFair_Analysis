"""Response cache: one SQLite table keyed by prompt hash. A cached prompt is never re-queried.

Safe for the runner's thread pool: one connection shared under a lock (SQLite serialises writes
anyway, and the payloads are small).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Optional

_COLS = ("key", "tier", "method", "repeat", "route", "model_id", "text", "input_tokens", "output_tokens",
         "latency_seconds", "called_utc", "extra")


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # timeout: another process (a second stage run, the judge) may hold the write lock for a moment
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=120)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS responses (
            key TEXT PRIMARY KEY, tier TEXT, method TEXT, repeat INTEGER, route TEXT,
            model_id TEXT, text TEXT, input_tokens INTEGER, output_tokens INTEGER,
            latency_seconds REAL, called_utc TEXT, extra TEXT)""")
        self.conn.commit()

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM responses WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        d = dict(zip(_COLS, row)); d["extra"] = json.loads(d["extra"] or "{}")
        return d

    def put(self, key: str, tier: str, method: str, repeat: int, route: str, model_id: str, text: str,
            input_tokens: int, output_tokens: int, latency_seconds: float, extra: Optional[Dict] = None) -> None:
        row = (key, tier, method, repeat, route, model_id, text, input_tokens, output_tokens, latency_seconds,
               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), json.dumps(extra or {}, ensure_ascii=False))
        for attempt in range(5):           # a lock held past the timeout must not lose a paid answer
            try:
                with self._lock:
                    self.conn.execute("INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
                    self.conn.commit()
                return
            except sqlite3.OperationalError:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))

    def priced_total(self, price) -> float:
        """Spend of every cached response; price(tier, input_tokens, output_tokens, route) -> USD."""
        with self._lock:
            rows = self.conn.execute("SELECT tier, route, SUM(input_tokens), SUM(output_tokens) FROM responses "
                                     "GROUP BY tier, route").fetchall()
        return float(sum(price(t, i or 0, o or 0, r or "") for t, r, i, o in rows))

    def count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
