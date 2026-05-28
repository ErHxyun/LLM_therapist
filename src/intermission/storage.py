"""Persistence for private intermission screening results."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config_loader import DATA_DIR
from src.utils.session_event_logger import get_event_db_path, get_session_id

_DB_LOCK = threading.Lock()
_VALID_STATUSES = {"ANSWERED", "SKIPPED", "UNRESOLVED"}
_DEFAULT_JSON_FILENAME = "phq_gad_results.json"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class IntermissionScreeningStore:
    """Per-session store for PHQ/GAD intermission answers.

    The store shares CaiTI's structured SQLite log by default, but writes only
    intermission-specific tables so these answers stay out of record.csv and
    the paper RL/CBT pipeline. It also mirrors the same PHQ/GAD results into a
    JSON file under data for easier review.
    """

    db_path: str | None = None
    json_path: str | None = None
    session_id: str | None = None

    @property
    def path(self) -> str:
        return self.db_path or get_event_db_path()

    @property
    def results_json_path(self) -> str:
        return self.json_path or os.path.join(DATA_DIR, _DEFAULT_JSON_FILENAME)

    @property
    def current_session_id(self) -> str:
        return self.session_id or get_session_id()

    def init(self) -> None:
        path = Path(self.path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        with _DB_LOCK:
            conn = sqlite3.connect(path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS intermission_screening (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        item_id TEXT NOT NULL,
                        scale TEXT NOT NULL,
                        status TEXT NOT NULL,
                        score INTEGER,
                        response_text TEXT,
                        reason TEXT,
                        updated_at TEXT NOT NULL,
                        UNIQUE(session_id, item_id)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_intermission_screening_session
                    ON intermission_screening (session_id, scale, item_id)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS intermission_screening_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        scale TEXT NOT NULL,
                        total INTEGER NOT NULL,
                        answered INTEGER NOT NULL,
                        expected INTEGER NOT NULL,
                        complete INTEGER NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(session_id, scale)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_intermission_summary_session
                    ON intermission_screening_summary (session_id, scale)
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def upsert_item(
        self,
        *,
        item_id: str,
        scale: str,
        status: str,
        score: int | None,
        response_text: str,
        reason: str,
    ) -> None:
        normalized_status = str(status or "").upper().strip()
        if normalized_status not in _VALID_STATUSES:
            raise ValueError(f"Invalid intermission screening status: {status}")
        self.init()
        with _DB_LOCK:
            conn = sqlite3.connect(self.path)
            try:
                conn.execute(
                    """
                    INSERT INTO intermission_screening (
                        session_id,
                        item_id,
                        scale,
                        status,
                        score,
                        response_text,
                        reason,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, item_id) DO UPDATE SET
                        scale=excluded.scale,
                        status=excluded.status,
                        score=excluded.score,
                        response_text=excluded.response_text,
                        reason=excluded.reason,
                        updated_at=excluded.updated_at
                    """,
                    (
                        self.current_session_id,
                        item_id,
                        scale,
                        normalized_status,
                        score,
                        response_text,
                        reason,
                        _utc_timestamp(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        self._write_results_json()

    def upsert_summary(self, totals: dict[str, dict[str, int | bool]]) -> None:
        self.init()
        with _DB_LOCK:
            conn = sqlite3.connect(self.path)
            try:
                for scale, data in totals.items():
                    conn.execute(
                        """
                        INSERT INTO intermission_screening_summary (
                            session_id,
                            scale,
                            total,
                            answered,
                            expected,
                            complete,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id, scale) DO UPDATE SET
                            total=excluded.total,
                            answered=excluded.answered,
                            expected=excluded.expected,
                            complete=excluded.complete,
                            updated_at=excluded.updated_at
                        """,
                        (
                            self.current_session_id,
                            scale,
                            int(data.get("total", 0)),
                            int(data.get("answered", 0)),
                            int(data.get("expected", 0)),
                            int(bool(data.get("complete", False))),
                            _utc_timestamp(),
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
        self._write_results_json()

    def fetch_items(self) -> list[dict[str, Any]]:
        self.init()
        with _DB_LOCK:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT item_id, scale, status, score, response_text, reason, updated_at
                    FROM intermission_screening
                    WHERE session_id = ?
                    ORDER BY id
                    """,
                    (self.current_session_id,),
                ).fetchall()
            finally:
                conn.close()
        return [dict(row) for row in rows]

    def fetch_summary(self) -> dict[str, dict[str, int | bool | str]]:
        self.init()
        with _DB_LOCK:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT scale, total, answered, expected, complete, updated_at
                    FROM intermission_screening_summary
                    WHERE session_id = ?
                    ORDER BY scale
                    """,
                    (self.current_session_id,),
                ).fetchall()
            finally:
                conn.close()
        return {
            str(row["scale"]): {
                "total": int(row["total"]),
                "answered": int(row["answered"]),
                "expected": int(row["expected"]),
                "complete": bool(row["complete"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    def _write_results_json(self) -> None:
        session_id = self.current_session_id
        updated_at = _utc_timestamp()
        session_payload = {
            "session_id": session_id,
            "updated_at": updated_at,
            "summary": self.fetch_summary(),
            "items": self.fetch_items(),
        }
        path = Path(self.results_json_path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        with _DB_LOCK:
            data: dict[str, Any] = {}
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        data = loaded
                except json.JSONDecodeError:
                    data = {}
            sessions = data.get("sessions")
            if not isinstance(sessions, dict):
                sessions = {}
            sessions[session_id] = session_payload
            data = {
                "version": 1,
                "updated_at": updated_at,
                "sessions": sessions,
            }
            tmp_path = path.with_name(f".{path.name}.tmp")
            tmp_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_path, path)


def build_intermission_store(
    db_path: str | None = None,
    json_path: str | None = None,
) -> IntermissionScreeningStore:
    clean_path = str(db_path or "").strip() or None
    if clean_path:
        clean_path = os.path.expanduser(clean_path)
    clean_json_path = str(json_path or "").strip() or None
    if clean_json_path:
        clean_json_path = os.path.expanduser(clean_json_path)
    return IntermissionScreeningStore(db_path=clean_path, json_path=clean_json_path)
