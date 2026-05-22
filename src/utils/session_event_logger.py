"""SQLite structured event logging for therapist review."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.local_llm.routing import resolve_adapter
from src.local_llm.types import LLMTask
from src.utils.config_loader import LOG_DIR, SUBJECT_ID

_DB_LOCK = threading.Lock()
_SESSION_ID = os.environ.get(
    "CAITI_SESSION_ID",
    f"{SUBJECT_ID}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
)


def get_session_id() -> str:
    return os.environ.get("CAITI_SESSION_ID", _SESSION_ID)


def get_event_db_path() -> str:
    return os.environ.get(
        "CAITI_STRUCTURED_LOG_DB",
        os.path.join(LOG_DIR, "session_events.sqlite3"),
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_dumps(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def init_structured_log(db_path: str | None = None) -> None:
    path = Path(db_path or get_event_db_path())
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    with _DB_LOCK:
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    turn_index INTEGER,
                    task TEXT NOT NULL,
                    adapter TEXT,
                    dimension TEXT,
                    score TEXT,
                    segment_text TEXT,
                    question_text TEXT,
                    raw_llm_output TEXT,
                    normalized_output TEXT,
                    latency_ms REAL,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_events_session_time
                ON session_events (session_id, timestamp)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_events_task
                ON session_events (task)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_events_dimension
                ON session_events (dimension)
                """
            )
            conn.commit()
        finally:
            conn.close()


def log_llm_event(
    *,
    task: LLMTask | str,
    dimension: str | None = None,
    score: int | str | None = None,
    segment_text: str | None = None,
    question_text: str | None = None,
    raw_llm_output: str | None = None,
    normalized_output: str | None = None,
    adapter: str | None = None,
    turn_index: int | None = None,
    latency_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    db_path: str | None = None,
) -> None:
    task_value = task.value if isinstance(task, LLMTask) else str(task)
    if adapter is None and isinstance(task, LLMTask) and task != LLMTask.BASE:
        adapter = resolve_adapter(task)

    init_structured_log(db_path)
    with _DB_LOCK:
        conn = sqlite3.connect(db_path or get_event_db_path())
        try:
            conn.execute(
                """
                INSERT INTO session_events (
                    session_id,
                    timestamp,
                    turn_index,
                    task,
                    adapter,
                    dimension,
                    score,
                    segment_text,
                    question_text,
                    raw_llm_output,
                    normalized_output,
                    latency_ms,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id or get_session_id(),
                    _utc_timestamp(),
                    turn_index,
                    task_value,
                    adapter,
                    dimension,
                    None if score is None else str(score),
                    segment_text,
                    question_text,
                    raw_llm_output,
                    normalized_output,
                    latency_ms,
                    _json_dumps(metadata),
                ),
            )
            conn.commit()
        finally:
            conn.close()


__all__ = [
    "get_event_db_path",
    "get_session_id",
    "init_structured_log",
    "log_llm_event",
]
