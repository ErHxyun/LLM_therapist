from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.runtime import user_context
from src.runtime.user_context import UserContext, activate_prepared_user_context
from src.utils import config_loader
from src.utils.session_event_logger import set_session_id

_LOCK = threading.RLock()
_CURRENT_SESSION: "SessionContext | None" = None
_RESUMABLE_STATUSES = {"CREATED", "ACTIVE", "INTERRUPTED"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_id(value: str) -> str:
    normalized = user_context.normalize_spoken_user_id(value)
    number = participant_number(normalized)
    if number is not None:
        return f"{number:03d}" if number < 1000 else str(number)
    safe = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    return safe or user_context.build_guest_user_id().lower()


def participant_number(value: str) -> int | None:
    text = user_context.normalize_spoken_user_id(value).strip()
    if not text.isdigit():
        return None
    try:
        return int(text)
    except ValueError:
        return None


def is_protected_participant(value: str) -> bool:
    if not config_loader.SESSION_ENABLED:
        return False
    if not config_loader.SESSION_PROTECT_PRE_CUTOVER_PARTICIPANTS:
        return False
    number = participant_number(value)
    return number is not None and 1 <= number < config_loader.SESSION_CUTOVER_PARTICIPANT_NUMBER


def uses_session_storage(value: str) -> bool:
    if not config_loader.SESSION_ENABLED:
        return False
    number = participant_number(value)
    return number is None or number >= config_loader.SESSION_CUTOVER_PARTICIPANT_NUMBER


@dataclass(frozen=True)
class SessionContext:
    participant_id: str
    raw_participant_id: str
    display_name: str
    session_id: str
    user_dir: str
    session_dir: str
    manifest_path: str
    record_csv: str
    question_lib_path: str
    report_file: str
    notes_file: str
    q_tables_dir: str
    emotion_results_jsonl_path: str
    emotion_audio_dir: str
    intermission_results_json_path: str
    structured_log_db_path: str
    session_log_file: str
    profile_path: str
    result_dir: str
    log_dir: str
    resumed: bool = False

    def runtime_user_context(self) -> UserContext:
        return UserContext(
            subject_id=self.participant_id,
            raw_subject_id=self.raw_participant_id,
            display_name=self.display_name,
            user_dir=self.session_dir,
            log_dir=self.log_dir,
            result_dir=self.result_dir,
            record_csv=self.record_csv,
            question_lib_path=self.question_lib_path,
            report_file=self.report_file,
            notes_file=self.notes_file,
            q_tables_dir=self.q_tables_dir,
            emotion_results_jsonl_path=self.emotion_results_jsonl_path,
            emotion_audio_dir=self.emotion_audio_dir,
            intermission_results_json_path=self.intermission_results_json_path,
            structured_log_db_path=self.structured_log_db_path,
            session_log_file=self.session_log_file,
            profile_path=self.profile_path,
        )


def _users_root() -> Path:
    return Path(user_context._USERS_ROOT_DIR)


def _question_template() -> Path:
    return Path(user_context._DEFAULT_QUESTION_LIB_PATH)


def _session_context_from_dir(
    *,
    participant_id: str,
    raw_participant_id: str,
    display_name: str,
    session_dir: Path,
    resumed: bool,
) -> SessionContext:
    session_id = session_dir.name
    result_dir = session_dir / "results"
    log_dir = session_dir / "logs"
    return SessionContext(
        participant_id=participant_id,
        raw_participant_id=raw_participant_id,
        display_name=" ".join(str(display_name or "").split()),
        session_id=session_id,
        user_dir=str(session_dir.parent.parent),
        session_dir=str(session_dir),
        manifest_path=str(session_dir / "session.json"),
        record_csv=str(session_dir / "record.csv"),
        question_lib_path=str(session_dir / "libs" / _question_template().name),
        report_file=str(result_dir / f"Report_{participant_id}.csv"),
        notes_file=str(result_dir / f"Notes_{participant_id}.csv"),
        q_tables_dir=str(session_dir / "q_tables"),
        emotion_results_jsonl_path=str(session_dir / "emotion" / "results.jsonl"),
        emotion_audio_dir=str(session_dir / "emotion" / "audio"),
        intermission_results_json_path=str(
            session_dir / "intermission" / "phq_gad_results.json"
        ),
        structured_log_db_path=str(log_dir / "session_events.sqlite3"),
        session_log_file=str(log_dir / f"session_{participant_id}_{session_id}.log"),
        profile_path=str(session_dir.parent.parent / "profile.json"),
        result_dir=str(result_dir),
        log_dir=str(log_dir),
        resumed=resumed,
    )


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _ensure_profile(context: SessionContext) -> None:
    path = Path(context.profile_path)
    existing: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}
    payload = {
        "subject_id": context.participant_id,
        "raw_subject_id": context.raw_participant_id,
        "display_name": context.display_name,
        "user_dir": context.user_dir,
        "updated_at": _utc_now(),
        "created_at": existing.get("created_at", _utc_now()),
    }
    _write_json_atomic(path, payload)


def _manifest_payload(context: SessionContext, status: str, **updates) -> dict:
    path = Path(context.manifest_path)
    payload: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}
    payload.update(
        {
            "version": 1,
            "participant_id": context.participant_id,
            "raw_participant_id": context.raw_participant_id,
            "display_name": context.display_name,
            "session_id": context.session_id,
            "status": status,
            "session_dir": context.session_dir,
            "updated_at": _utc_now(),
        }
    )
    payload.update(updates)
    return payload


def update_session_status(context: SessionContext, status: str, **updates) -> None:
    normalized = str(status or "").strip().upper()
    if normalized not in {"CREATED", "ACTIVE", "COMPLETED", "INTERRUPTED", "ABANDONED"}:
        raise ValueError(f"Unsupported session status: {status}")
    payload = _manifest_payload(context, normalized, **updates)
    if normalized == "ACTIVE" and not payload.get("started_at"):
        payload["started_at"] = _utc_now()
    if normalized == "COMPLETED":
        payload["completed_at"] = _utc_now()
    if normalized == "INTERRUPTED":
        payload["interrupted_at"] = _utc_now()
    if normalized == "ABANDONED":
        payload["abandoned_at"] = _utc_now()
    _write_json_atomic(Path(context.manifest_path), payload)


def create_new_session(participant_id: str, display_name: str = "") -> SessionContext:
    if is_protected_participant(participant_id):
        raise PermissionError(f"Participant {participant_id} is protected")
    if not uses_session_storage(participant_id):
        raise ValueError(f"Participant {participant_id} does not use session storage")

    raw_id = user_context.normalize_spoken_user_id(participant_id)
    safe_id = _safe_id(raw_id)
    session_id = (
        f"{safe_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    session_dir = _users_root() / safe_id / "sessions" / session_id
    context = _session_context_from_dir(
        participant_id=safe_id,
        raw_participant_id=raw_id,
        display_name=display_name,
        session_dir=session_dir,
        resumed=False,
    )
    for directory in (
        Path(context.session_dir),
        Path(context.question_lib_path).parent,
        Path(context.q_tables_dir),
        Path(context.result_dir),
        Path(context.log_dir),
        Path(context.emotion_results_jsonl_path).parent,
        Path(context.emotion_audio_dir),
        Path(context.intermission_results_json_path).parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_question_template(), context.question_lib_path)
    _ensure_profile(context)
    update_session_status(context, "CREATED", created_at=_utc_now())
    return context


def find_resumable_session(participant_id: str) -> SessionContext | None:
    if not config_loader.SESSION_RESUME_INCOMPLETE:
        return None
    raw_id = user_context.normalize_spoken_user_id(participant_id)
    safe_id = _safe_id(raw_id)
    sessions_dir = _users_root() / safe_id / "sessions"
    if not sessions_dir.exists():
        return None
    candidates: list[tuple[str, Path, dict]] = []
    for manifest in sessions_dir.glob("*/session.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("status", "")).upper() not in _RESUMABLE_STATUSES:
            continue
        candidates.append((str(payload.get("updated_at", "")), manifest.parent, payload))
    if not candidates:
        return None
    _, session_dir, payload = max(candidates, key=lambda item: item[0])
    return _session_context_from_dir(
        participant_id=safe_id,
        raw_participant_id=str(payload.get("raw_participant_id", raw_id)),
        display_name=str(payload.get("display_name", "")),
        session_dir=session_dir,
        resumed=True,
    )


def activate_session_context(context: SessionContext) -> SessionContext:
    global _CURRENT_SESSION
    with _LOCK:
        activate_prepared_user_context(context.runtime_user_context())
        set_session_id(context.session_id)
        update_session_status(context, "ACTIVE", resumed=bool(context.resumed))
        _CURRENT_SESSION = context
    return context


def get_current_session_context() -> SessionContext | None:
    with _LOCK:
        return _CURRENT_SESSION


def complete_current_session() -> SessionContext | None:
    context = get_current_session_context()
    if context is not None:
        update_session_status(context, "COMPLETED")
    return context


def interrupt_current_session(reason: str = "") -> SessionContext | None:
    context = get_current_session_context()
    if context is None:
        return None
    manifest = Path(context.manifest_path)
    status = ""
    try:
        status = str(json.loads(manifest.read_text(encoding="utf-8")).get("status", ""))
    except Exception:
        pass
    if status.upper() != "COMPLETED":
        update_session_status(context, "INTERRUPTED", interruption_reason=str(reason or ""))
    return context


def deactivate_session_context() -> None:
    global _CURRENT_SESSION
    with _LOCK:
        _CURRENT_SESSION = None
        user_context.deactivate_user_context()


__all__ = [
    "SessionContext",
    "activate_session_context",
    "complete_current_session",
    "create_new_session",
    "deactivate_session_context",
    "find_resumable_session",
    "get_current_session_context",
    "interrupt_current_session",
    "is_protected_participant",
    "participant_number",
    "update_session_status",
    "uses_session_storage",
]
