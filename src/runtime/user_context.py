from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from src.runtime.status_monitor import get_active_status_monitor
from src.utils import config_loader, log_util

_LOCK = threading.Lock()
_DEFAULT_SUBJECT_ID = str(config_loader.SUBJECT_ID)
_DEFAULT_DATA_DIR = str(config_loader.DATA_DIR)
_DEFAULT_LOG_DIR = str(config_loader.LOG_DIR)
_DEFAULT_RESULT_DIR = str(config_loader.RESULT_DIR)
_DEFAULT_RECORD_CSV = str(config_loader.RECORD_CSV)
_DEFAULT_QUESTION_LIB_PATH = str(config_loader.QUESTION_LIB_FILENAME)
_DEFAULT_REPORT_FILE = str(config_loader.REPORT_FILE)
_DEFAULT_NOTES_FILE = str(config_loader.NOTES_FILE)
_USERS_ROOT_DIR = os.path.join(_DEFAULT_DATA_DIR, "users")
_CURRENT_CONTEXT = None


@dataclass(frozen=True)
class UserContext:
    subject_id: str
    raw_subject_id: str
    display_name: str
    user_dir: str
    log_dir: str
    result_dir: str
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify_fragment(value: str, fallback: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or fallback


_NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def normalize_spoken_user_id(raw_value: str) -> str:
    text = " ".join(str(raw_value or "").strip().split())
    if not text:
        return ""
    lowered = text.lower().replace("-", " ")
    tokens = [token for token in lowered.split() if token]
    if tokens and all(token in _NUMBER_WORDS for token in tokens):
        return "".join(_NUMBER_WORDS[token] for token in tokens)
    return text


def build_guest_user_id() -> str:
    return f"guest_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def build_user_context(subject_id: str, display_name: str = "") -> UserContext:
    normalized_id = normalize_spoken_user_id(subject_id) or build_guest_user_id()
    safe_id = _slugify_fragment(normalized_id, _slugify_fragment(_DEFAULT_SUBJECT_ID, "user"))
    user_dir = os.path.join(_USERS_ROOT_DIR, safe_id)
    log_dir = os.path.join(user_dir, "logs")
    result_dir = os.path.join(user_dir, "results")
    record_csv = os.path.join(user_dir, "record.csv")
    question_lib_path = os.path.join(user_dir, "libs", os.path.basename(_DEFAULT_QUESTION_LIB_PATH))
    report_file = os.path.join(result_dir, f"Report_{safe_id}.csv")
    notes_file = os.path.join(result_dir, f"Notes_{safe_id}.csv")
    q_tables_dir = os.path.join(user_dir, "q_tables")
    emotion_dir = os.path.join(user_dir, "emotion")
    intermission_dir = os.path.join(user_dir, "intermission")
    session_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_log_file = os.path.join(log_dir, f"session_{safe_id}_{session_tag}.log")
    return UserContext(
        subject_id=safe_id,
        raw_subject_id=normalized_id,
        display_name=" ".join(str(display_name or "").strip().split()),
        user_dir=user_dir,
        log_dir=log_dir,
        result_dir=result_dir,
        record_csv=record_csv,
        question_lib_path=question_lib_path,
        report_file=report_file,
        notes_file=notes_file,
        q_tables_dir=q_tables_dir,
        emotion_results_jsonl_path=os.path.join(emotion_dir, "results.jsonl"),
        emotion_audio_dir=os.path.join(emotion_dir, "audio"),
        intermission_results_json_path=os.path.join(intermission_dir, "phq_gad_results.json"),
        structured_log_db_path=os.path.join(log_dir, "session_events.sqlite3"),
        session_log_file=session_log_file,
        profile_path=os.path.join(user_dir, "profile.json"),
    )


def ensure_user_context_files(context: UserContext) -> None:
    for path in (
        context.user_dir,
        context.log_dir,
        context.result_dir,
        os.path.dirname(context.record_csv),
        os.path.dirname(context.question_lib_path),
        context.q_tables_dir,
        os.path.dirname(context.emotion_results_jsonl_path),
        context.emotion_audio_dir,
        os.path.dirname(context.intermission_results_json_path),
    ):
        os.makedirs(path, exist_ok=True)
    if not os.path.exists(context.question_lib_path):
        shutil.copyfile(_DEFAULT_QUESTION_LIB_PATH, context.question_lib_path)
    profile_payload = {
        "subject_id": context.subject_id,
        "raw_subject_id": context.raw_subject_id,
        "display_name": context.display_name,
        "user_dir": context.user_dir,
        "updated_at": _now_iso(),
    }
    if os.path.exists(context.profile_path):
        try:
            with open(context.profile_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except Exception:
            existing = {}
        if "created_at" in existing:
            profile_payload["created_at"] = existing["created_at"]
    if "created_at" not in profile_payload:
        profile_payload["created_at"] = _now_iso()
    with open(context.profile_path, "w", encoding="utf-8") as handle:
        json.dump(profile_payload, handle, ensure_ascii=False, indent=2)


def _patch_module_attr(module_name: str, attr_name: str, value) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    setattr(module, attr_name, value)


def _patch_runtime_paths(context: UserContext) -> None:
    config_loader.SUBJECT_ID = context.subject_id
    config_loader.DATA_DIR = context.user_dir
    config_loader.LOG_DIR = context.log_dir
    config_loader.RESULT_DIR = context.result_dir
    config_loader.QUESTION_LIB_FILENAME = context.question_lib_path
    config_loader.REPORT_FILE = context.report_file
    config_loader.NOTES_FILE = context.notes_file
    config_loader.RECORD_CSV = context.record_csv
    config_loader.EMOTION_USER_ID = context.subject_id
    config_loader.EMOTION_RESULTS_JSONL_PATH = context.emotion_results_jsonl_path
    config_loader.EMOTION_AUDIO_DIR = context.emotion_audio_dir
    config_loader.INTERMISSION_DB_PATH = context.structured_log_db_path
    config_loader.INTERMISSION_RESULTS_JSON_PATH = context.intermission_results_json_path

    for module_name, updates in {
        "src.handler_rl": {
            "SUBJECT_ID": context.subject_id,
            "DATA_DIR": context.user_dir,
            "RESULT_DIR": context.result_dir,
            "QUESTION_LIB_FILENAME": context.question_lib_path,
            "REPORT_FILE": context.report_file,
            "NOTES_FILE": context.notes_file,
            "RECORD_CSV": context.record_csv,
        },
        "src.utils.io_question_lib": {
            "REPORT_FILE": context.report_file,
            "NOTES_FILE": context.notes_file,
        },
        "src.utils.io_record": {
            "RECORD_CSV": context.record_csv,
        },
        "src.voice.io_loop": {
            "RECORD_CSV": context.record_csv,
        },
        "src.utils.session_event_logger": {
            "SUBJECT_ID": context.subject_id,
            "LOG_DIR": context.log_dir,
        },
        "src.utils.log_util": {
            "LOG_DIR": context.log_dir,
        },
    }.items():
        for attr_name, value in updates.items():
            _patch_module_attr(module_name, attr_name, value)


def activate_user_context(subject_id: str, display_name: str = "") -> UserContext:
    global _CURRENT_CONTEXT
    context = build_user_context(subject_id, display_name)
    ensure_user_context_files(context)
    with _LOCK:
        _CURRENT_CONTEXT = context
        os.environ["CAITI_RUNTIME_USER_ID"] = context.subject_id
        os.environ["CAITI_RUNTIME_USER_NAME"] = context.display_name
        os.environ["CAITI_STRUCTURED_LOG_DB"] = context.structured_log_db_path
        os.environ["CAITI_EMOTION_USER_ID"] = context.subject_id
        os.environ["CAITI_EMOTION_RESULTS_JSONL_PATH"] = context.emotion_results_jsonl_path
        os.environ["CAITI_EMOTION_AUDIO_DIR"] = context.emotion_audio_dir
        os.environ["CAITI_INTERMISSION_DB_PATH"] = context.structured_log_db_path
        os.environ["CAITI_INTERMISSION_RESULTS_JSON_PATH"] = context.intermission_results_json_path
        _patch_runtime_paths(context)
        log_util.set_session_log_file(context.session_log_file)
        monitor = get_active_status_monitor()
        if monitor is not None:
            setter = getattr(monitor, "set_user", None)
            if callable(setter):
                setter(
                    subject_id=context.subject_id,
                    raw_subject_id=context.raw_subject_id,
                    display_name=context.display_name,
                    user_dir=context.user_dir,
                )
    return context


def get_current_user_context() -> UserContext | None:
    return _CURRENT_CONTEXT


__all__ = [
    "UserContext",
    "activate_user_context",
    "build_guest_user_id",
    "build_user_context",
    "get_current_user_context",
    "normalize_spoken_user_id",
]
