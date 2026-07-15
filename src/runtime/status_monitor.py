"""Small local HTTP/SSE monitor for CaiTI runtime state."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("StatusMonitor")
_ACTIVE_STATUS_MONITOR = None


@dataclass(frozen=True)
class StatusMonitorSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class NullStatusMonitor:
    def start(self) -> bool:
        return False

    def stop(self) -> None:
        return

    @property
    def url(self) -> str:
        return ""

    def snapshot(self) -> dict[str, Any]:
        return {}

    def set_phase(self, phase: str) -> None:
        return

    def set_light(self, color: str, active: bool) -> None:
        return

    def set_button_event(self, event: str) -> None:
        return

    def set_user(self, **kwargs) -> None:
        return

    def set_session(self, session_id: str, started_at: str | None = None) -> None:
        return

    def set_prompt(
        self,
        *,
        text: str,
        source: str = "",
        expects_response: bool = False,
        item_id: str = "",
        question_index: str = "",
        dimension: str = "",
    ) -> None:
        return

    def set_response(self, *, text: str, source: str = "") -> None:
        return

    def set_score(
        self,
        *,
        item_id: str,
        question_index: str,
        dimension: str,
        score: Any,
        user_input: str = "",
        classification: list[Any] | None = None,
        followup_text: str = "",
    ) -> None:
        return

    def set_emotion_result(self, record: dict[str, Any]) -> None:
        return

    def set_intermission_state(self, *, summary: dict[str, Any], items: list[dict[str, Any]]) -> None:
        return


@dataclass
class StatusMonitor:
    settings: StatusMonitorSettings
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _condition: threading.Condition = field(init=False)
    _server: ThreadingHTTPServer | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _version: int = field(default=0, init=False)
    _state: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)
        now = _now_iso()
        self._state = {
            "app": "CaiTI",
            "phase": "booting",
            "lights": {
                "white": False,
                "yellow": False,
                "blue": False,
                "green": False,
            },
            "button": {
                "last_event": "",
                "updated_at": "",
            },
            "user": {
                "subject_id": "",
                "raw_subject_id": "",
                "display_name": "",
                "user_dir": "",
                "updated_at": "",
            },
            "session": {
                "id": "",
                "started_at": "",
                "turn_count": 0,
                "updated_at": "",
            },
            "current_prompt": {
                "text": "",
                "source": "",
                "expects_response": False,
                "item_id": "",
                "question_index": "",
                "dimension": "",
                "updated_at": "",
            },
            "latest_response": {
                "text": "",
                "source": "",
                "updated_at": "",
            },
            "latest_score": {
                "item_id": "",
                "question_index": "",
                "dimension": "",
                "score": None,
                "user_input": "",
                "classification": [],
                "followup_text": "",
                "updated_at": "",
            },
            "emotion": {
                "latest": {},
                "recent": [],
                "updated_at": "",
            },
            "intermission": {
                "summary": {},
                "items": [],
                "updated_at": "",
            },
            "recent_events": [],
            "started_at": now,
            "updated_at": now,
            "version": 0,
        }

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self.settings.host}:{self.settings.port}"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> bool:
        if not self.settings.enabled:
            logger.info("Status monitor disabled.")
            return False
        with self._lock:
            if self._started:
                return True
            try:
                server = _StatusThreadingHTTPServer((self.settings.host, self.settings.port), _StatusRequestHandler)
                server.status_monitor = self
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                self._server = server
                self._thread = thread
                self._started = True
            except Exception as exc:
                self._server = None
                self._thread = None
                self._started = False
                logger.warning("Status monitor failed to start: %s", exc)
                return False
        logger.info("Status monitor started at %s", self.url)
        return True

    def stop(self) -> None:
        server = None
        thread = None
        with self._lock:
            if not self._started:
                return
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
            self._started = False
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=1.0)
        logger.info("Status monitor stopped.")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def wait_for_update(self, last_version: int, timeout_sec: float = 15.0) -> dict[str, Any]:
        with self._condition:
            if self._version <= last_version:
                self._condition.wait(timeout=timeout_sec)
            return self.snapshot()

    def set_phase(self, phase: str) -> None:
        phase = str(phase or "").strip() or "unknown"

        def mutate(state: dict[str, Any]) -> None:
            previous = str(state.get("phase", ""))
            state["phase"] = phase
            if phase != previous:
                self._record_event(
                    state,
                    kind="phase",
                    title=f"Phase changed to {phase}",
                    detail=f"Previous phase: {previous or 'unknown'}",
                )

        self._update(mutate)

    def set_light(self, color: str, active: bool) -> None:
        color = str(color or "").strip().lower()
        if color == "red":
            color = "blue"
        if color not in {"white", "yellow", "blue", "green"}:
            return

        def mutate(state: dict[str, Any]) -> None:
            state["lights"][color] = bool(active)

        self._update(mutate)

    def set_button_event(self, event: str) -> None:
        event = str(event or "").strip()

        def mutate(state: dict[str, Any]) -> None:
            state["button"]["last_event"] = event
            state["button"]["updated_at"] = _now_iso()
            if event:
                self._record_event(
                    state,
                    kind="button",
                    title="Button event",
                    detail=event,
                )

        self._update(mutate)

    def set_user(
        self,
        *,
        subject_id: str,
        raw_subject_id: str = "",
        display_name: str = "",
        user_dir: str = "",
    ) -> None:
        def mutate(state: dict[str, Any]) -> None:
            previous = dict(state.get("user", {}))
            state["user"] = {
                "subject_id": str(subject_id or ""),
                "raw_subject_id": str(raw_subject_id or ""),
                "display_name": str(display_name or ""),
                "user_dir": str(user_dir or ""),
                "updated_at": _now_iso(),
            }
            if (
                previous.get("subject_id") != state["user"]["subject_id"]
                or previous.get("display_name") != state["user"]["display_name"]
            ):
                label = state["user"]["display_name"] or state["user"]["subject_id"] or "Unknown user"
                detail_parts = []
                if state["user"]["subject_id"]:
                    detail_parts.append(f"ID: {state['user']['subject_id']}")
                if state["user"]["display_name"]:
                    detail_parts.append(f"Name: {state['user']['display_name']}")
                self._record_event(
                    state,
                    kind="user",
                    title=f"Active user: {label}",
                    detail=" | ".join(detail_parts),
                )

        self._update(mutate)

    def set_session(self, session_id: str, started_at: str | None = None) -> None:
        started = str(started_at or "").strip() or _now_iso()

        def mutate(state: dict[str, Any]) -> None:
            state["session"] = {
                "id": str(session_id or ""),
                "started_at": started,
                "turn_count": 0,
                "updated_at": _now_iso(),
            }
            state["current_prompt"] = {
                "text": "",
                "source": "",
                "expects_response": False,
                "item_id": "",
                "question_index": "",
                "dimension": "",
                "updated_at": "",
            }
            state["latest_response"] = {
                "text": "",
                "source": "",
                "updated_at": "",
            }
            state["latest_score"] = {
                "item_id": "",
                "question_index": "",
                "dimension": "",
                "score": None,
                "user_input": "",
                "classification": [],
                "followup_text": "",
                "updated_at": "",
            }
            state["emotion"] = {
                "latest": {},
                "recent": [],
                "updated_at": "",
            }
            state["intermission"] = {
                "summary": {},
                "items": [],
                "updated_at": "",
            }
            state["recent_events"] = []
            if session_id:
                self._record_event(
                    state,
                    kind="session",
                    title=f"Session started: {session_id}",
                    detail=f"Started at {started}",
                )

        self._update(mutate)

    def set_prompt(
        self,
        *,
        text: str,
        source: str = "",
        expects_response: bool = False,
        item_id: str = "",
        question_index: str = "",
        dimension: str = "",
    ) -> None:
        def mutate(state: dict[str, Any]) -> None:
            existing = dict(state.get("current_prompt", {}))
            state["current_prompt"] = {
                "text": str(text or ""),
                "source": str(source or ""),
                "expects_response": bool(expects_response),
                "item_id": str(item_id or existing.get("item_id", "")),
                "question_index": str(question_index or existing.get("question_index", "")),
                "dimension": str(dimension or existing.get("dimension", "")),
                "updated_at": _now_iso(),
            }
            prompt_text = str(text or "").strip()
            if prompt_text:
                prompt_type = "Question asked" if expects_response else "System message"
                prompt_meta = []
                if dimension or existing.get("dimension", ""):
                    prompt_meta.append(f"dimension={dimension or existing.get('dimension', '')}")
                if item_id or existing.get("item_id", ""):
                    prompt_meta.append(f"item={item_id or existing.get('item_id', '')}")
                self._record_event(
                    state,
                    kind="prompt",
                    title=prompt_type,
                    detail=" | ".join([prompt_text, *prompt_meta]) if prompt_meta else prompt_text,
                )

        self._update(mutate)

    def set_response(self, *, text: str, source: str = "") -> None:
        def mutate(state: dict[str, Any]) -> None:
            state["latest_response"] = {
                "text": str(text or ""),
                "source": str(source or ""),
                "updated_at": _now_iso(),
            }
            response_text = str(text or "").strip()
            if response_text:
                self._record_event(
                    state,
                    kind="response",
                    title="User response received",
                    detail=response_text,
                )

        self._update(mutate)

    def set_score(
        self,
        *,
        item_id: str,
        question_index: str,
        dimension: str,
        score: Any,
        user_input: str = "",
        classification: list[Any] | None = None,
        followup_text: str = "",
    ) -> None:
        def mutate(state: dict[str, Any]) -> None:
            state["latest_score"] = {
                "item_id": str(item_id or ""),
                "question_index": str(question_index or ""),
                "dimension": str(dimension or ""),
                "score": score,
                "user_input": str(user_input or ""),
                "classification": list(classification or []),
                "followup_text": str(followup_text or ""),
                "updated_at": _now_iso(),
            }
            session = dict(state.get("session", {}))
            session["turn_count"] = int(session.get("turn_count", 0)) + 1
            session["updated_at"] = _now_iso()
            state["session"] = session
            title = f"Score {score}"
            if dimension:
                title = f"{title} · {dimension}"
            detail_parts = []
            if user_input:
                detail_parts.append(f"user: {user_input}")
            if followup_text:
                detail_parts.append(f"follow-up: {followup_text}")
            self._record_event(
                state,
                kind="score",
                title=title,
                detail=" | ".join(detail_parts),
                level="warn" if score in {1, 2, "1", "2"} else "info",
            )

        self._update(mutate)

    def set_emotion_result(self, record: dict[str, Any]) -> None:
        summary = _summarize_emotion_record(record)

        def mutate(state: dict[str, Any]) -> None:
            recent = list(state.get("emotion", {}).get("recent", []))
            if summary:
                recent.append(summary)
            recent = recent[-10:]
            state["emotion"] = {
                "latest": summary,
                "recent": recent,
                "updated_at": _now_iso(),
            }
            if summary:
                detail_parts = []
                if summary.get("risk") is not None:
                    detail_parts.append(f"risk={summary.get('risk')}")
                if summary.get("risk_level"):
                    detail_parts.append(f"level={summary.get('risk_level')}")
                if summary.get("contradiction_or_sarcasm") is not None:
                    detail_parts.append(f"sarcasm={summary.get('contradiction_or_sarcasm')}")
                if summary.get("confidence") is not None:
                    detail_parts.append(f"confidence={summary.get('confidence')}")
                self._record_event(
                    state,
                    kind="emotion",
                    title="Emotion side-channel updated",
                    detail=" | ".join(detail_parts),
                    level="warn" if summary.get("contradiction_or_sarcasm") else "info",
                )

        self._update(mutate)

    def set_intermission_state(self, *, summary: dict[str, Any], items: list[dict[str, Any]]) -> None:
        def mutate(state: dict[str, Any]) -> None:
            state["intermission"] = {
                "summary": json.loads(json.dumps(summary or {})),
                "items": json.loads(json.dumps(items or [])),
                "updated_at": _now_iso(),
            }
            if summary or items:
                summary_parts = []
                for scale, data in (summary or {}).items():
                    if isinstance(data, dict):
                        summary_parts.append(
                            f"{scale}: total={data.get('total', 0)}, answered={data.get('answered', 0)}, complete={data.get('complete', False)}"
                        )
                self._record_event(
                    state,
                    kind="intermission",
                    title="Intermission updated",
                    detail=" | ".join(summary_parts),
                )

        self._update(mutate)

    def _record_event(
        self,
        state: dict[str, Any],
        *,
        kind: str,
        title: str,
        detail: str = "",
        level: str = "info",
    ) -> None:
        events = list(state.get("recent_events", []))
        events.append(
            {
                "timestamp": _now_iso(),
                "kind": str(kind or ""),
                "title": str(title or ""),
                "detail": _truncate_text(str(detail or ""), 480),
                "level": str(level or "info"),
                "phase": str(state.get("phase", "")),
            }
        )
        state["recent_events"] = events[-30:]

    def _update(self, mutate) -> None:
        with self._condition:
            mutate(self._state)
            self._version += 1
            self._state["version"] = self._version
            self._state["updated_at"] = _now_iso()
            self._condition.notify_all()


class _StatusRequestHandler(BaseHTTPRequestHandler):
    server_version = "CaiTIStatusMonitor/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html()
            return
        if path == "/status":
            self._send_json(self._monitor().snapshot())
            return
        if path == "/history":
            self._send_json(build_history_snapshot())
            return
        if path == "/events":
            self._send_events()
            return
        self.send_error(404, "Not found")

    def log_message(self, format: str, *args) -> None:
        logger.debug("HTTP %s", format % args)

    def _monitor(self) -> StatusMonitor:
        return self.server.status_monitor

    def _send_json(self, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_body(body)

    def _send_html(self) -> None:
        body = _HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_body(body)

    def _write_body(self, body: bytes) -> None:
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def _send_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        last_version = -1
        try:
            while True:
                snapshot = self._monitor().wait_for_update(last_version)
                last_version = int(snapshot.get("version", last_version))
                payload = json.dumps(snapshot, ensure_ascii=False)
                self.wfile.write(f"event: status\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return


class _StatusThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        _, exc, _ = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError, OSError)):
            logger.debug("Status monitor client disconnected: %s", exc)
            return
        super().handle_error(request, client_address)

    def shutdown_request(self, request) -> None:
        try:
            super().shutdown_request(request)
        except OSError:
            return


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _truncate_text(value: str, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def set_active_status_monitor(monitor) -> None:
    global _ACTIVE_STATUS_MONITOR
    _ACTIVE_STATUS_MONITOR = monitor


def get_active_status_monitor():
    return _ACTIVE_STATUS_MONITOR


def _data_root_dir() -> Path:
    configured = str(config_loader.PATHS.get("data_dir", "data"))
    expanded = configured.replace("${subject_id}", str(config_loader.APP.get("subject_id", "")))
    return Path(os.path.abspath(expanded))


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return {}


def _load_session_summaries(result_dir: Path, limit: int) -> list[dict[str, Any]]:
    sessions = []
    for path in sorted(result_dir.glob("SessionSummary_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        data = _safe_load_json(path)
        sessions.append(
            {
                "file": str(path),
                "run_id": str(data.get("run_id", "")),
                "timestamp": str(data.get("timestamp", "")),
                "screening_turn_count": int(data.get("screening_turn_count", 0) or 0),
                "cbt_used": bool(data.get("cbt_used", False)),
                "cbt_candidates": list(data.get("cbt_candidates", []) or []),
            }
        )
    return sessions


def _load_recent_emotion(user_dir: Path) -> dict[str, Any]:
    path = user_dir / "emotion" / "results.jsonl"
    if not path.exists():
        return {}
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return {}
        record = json.loads(lines[-1])
        return _summarize_emotion_record(record)
    except Exception:
        return {}


def _load_recent_intermission(user_dir: Path) -> dict[str, Any]:
    path = user_dir / "intermission" / "phq_gad_results.json"
    if not path.exists():
        return {}
    return _safe_load_json(path)


def build_history_snapshot(limit_users: int = 20, limit_sessions_per_user: int = 5) -> dict[str, Any]:
    data_root = _data_root_dir()
    users_root = data_root / "users"
    users: list[dict[str, Any]] = []
    if users_root.exists():
        for user_dir in sorted(
            [path for path in users_root.iterdir() if path.is_dir()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:limit_users]:
            profile = _safe_load_json(user_dir / "profile.json")
            users.append(
                {
                    "subject_id": str(profile.get("subject_id", user_dir.name)),
                    "raw_subject_id": str(profile.get("raw_subject_id", "")),
                    "display_name": str(profile.get("display_name", "")),
                    "updated_at": str(profile.get("updated_at", "")),
                    "created_at": str(profile.get("created_at", "")),
                    "user_dir": str(user_dir),
                    "sessions": _load_session_summaries(user_dir / "results", limit_sessions_per_user),
                    "latest_emotion": _load_recent_emotion(user_dir),
                    "intermission": _load_recent_intermission(user_dir),
                }
            )

    legacy_users: dict[str, list[dict[str, Any]]] = {}
    legacy_result_dir = data_root / "results"
    if legacy_result_dir.exists():
        for path in sorted(legacy_result_dir.glob("SessionSummary_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            data = _safe_load_json(path)
            subject_id = str(data.get("subject_id", "legacy") or "legacy")
            legacy_users.setdefault(subject_id, [])
            if len(legacy_users[subject_id]) >= limit_sessions_per_user:
                continue
            legacy_users[subject_id].append(
                {
                    "file": str(path),
                    "run_id": str(data.get("run_id", "")),
                    "timestamp": str(data.get("timestamp", "")),
                    "screening_turn_count": int(data.get("screening_turn_count", 0) or 0),
                    "cbt_used": bool(data.get("cbt_used", False)),
                    "cbt_candidates": list(data.get("cbt_candidates", []) or []),
                }
            )

    return {
        "generated_at": _now_iso(),
        "users": users,
        "legacy_users": [
            {
                "subject_id": subject_id,
                "sessions": sessions,
            }
            for subject_id, sessions in sorted(legacy_users.items())
        ],
    }


def _summarize_emotion_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    final_assessment = response.get("final_assessment", {}) if isinstance(response, dict) else {}
    final_result = response.get("final_result", {}) if isinstance(response, dict) else {}
    comparison = response.get("emotion_comparison", {}) if isinstance(response, dict) else {}
    audio_scores = response.get("audio_scores", {}) if isinstance(response, dict) else {}
    text_scores = response.get("text_scores", {}) if isinstance(response, dict) else {}
    request_payload = record.get("request") if isinstance(record.get("request"), dict) else {}
    return {
        "status": str(record.get("status", "")),
        "utterance_id": str(record.get("utterance_id", "")),
        "created_at": record.get("created_at"),
        "latency_sec": record.get("latency_sec"),
        "transcript": str(request_payload.get("transcript", response.get("transcript", ""))),
        "risk": final_result.get("credibility_risk", final_assessment.get("credibility_risk")),
        "risk_level": final_result.get("risk_level", final_assessment.get("risk_level")),
        "confidence": final_assessment.get("confidence"),
        "uncertainty": final_assessment.get("uncertainty"),
        "audio_emotion": response.get("audio_emotion"),
        "context_emotion": response.get("context_emotion"),
        "consistent": comparison.get("audio_vs_context_consistent"),
        "arousal_conflict": comparison.get("arousal_conflict"),
        "valence_conflict": comparison.get("valence_conflict"),
        "contradiction_or_sarcasm": comparison.get("contradiction_or_sarcasm"),
        "audio_arousal": audio_scores.get("arousal"),
        "audio_tension": audio_scores.get("tension"),
        "audio_stability": audio_scores.get("stability"),
        "text_valence": text_scores.get("text_valence"),
        "certainty": text_scores.get("certainty"),
    }


def build_status_monitor_settings() -> StatusMonitorSettings:
    return StatusMonitorSettings(
        enabled=config_loader.MONITOR_ENABLED,
        host=config_loader.MONITOR_HOST,
        port=config_loader.MONITOR_PORT,
    )


def build_status_monitor() -> StatusMonitor | NullStatusMonitor:
    settings = build_status_monitor_settings()
    if not settings.enabled:
        return NullStatusMonitor()
    return StatusMonitor(settings)


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CaiTI Research Console</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f8;
      color: #151923;
      --surface: #ffffff;
      --surface-alt: #f8fafc;
      --line: #d9dee7;
      --line-soft: #e8ebf0;
      --muted: #5e6878;
      --text-soft: #364152;
      --blue: #2563eb;
      --green: #1f9d55;
      --amber: #b7791f;
      --rose: #be123c;
      --slate: #64748b;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      padding: 20px;
      background: #f5f6f8;
      color: #151923;
    }
    main {
      width: min(1440px, 100%);
      margin: 0 auto;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 24px;
      padding: 10px 0 20px;
      border-bottom: 1px solid var(--line);
    }
    .eyebrow {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      letter-spacing: 0;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    h1 {
      margin: 0;
      font-size: 32px;
      line-height: 1.1;
      font-weight: 760;
    }
    .subhead {
      margin: 10px 0 0;
      max-width: 760px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 15px;
    }
    .top-meta {
      min-width: 260px;
      display: grid;
      gap: 8px;
      justify-items: end;
      text-align: right;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--surface);
      font-size: 13px;
      font-weight: 700;
    }
    .status-pill::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--slate);
    }
    .status-pill.live::before { background: var(--green); box-shadow: 0 0 0 5px rgba(31,157,85,.12); }
    .status-pill.syncing::before { background: var(--amber); box-shadow: 0 0 0 5px rgba(183,121,31,.12); }
    .status-pill.offline::before { background: var(--rose); box-shadow: 0 0 0 5px rgba(190,18,60,.12); }
    .meta-text {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .band {
      margin-top: 22px;
      padding: 0;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 12px;
    }
    .section-head h2 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }
    .section-note {
      color: var(--muted);
      font-size: 13px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
    }
    .metric {
      min-height: 118px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .metric-label {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 700;
    }
    .metric-value {
      font-size: 24px;
      line-height: 1.1;
      font-weight: 760;
      word-break: break-word;
    }
    .metric-sub {
      color: var(--text-soft);
      font-size: 13px;
      line-height: 1.4;
    }
    .light-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .light-card,
    .panel,
    .history-card,
    .event-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }
    .light-card {
      padding: 16px;
      min-height: 132px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .light-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .light-dot {
      width: 34px;
      height: 34px;
      border-radius: 999px;
      background: #afb8c5;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,.12);
    }
    .light-card.on .light-dot { box-shadow: 0 0 24px currentColor, inset 0 0 0 1px rgba(0,0,0,.12); }
    .light-status {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
    }
    .light-card.white { color: #6b7280; }
    .light-card.white.on .light-dot { background: #fbfdff; }
    .light-card.yellow { color: var(--amber); }
    .light-card.yellow.on .light-dot { background: #facc15; }
    .light-card.blue { color: var(--blue); }
    .light-card.blue.on .light-dot { background: #3b82f6; }
    .light-card.green { color: var(--green); }
    .light-card.green.on .light-dot { background: #22c55e; }
    .light-name {
      font-size: 18px;
      font-weight: 730;
      text-transform: capitalize;
    }
    .light-help {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.4;
    }
    .content-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, .95fr);
      gap: 16px;
      align-items: start;
    }
    .stack {
      display: grid;
      gap: 16px;
    }
    .panel {
      padding: 16px;
    }
    .panel h3 {
      margin: 0 0 12px;
      font-size: 16px;
      line-height: 1.2;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 14px;
      margin-bottom: 12px;
    }
    .detail-item {
      min-height: 54px;
      padding: 10px 12px;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: var(--surface-alt);
    }
    .detail-item.wide {
      grid-column: 1 / -1;
    }
    .detail-key {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .detail-value {
      color: #151923;
      font-size: 14px;
      line-height: 1.45;
      font-weight: 650;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .text-block {
      min-height: 96px;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      background: var(--surface-alt);
      padding: 14px;
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 14px;
      word-break: break-word;
    }
    .timeline {
      display: grid;
      gap: 10px;
    }
    .event-item {
      padding: 12px 14px;
      display: grid;
      gap: 8px;
    }
    .event-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    .event-kind {
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: #edf2ff;
      color: #2f3c86;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .event-kind.warn {
      background: #fff3d6;
      color: #8a5a00;
    }
    .event-time {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    .event-title {
      font-size: 15px;
      font-weight: 720;
      line-height: 1.35;
    }
    .event-detail {
      color: var(--text-soft);
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .history-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .history-card {
      padding: 16px;
      display: grid;
      gap: 10px;
    }
    .history-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }
    .history-title {
      font-size: 17px;
      font-weight: 760;
      line-height: 1.25;
    }
    .history-sub {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      word-break: break-word;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 700;
    }
    .session-list {
      display: grid;
      gap: 10px;
    }
    .session-item {
      padding-top: 10px;
      border-top: 1px solid var(--line-soft);
      display: grid;
      gap: 4px;
    }
    .session-summary {
      font-size: 14px;
      line-height: 1.45;
      color: #151923;
      font-weight: 650;
    }
    .session-path {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      word-break: break-word;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 18px;
      color: var(--muted);
      background: var(--surface);
      font-size: 14px;
    }
    .legacy-wrap {
      margin-top: 14px;
      display: grid;
      gap: 12px;
    }
    .muted {
      color: var(--muted);
    }
    @media (max-width: 1200px) {
      .metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .history-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 980px) {
      .content-grid { grid-template-columns: 1fr; }
      .light-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .topbar { flex-direction: column; align-items: flex-start; }
      .top-meta { justify-items: start; text-align: left; }
    }
    @media (max-width: 680px) {
      body { padding: 14px; }
      .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .detail-grid { grid-template-columns: 1fr; }
      .light-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header class="topbar">
      <div>
        <div class="eyebrow">CaiTI Research Console</div>
        <h1>Live Session Monitor</h1>
        <p class="subhead">Track the active participant, current question, latest answer, LED state, emotion side-channel, intermission screening, and recent session history from one page.</p>
      </div>
      <div class="top-meta">
        <div class="status-pill syncing" id="connection-pill">Connecting</div>
        <div class="meta-text" id="connection-text">Waiting for live updates...</div>
        <div class="meta-text" id="served-url"></div>
      </div>
    </header>

    <section class="band">
      <div class="metric-grid">
        <article class="metric">
          <div class="metric-label">System Phase</div>
          <div class="metric-value" id="metric-phase">-</div>
          <div class="metric-sub" id="metric-phase-sub">Waiting for runtime state</div>
        </article>
        <article class="metric">
          <div class="metric-label">Active User</div>
          <div class="metric-value" id="metric-user">-</div>
          <div class="metric-sub" id="metric-user-sub">No participant selected</div>
        </article>
        <article class="metric">
          <div class="metric-label">Session</div>
          <div class="metric-value" id="metric-session">-</div>
          <div class="metric-sub" id="metric-session-sub">No active session yet</div>
        </article>
        <article class="metric">
          <div class="metric-label">Turn Count</div>
          <div class="metric-value" id="metric-turns">0</div>
          <div class="metric-sub" id="metric-turns-sub">Scored screening turns</div>
        </article>
        <article class="metric">
          <div class="metric-label">Latest Score</div>
          <div class="metric-value" id="metric-score">-</div>
          <div class="metric-sub" id="metric-score-sub">No scored answer yet</div>
        </article>
        <article class="metric">
          <div class="metric-label">Emotion Check</div>
          <div class="metric-value" id="metric-emotion">-</div>
          <div class="metric-sub" id="metric-emotion-sub">No emotion result yet</div>
        </article>
      </div>
    </section>

    <section class="band">
      <div class="section-head">
        <h2>LED State</h2>
        <div class="section-note">White = system running, yellow = session active, blue = system speaking, green = user speaking</div>
      </div>
      <div class="light-grid" id="lights"></div>
    </section>

    <section class="band content-grid">
      <div class="stack">
        <section class="panel">
          <h3>Current Prompt</h3>
          <div class="detail-grid">
            <div class="detail-item">
              <div class="detail-key">Source</div>
              <div class="detail-value" id="prompt-source">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Expects Response</div>
              <div class="detail-value" id="prompt-response">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Dimension</div>
              <div class="detail-value" id="prompt-dimension">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Question Index</div>
              <div class="detail-value" id="prompt-question">-</div>
            </div>
          </div>
          <div class="text-block" id="prompt-text">-</div>
        </section>

        <section class="panel">
          <h3>Latest Response</h3>
          <div class="detail-grid">
            <div class="detail-item">
              <div class="detail-key">Source</div>
              <div class="detail-value" id="response-source">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Updated</div>
              <div class="detail-value" id="response-updated">-</div>
            </div>
          </div>
          <div class="text-block" id="response-text">-</div>
        </section>

        <section class="panel">
          <div class="section-head">
            <h3>Recent Activity</h3>
            <div class="section-note" id="event-count">0 events</div>
          </div>
          <div class="timeline" id="recent-events"></div>
        </section>
      </div>

      <div class="stack">
        <section class="panel">
          <h3>Runtime Details</h3>
          <div class="detail-grid">
            <div class="detail-item">
              <div class="detail-key">Button</div>
              <div class="detail-value" id="runtime-button">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Updated</div>
              <div class="detail-value" id="runtime-updated">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">User ID</div>
              <div class="detail-value" id="runtime-user-id">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Display Name</div>
              <div class="detail-value" id="runtime-user-name">-</div>
            </div>
            <div class="detail-item wide">
              <div class="detail-key">User Folder</div>
              <div class="detail-value" id="runtime-user-dir">-</div>
            </div>
            <div class="detail-item wide">
              <div class="detail-key">Session ID</div>
              <div class="detail-value" id="runtime-session-id">-</div>
            </div>
          </div>
        </section>

        <section class="panel">
          <h3>Latest Score</h3>
          <div class="detail-grid">
            <div class="detail-item">
              <div class="detail-key">Dimension</div>
              <div class="detail-value" id="score-dimension">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Score</div>
              <div class="detail-value" id="score-value">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Item ID</div>
              <div class="detail-value" id="score-item">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Question</div>
              <div class="detail-value" id="score-question">-</div>
            </div>
            <div class="detail-item wide">
              <div class="detail-key">Follow-up</div>
              <div class="detail-value" id="score-followup">-</div>
            </div>
          </div>
        </section>

        <section class="panel">
          <h3>Emotion Side-Channel</h3>
          <div class="detail-grid">
            <div class="detail-item">
              <div class="detail-key">Risk</div>
              <div class="detail-value" id="emotion-risk">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Risk Level</div>
              <div class="detail-value" id="emotion-risk-level">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Confidence</div>
              <div class="detail-value" id="emotion-confidence">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Latency</div>
              <div class="detail-value" id="emotion-latency">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Sarcasm / Contradiction</div>
              <div class="detail-value" id="emotion-sarcasm">-</div>
            </div>
            <div class="detail-item">
              <div class="detail-key">Consistency</div>
              <div class="detail-value" id="emotion-consistent">-</div>
            </div>
          </div>
          <div class="text-block" id="emotion-transcript">-</div>
        </section>

        <section class="panel">
          <h3>Intermission</h3>
          <div class="text-block" id="intermission-summary">-</div>
        </section>
      </div>
    </section>

    <section class="band">
      <div class="section-head">
        <h2>User History</h2>
        <div class="section-note" id="history-updated">Refreshing...</div>
      </div>
      <div class="history-grid" id="history-users"></div>
      <div class="legacy-wrap" id="history-legacy"></div>
    </section>
  </main>
  <script>
    const meanings = {
      white: "Core application is loaded and alive.",
      yellow: "A main session is active.",
      blue: "CaiTI is speaking.",
      green: "The participant is speaking."
    };

    const lightsRoot = document.getElementById("lights");
    for (const color of ["white", "yellow", "blue", "green"]) {
      const card = document.createElement("article");
      card.className = `light-card ${color}`;
      card.id = `light-${color}`;
      card.innerHTML = `
        <div class="light-top">
          <div class="light-dot"></div>
          <div class="light-status" id="light-status-${color}">off</div>
        </div>
        <div>
          <div class="light-name">${color}</div>
          <div class="light-help">${meanings[color]}</div>
        </div>
      `;
      lightsRoot.appendChild(card);
    }

    function safe(value, fallback = "-") {
      if (value === null || value === undefined || value === "") return fallback;
      return String(value);
    }

    function setText(id, value, fallback = "-") {
      const el = document.getElementById(id);
      if (el) el.textContent = safe(value, fallback);
    }

    function boolText(value) {
      if (value === true) return "yes";
      if (value === false) return "no";
      return "-";
    }

    function formatEmotionSummary(emotion) {
      if (!emotion || Object.keys(emotion).length === 0) return "No emotion result yet";
      const parts = [];
      if (emotion.risk_level) parts.push(emotion.risk_level);
      if (emotion.risk !== undefined && emotion.risk !== null && emotion.risk !== "") parts.push(`risk ${emotion.risk}`);
      if (emotion.contradiction_or_sarcasm === true) parts.push("possible mismatch");
      return parts.join(" · ") || "Emotion result available";
    }

    function formatIntermission(intermission) {
      const summary = intermission && intermission.summary ? intermission.summary : {};
      const items = intermission && Array.isArray(intermission.items) ? intermission.items : [];
      const lines = [];
      for (const [scale, data] of Object.entries(summary)) {
        lines.push(`${scale}: total=${safe(data.total, 0)}, answered=${safe(data.answered, 0)}, expected=${safe(data.expected, 0)}, complete=${safe(data.complete, false)}`);
      }
      for (const item of items.slice(-6)) {
        lines.push(`${safe(item.item_id)} · ${safe(item.scale)} · ${safe(item.status)} · score=${safe(item.score)}`);
      }
      return lines.length ? lines.join("\\n") : "-";
    }

    function renderRecentEvents(events) {
      const root = document.getElementById("recent-events");
      root.innerHTML = "";
      const list = Array.isArray(events) ? events.slice(-12).reverse() : [];
      document.getElementById("event-count").textContent = `${list.length} events`;
      if (!list.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No recent activity yet.";
        root.appendChild(empty);
        return;
      }
      for (const item of list) {
        const article = document.createElement("article");
        article.className = "event-item";

        const meta = document.createElement("div");
        meta.className = "event-meta";

        const badge = document.createElement("span");
        badge.className = `event-kind ${item.level === "warn" ? "warn" : ""}`;
        badge.textContent = safe(item.kind, "event");

        const time = document.createElement("span");
        time.className = "event-time";
        time.textContent = safe(item.timestamp);

        meta.appendChild(badge);
        meta.appendChild(time);

        const title = document.createElement("div");
        title.className = "event-title";
        title.textContent = safe(item.title);

        const detail = document.createElement("div");
        detail.className = "event-detail";
        detail.textContent = safe(item.detail);

        article.appendChild(meta);
        article.appendChild(title);
        article.appendChild(detail);
        root.appendChild(article);
      }
    }

    function renderStatus(state) {
      const lights = state.lights || {};
      for (const color of ["white", "yellow", "blue", "green"]) {
        const card = document.getElementById(`light-${color}`);
        const active = Boolean(lights[color]);
        if (card) card.classList.toggle("on", active);
        setText(`light-status-${color}`, active ? "on" : "off");
      }

      const user = state.user || {};
      const session = state.session || {};
      const prompt = state.current_prompt || {};
      const response = state.latest_response || {};
      const score = state.latest_score || {};
      const emotion = state.emotion && state.emotion.latest ? state.emotion.latest : {};

      setText("metric-phase", state.phase);
      setText("metric-phase-sub", `Updated ${safe(state.updated_at)}`);
      setText("metric-user", user.display_name || user.subject_id);
      setText("metric-user-sub", user.subject_id ? `ID ${safe(user.subject_id)}` : "No participant selected");
      setText("metric-session", session.id);
      setText("metric-session-sub", session.started_at ? `Started ${safe(session.started_at)}` : "No active session yet");
      setText("metric-turns", session.turn_count, "0");
      setText("metric-turns-sub", prompt.dimension ? `Current dimension: ${safe(prompt.dimension)}` : "Scored screening turns");
      setText("metric-score", score.score);
      setText("metric-score-sub", score.dimension ? `${safe(score.dimension)} · item ${safe(score.item_id)}` : "No scored answer yet");
      setText("metric-emotion", emotion.risk_level || emotion.risk);
      setText("metric-emotion-sub", formatEmotionSummary(emotion));

      setText("runtime-button", state.button && state.button.last_event);
      setText("runtime-updated", state.updated_at);
      setText("runtime-user-id", user.subject_id);
      setText("runtime-user-name", user.display_name);
      setText("runtime-user-dir", user.user_dir);
      setText("runtime-session-id", session.id);

      setText("prompt-source", prompt.source);
      setText("prompt-response", boolText(prompt.expects_response));
      setText("prompt-dimension", prompt.dimension);
      setText("prompt-question", prompt.question_index);
      setText("prompt-text", prompt.text);

      setText("response-source", response.source);
      setText("response-updated", response.updated_at);
      setText("response-text", response.text);

      setText("score-dimension", score.dimension);
      setText("score-value", score.score);
      setText("score-item", score.item_id);
      setText("score-question", score.question_index);
      setText("score-followup", score.followup_text);

      setText("emotion-risk", emotion.risk);
      setText("emotion-risk-level", emotion.risk_level);
      setText("emotion-confidence", emotion.confidence);
      setText("emotion-latency", emotion.latency_sec);
      setText("emotion-sarcasm", emotion.contradiction_or_sarcasm);
      setText("emotion-consistent", emotion.consistent);
      setText("emotion-transcript", emotion.transcript);

      setText("intermission-summary", formatIntermission(state.intermission || {}));
      renderRecentEvents(state.recent_events || []);

      document.getElementById("connection-pill").className = "status-pill live";
      document.getElementById("connection-pill").textContent = `Live · v${safe(state.version, 0)}`;
      document.getElementById("connection-text").textContent = `Receiving updates from ${window.location.origin}`;
      document.getElementById("served-url").textContent = `Status API: ${window.location.origin}/status`;
    }

    function appendSessionList(root, sessions) {
      const list = document.createElement("div");
      list.className = "session-list";
      if (!Array.isArray(sessions) || !sessions.length) {
        const empty = document.createElement("div");
        empty.className = "session-item muted";
        empty.textContent = "No session summaries yet.";
        list.appendChild(empty);
        root.appendChild(list);
        return;
      }
      for (const session of sessions) {
        const item = document.createElement("div");
        item.className = "session-item";

        const summary = document.createElement("div");
        summary.className = "session-summary";
        summary.textContent = `${safe(session.timestamp)} · turns ${safe(session.screening_turn_count, 0)} · CBT ${session.cbt_used ? "yes" : "no"}`;

        const path = document.createElement("div");
        path.className = "session-path";
        path.textContent = safe(session.file);

        item.appendChild(summary);
        item.appendChild(path);
        list.appendChild(item);
      }
      root.appendChild(list);
    }

    function renderHistory(data) {
      const usersRoot = document.getElementById("history-users");
      const legacyRoot = document.getElementById("history-legacy");
      usersRoot.innerHTML = "";
      legacyRoot.innerHTML = "";
      setText("history-updated", `History refreshed ${safe(data.generated_at)}`);

      const users = Array.isArray(data.users) ? data.users : [];
      const legacyUsers = Array.isArray(data.legacy_users) ? data.legacy_users : [];

      if (!users.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No per-user session folders yet.";
        usersRoot.appendChild(empty);
      } else {
        for (const user of users) {
          const card = document.createElement("article");
          card.className = "history-card";

          const head = document.createElement("div");
          head.className = "history-head";

          const left = document.createElement("div");
          const title = document.createElement("div");
          title.className = "history-title";
          title.textContent = safe(user.display_name || user.subject_id, "Unknown user");
          const sub = document.createElement("div");
          sub.className = "history-sub";
          sub.textContent = safe(user.user_dir);
          left.appendChild(title);
          left.appendChild(sub);

          head.appendChild(left);
          card.appendChild(head);

          const chips = document.createElement("div");
          chips.className = "chip-row";
          if (user.subject_id) {
            const chip = document.createElement("span");
            chip.className = "chip";
            chip.textContent = `ID ${user.subject_id}`;
            chips.appendChild(chip);
          }
          if (user.latest_emotion && user.latest_emotion.risk_level) {
            const chip = document.createElement("span");
            chip.className = "chip";
            chip.textContent = `Emotion ${safe(user.latest_emotion.risk_level)}`;
            chips.appendChild(chip);
          }
          const intermissionSummary = user.intermission && typeof user.intermission === "object" ? Object.keys(user.intermission) : [];
          if (intermissionSummary.length) {
            const chip = document.createElement("span");
            chip.className = "chip";
            chip.textContent = `Intermission ${intermissionSummary.join("/")}`;
            chips.appendChild(chip);
          }
          if (chips.childNodes.length) card.appendChild(chips);

          appendSessionList(card, user.sessions || []);
          usersRoot.appendChild(card);
        }
      }

      if (legacyUsers.length) {
        const title = document.createElement("div");
        title.className = "section-note";
        title.textContent = "Legacy shared result folders";
        legacyRoot.appendChild(title);

        for (const user of legacyUsers) {
          const card = document.createElement("article");
          card.className = "history-card";

          const label = document.createElement("div");
          label.className = "history-title";
          label.textContent = safe(user.subject_id);
          card.appendChild(label);

          appendSessionList(card, user.sessions || []);
          legacyRoot.appendChild(card);
        }
      }
    }

    async function pollStatus() {
      const res = await fetch("/status", { cache: "no-store" });
      renderStatus(await res.json());
    }

    async function pollHistory() {
      const res = await fetch("/history", { cache: "no-store" });
      renderHistory(await res.json());
    }

    async function bootstrap() {
      try {
        await pollStatus();
      } catch (_err) {
        document.getElementById("connection-pill").className = "status-pill offline";
        document.getElementById("connection-pill").textContent = "Offline";
        document.getElementById("connection-text").textContent = "Unable to fetch current status yet.";
      }
      try {
        await pollHistory();
      } catch (_err) {
        setText("history-updated", "History unavailable");
      }
    }

    bootstrap();
    setInterval(pollHistory, 8000);

    if ("EventSource" in window) {
      const events = new EventSource("/events");
      events.addEventListener("status", event => {
        renderStatus(JSON.parse(event.data));
      });
      events.onerror = () => {
        document.getElementById("connection-pill").className = "status-pill syncing";
        document.getElementById("connection-pill").textContent = "Reconnecting";
        document.getElementById("connection-text").textContent = "Live stream dropped. Retrying...";
      };
    } else {
      setInterval(pollStatus, 1200);
    }
  </script>
</body>
</html>
"""
