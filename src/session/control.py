"""Soft session control for the voice app hardware button."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("SessionControl")


class SessionShutdownRequested(RuntimeError):
    """Raised when the user requests that CaiTI should close."""


@dataclass(frozen=True)
class SessionControlSettings:
    enabled: bool = False
    pause_message: str = "Caiti is paused. Press the button again when you are ready to continue."
    resume_message: str = "Okay, we will continue."


@dataclass
class SessionControl:
    settings: SessionControlSettings
    status_monitor: object | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _started_event: threading.Event = field(default_factory=threading.Event, init=False)
    _resume_event: threading.Event = field(default_factory=threading.Event, init=False)
    _shutdown_event: threading.Event = field(default_factory=threading.Event, init=False)
    _started: bool = field(default=False, init=False)
    _phase: str = field(default="waiting_start", init=False)
    _pause_requested: bool = field(default=False, init=False)
    _paused: bool = field(default=False, init=False)
    _phase_before_pause: str = field(default="waiting_start", init=False)
    _skip_to_cbt_requested: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._resume_event.set()
        if not self.settings.enabled:
            self.request_start("disabled")

    def request_start(self, source: str = "button") -> bool:
        with self._lock:
            if self._started:
                return False
            self._started = True
            self._set_phase_locked("loading")
            self._started_event.set()
            self._publish_button_event(f"start:{source}")
            logger.info("Session start requested by %s.", source)
            return True

    def handle_short_press(self) -> str:
        with self._lock:
            if not self._started:
                should_start = True
            else:
                should_start = False

        if should_start:
            self.request_start("button")
            return "start"

        with self._lock:
            effective_phase = self._phase_before_pause if self._phase == "paused" else self._phase
            if effective_phase not in {"screening", "cbt"}:
                logger.info("Ignoring short press during non-pausable phase: %s", effective_phase)
                self._publish_button_event(f"ignored:{effective_phase}")
                return "ignored_busy"

            if self._paused:
                self._paused = False
                self._pause_requested = False
                self._resume_event.set()
                if self._phase == "paused":
                    self._set_phase_locked(self._phase_before_pause)
                self._publish_button_event("resume")
                logger.info("Session resume requested by button.")
                return "resume"
            self._pause_requested = True
            self._paused = True
            self._phase_before_pause = self._phase
            self._resume_event.clear()
            self._set_phase_locked("paused")
            self._publish_button_event("pause")
            logger.info("Session pause requested by button.")
            return "pause"

    def handle_long_press(self) -> str:
        with self._lock:
            if not self._started:
                logger.info("Ignoring long press before session start.")
                self._publish_button_event("long_press_before_start")
                return "long_press_before_start"
            if self._phase in {"preloading", "loading", "cleanup"}:
                logger.info("Ignoring long press during busy phase: %s", self._phase)
                self._publish_button_event(f"ignored:{self._phase}")
                return "ignored_busy"
            effective_phase = self._phase_before_pause if self._phase == "paused" else self._phase
            if effective_phase == "screening" and not self._skip_to_cbt_requested:
                self._skip_to_cbt_requested = True
                self._paused = False
                self._pause_requested = False
                self._resume_event.set()
                self._set_phase_locked("cbt")
                self._publish_button_event("skip_to_cbt")
                logger.info("Skip-to-CBT requested by long button press.")
                return "skip_to_cbt"
            self._publish_button_event("shutdown_requested_by_long_press")
            self._paused = False
            self._pause_requested = False
            self._resume_event.set()
            self._request_shutdown_locked(f"long press in {effective_phase}")
            return "shutdown_requested_by_long_press"

    def wait_for_start(self, poll_interval_sec: float = 0.1) -> bool:
        while not self._shutdown_event.is_set():
            if self._started_event.wait(timeout=poll_interval_sec):
                return True
        return False

    def mark_screening(self) -> None:
        with self._lock:
            self._set_phase_locked("screening")

    def mark_cbt(self) -> None:
        with self._lock:
            self._set_phase_locked("cbt")

    def mark_closing(self) -> None:
        with self._lock:
            self._set_phase_locked("closing")

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._set_phase_locked(str(phase or "").strip() or "unknown")

    def reset_for_next_session(self) -> None:
        with self._lock:
            self._started = False
            self._started_event.clear()
            self._resume_event.set()
            self._shutdown_event.clear()
            self._phase = "waiting_start"
            self._pause_requested = False
            self._paused = False
            self._phase_before_pause = "waiting_start"
            self._skip_to_cbt_requested = False
            self._set_phase_locked("waiting_start")

    def checkpoint(self, location: str) -> str:
        """Honor pending button events at a safe workflow boundary.

        Returns:
            "continue" for normal flow.
            "skip_to_cbt" when screening should stop and CBT should begin.

        Raises:
            SessionShutdownRequested when shutdown is confirmed.
        """
        if not self.settings.enabled:
            return "continue"
        if self.is_shutdown_requested():
            raise SessionShutdownRequested("Session shutdown requested.")
        if self.is_paused():
            self._pause_until_resumed()
            if self.is_shutdown_requested():
                raise SessionShutdownRequested("Session shutdown requested.")
        if location == "screening" and self._consume_skip_to_cbt_request():
            return "skip_to_cbt"
        return "continue"

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_event.is_set()

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def should_interrupt_voice(self) -> bool:
        with self._lock:
            return bool(
                self._shutdown_event.is_set()
                or self._paused
                or self._skip_to_cbt_requested
            )

    def should_interrupt_workflow_wait(self) -> bool:
        with self._lock:
            return bool(
                self._shutdown_event.is_set()
                or self._skip_to_cbt_requested
            )

    def should_discard_interrupted_voice_turn(self) -> bool:
        with self._lock:
            return bool(
                self._shutdown_event.is_set()
                or self._skip_to_cbt_requested
            )

    def should_keep_music_on_interrupted_voice_turn(self) -> bool:
        with self._lock:
            return bool(
                self._skip_to_cbt_requested
                and not self._shutdown_event.is_set()
            )

    def wait_while_paused(self, poll_interval_sec: float = 0.1) -> None:
        while not self.is_shutdown_requested() and self.is_paused():
            self._resume_event.wait(timeout=poll_interval_sec)

    def _consume_pause_request(self) -> bool:
        with self._lock:
            if not self._paused and not self._pause_requested:
                return False
            self._pause_requested = False
            self._paused = True
            self._resume_event.clear()
            return True

    def _pause_until_resumed(self) -> None:
        self._consume_pause_request()
        logger.info("Session paused.")
        while not self._shutdown_event.is_set():
            if self._resume_event.wait(timeout=0.1):
                break
        if not self.is_shutdown_requested():
            logger.info("Session resumed.")

    def _consume_skip_to_cbt_request(self) -> bool:
        with self._lock:
            if not self._skip_to_cbt_requested:
                return False
            self._skip_to_cbt_requested = False
            return True

    def _request_shutdown_locked(self, source: str) -> None:
        self._shutdown_event.set()
        self._resume_event.set()
        self._set_phase_locked("closing")
        logger.info("Session shutdown requested by %s.", source)

    def _set_phase_locked(self, phase: str) -> None:
        self._phase = phase
        self._publish_phase(phase)

    def _publish_phase(self, phase: str) -> None:
        if self.status_monitor is None:
            return
        method = getattr(self.status_monitor, "set_phase", None)
        if not callable(method):
            return
        try:
            method(phase)
        except Exception as exc:
            logger.warning("Status monitor phase update failed: %s", exc)

    def _publish_button_event(self, event: str) -> None:
        if self.status_monitor is None:
            return
        method = getattr(self.status_monitor, "set_button_event", None)
        if not callable(method):
            return
        try:
            method(event)
        except Exception as exc:
            logger.warning("Status monitor button update failed: %s", exc)


def build_session_control(status_monitor=None) -> SessionControl:
    return SessionControl(
        SessionControlSettings(
            enabled=config_loader.HARDWARE_SESSION_BUTTON_ENABLED,
        ),
        status_monitor=status_monitor,
    )


class NullSessionControl:
    settings = SessionControlSettings(enabled=False)

    def request_start(self, source: str = "button") -> None:
        return

    def handle_short_press(self) -> str:
        return "ignored"

    def handle_long_press(self) -> str:
        return "ignored"

    def should_interrupt_voice(self) -> bool:
        return False

    def should_interrupt_workflow_wait(self) -> bool:
        return False

    def should_discard_interrupted_voice_turn(self) -> bool:
        return False

    def should_keep_music_on_interrupted_voice_turn(self) -> bool:
        return False

    def wait_for_start(self, poll_interval_sec: float = 0.1) -> bool:
        return True

    def mark_screening(self) -> None:
        return

    def mark_cbt(self) -> None:
        return

    def mark_closing(self) -> None:
        return

    def set_phase(self, phase: str) -> None:
        return

    def reset_for_next_session(self) -> None:
        return

    def checkpoint(self, location: str) -> str:
        return "continue"

    def is_shutdown_requested(self) -> bool:
        return False
