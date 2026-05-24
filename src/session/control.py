"""Soft session control for the voice app hardware button.

This layer intentionally changes workflow only at safe boundaries. Button
events request state changes; screening and CBT code decide when to honor them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from src.utils import config_loader
from src.utils.io_record import get_resp_log, log_question, log_system_message
from src.utils.log_util import get_logger

logger = get_logger("SessionControl")


class SessionShutdownRequested(RuntimeError):
    """Raised when the user confirms that CaiTI should close."""


@dataclass(frozen=True)
class SessionControlSettings:
    enabled: bool = False
    pause_message: str = "Caiti is paused. Press the large button again when you are ready to continue."
    resume_message: str = "Okay, we will continue."
    close_prompt: str = (
        "Do you want to close Caiti now? Please say yes to close, "
        "or press and hold the large button again for three seconds."
    )
    close_cancel_message: str = "Okay, we will keep going."


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
    _skip_to_cbt_requested: bool = field(default=False, init=False)
    _shutdown_confirm_requested: bool = field(default=False, init=False)
    _awaiting_shutdown_confirmation: bool = field(default=False, init=False)
    _phase_before_confirmation: str = field(default="waiting_start", init=False)

    def __post_init__(self) -> None:
        self._resume_event.set()
        if not self.settings.enabled:
            self.request_start("disabled")

    def request_start(self, source: str = "button") -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._set_phase_locked("loading")
            self._started_event.set()
            self._publish_button_event(f"start:{source}")
            logger.info("Session start requested by %s.", source)

    def handle_short_press(self) -> None:
        with self._lock:
            if not self._started:
                should_start = True
            else:
                should_start = False

        if should_start:
            self.request_start("button")
            return

        with self._lock:
            if self._awaiting_shutdown_confirmation:
                logger.info("Ignoring short press while waiting for shutdown confirmation.")
                return
            if self._paused:
                self._paused = False
                self._pause_requested = False
                self._resume_event.set()
                self._publish_button_event("resume")
                logger.info("Session resume requested by button.")
                return
            self._pause_requested = True
            self._publish_button_event("pause")
            logger.info("Session pause requested by button.")

    def handle_long_press(self) -> None:
        with self._lock:
            if not self._started:
                logger.info("Ignoring long press before session start.")
                self._publish_button_event("long_press_before_start")
                return
            if self._awaiting_shutdown_confirmation:
                self._publish_button_event("shutdown_confirmed_by_long_press")
                self._request_shutdown_locked("second long press")
                return
            if self._phase in {"screening", "loading"} and not self._skip_to_cbt_requested:
                self._skip_to_cbt_requested = True
                self._paused = False
                self._resume_event.set()
                self._publish_button_event("skip_to_cbt")
                logger.info("Skip-to-CBT requested by long button press.")
                return
            self._shutdown_confirm_requested = True
            self._paused = False
            self._resume_event.set()
            self._publish_button_event("shutdown_confirmation")
            logger.info("Shutdown confirmation requested by long button press.")

    def wait_for_start(self, poll_interval_sec: float = 0.1) -> bool:
        while not self._shutdown_event.is_set():
            if self._started_event.wait(timeout=poll_interval_sec):
                return True
        return False

    def mark_screening(self) -> None:
        with self._lock:
            if self._phase != "confirm_exit":
                self._set_phase_locked("screening")

    def mark_cbt(self) -> None:
        with self._lock:
            if self._phase != "confirm_exit":
                self._set_phase_locked("cbt")

    def mark_closing(self) -> None:
        with self._lock:
            self._set_phase_locked("closing")

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
        if self._consume_shutdown_confirmation_request():
            self._confirm_shutdown_with_voice()
            if self.is_shutdown_requested():
                raise SessionShutdownRequested("Session shutdown requested.")
        if self._consume_pause_request():
            self._pause_until_resumed()
            if self.is_shutdown_requested():
                raise SessionShutdownRequested("Session shutdown requested.")
        if location == "screening" and self._consume_skip_to_cbt_request():
            return "skip_to_cbt"
        return "continue"

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_event.is_set()

    def _consume_pause_request(self) -> bool:
        with self._lock:
            if not self._pause_requested:
                return False
            self._pause_requested = False
            self._paused = True
            self._resume_event.clear()
            return True

    def _pause_until_resumed(self) -> None:
        log_system_message(self.settings.pause_message)
        logger.info("Session paused at safe boundary.")
        while not self._shutdown_event.is_set():
            if self._consume_shutdown_confirmation_request():
                self._confirm_shutdown_with_voice()
                if self.is_shutdown_requested():
                    return
                if not self._paused:
                    break
            if self._resume_event.wait(timeout=0.1):
                break
        if not self.is_shutdown_requested():
            log_system_message(self.settings.resume_message)
            logger.info("Session resumed.")

    def _consume_skip_to_cbt_request(self) -> bool:
        with self._lock:
            if not self._skip_to_cbt_requested:
                return False
            self._skip_to_cbt_requested = False
            return True

    def _consume_shutdown_confirmation_request(self) -> bool:
        with self._lock:
            if not self._shutdown_confirm_requested:
                return False
            self._shutdown_confirm_requested = False
            self._awaiting_shutdown_confirmation = True
            self._phase_before_confirmation = self._phase
            self._set_phase_locked("confirm_exit")
            return True

    def _confirm_shutdown_with_voice(self) -> None:
        logger.info("Asking user to confirm shutdown.")
        log_question(self.settings.close_prompt)
        response = get_resp_log(should_stop=self.is_shutdown_requested)
        if self.is_shutdown_requested():
            return
        if _is_affirmative_shutdown_response(response):
            with self._lock:
                self._request_shutdown_locked("voice confirmation")
            return
        with self._lock:
            self._awaiting_shutdown_confirmation = False
            if self._phase == "confirm_exit":
                self._set_phase_locked(self._phase_before_confirmation)
        log_system_message(self.settings.close_cancel_message)
        logger.info("Shutdown confirmation cancelled by voice response: %r", response)

    def _request_shutdown_locked(self, source: str) -> None:
        self._shutdown_event.set()
        self._resume_event.set()
        self._awaiting_shutdown_confirmation = False
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


def _is_affirmative_shutdown_response(response: object) -> bool:
    text = str(response or "").strip().lower()
    if not text:
        return False
    yes_phrases = {
        "yes",
        "yeah",
        "yep",
        "sure",
        "confirm",
        "confirmed",
        "close",
        "close it",
        "close app",
        "close the app",
        "shut down",
        "shutdown",
        "turn off",
        "exit",
        "quit",
    }
    if text in yes_phrases:
        return True
    tokens = {token.strip(".,!?;:") for token in text.split()}
    return bool(tokens & {"yes", "yeah", "yep", "confirm", "exit", "quit"})


def build_session_control(status_monitor=None) -> SessionControl:
    return SessionControl(
        SessionControlSettings(
            enabled=config_loader.HARDWARE_SESSION_BUTTON_ENABLED,
        ),
        status_monitor=status_monitor,
    )


class NullSessionControl:
    def request_start(self, source: str = "button") -> None:
        return

    def handle_short_press(self) -> None:
        return

    def handle_long_press(self) -> None:
        return

    def wait_for_start(self, poll_interval_sec: float = 0.1) -> bool:
        return True

    def mark_screening(self) -> None:
        return

    def mark_cbt(self) -> None:
        return

    def mark_closing(self) -> None:
        return

    def checkpoint(self, location: str) -> str:
        return "continue"

    def is_shutdown_requested(self) -> bool:
        return False
