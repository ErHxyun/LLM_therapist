"""GPIO controller for the CaiTI session button."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("SessionButton")


@dataclass(frozen=True)
class SessionButtonSettings:
    enabled: bool = False
    board_pin: int = 37
    long_press_sec: float = 3.0
    debounce_sec: float = 0.5
    poll_interval_sec: float = 0.01
    active_low: bool = True


@dataclass
class _PressState:
    pressed: bool = False
    press_started_at: float | None = None
    long_press_emitted: bool = False
    last_transition_at: float = -1_000_000.0


@dataclass
class NullSessionButtonController:
    def start(self) -> bool:
        return False

    def stop(self) -> None:
        return


@dataclass
class SessionButtonController:
    settings: SessionButtonSettings
    on_short_press: Callable[[], None]
    on_long_press: Callable[[], None]
    gpio_module: object | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _state: _PressState = field(default_factory=_PressState, init=False)

    def start(self) -> bool:
        if not self.settings.enabled:
            logger.info("Session button disabled.")
            return False
        if self._thread and self._thread.is_alive():
            return True
        try:
            gpio = self.gpio_module or self._load_gpio_module()
            self.gpio_module = gpio
            gpio.setmode(gpio.BOARD)
            gpio.setup(self.settings.board_pin, gpio.IN)
            initial_pressed = gpio.input(self.settings.board_pin) == self._pressed_value
            self._state = _PressState(pressed=initial_pressed)
        except Exception as exc:
            logger.warning("Session button disabled: GPIO setup failed: %s", exc)
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="caiti-session-button", daemon=True)
        self._thread.start()
        logger.info("Session button started: BOARD %s", self.settings.board_pin)
        return True

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        self._cleanup_gpio()

    @staticmethod
    def _load_gpio_module():
        try:
            import RPi.GPIO as GPIO
        except ImportError:
            import Jetson.GPIO as GPIO
        return GPIO

    @property
    def _pressed_value(self) -> int:
        return 0 if self.settings.active_low else 1

    def _cleanup_gpio(self) -> None:
        gpio = self.gpio_module
        if gpio is None:
            return
        try:
            gpio.cleanup([self.settings.board_pin])
        except Exception:
            try:
                gpio.cleanup()
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop_event.wait(self.settings.poll_interval_sec):
            try:
                self._process_value(self.gpio_module.input(self.settings.board_pin), time.monotonic())
            except Exception as exc:
                logger.warning("Session button polling failed: %s", exc)
                time.sleep(0.2)

    def _process_value(self, value: int, now: float) -> str | None:
        pressed = value == self._pressed_value
        state = self._state

        if pressed and not state.pressed:
            if now - state.last_transition_at < self.settings.debounce_sec:
                return None
            state.pressed = True
            state.press_started_at = now
            state.long_press_emitted = False
            state.last_transition_at = now
            return None

        if pressed and state.pressed:
            if state.press_started_at is None:
                state.press_started_at = now
            if not state.long_press_emitted and now - state.press_started_at >= self.settings.long_press_sec:
                state.long_press_emitted = True
                logger.info("Button long press detected.")
                self.on_long_press()
                return "long"
            return None

        if not pressed and state.pressed:
            if now - state.last_transition_at < self.settings.debounce_sec:
                return None
            press_started_at = state.press_started_at
            long_emitted = state.long_press_emitted
            state.pressed = False
            state.press_started_at = None
            state.long_press_emitted = False
            state.last_transition_at = now
            if press_started_at is not None and not long_emitted:
                duration = now - press_started_at
                if duration >= self.settings.debounce_sec:
                    logger.info("Button short press detected.")
                    self.on_short_press()
                    return "short"
        return None


def build_session_button_settings() -> SessionButtonSettings:
    return SessionButtonSettings(
        enabled=config_loader.HARDWARE_SESSION_BUTTON_ENABLED,
        board_pin=config_loader.HARDWARE_SESSION_BUTTON_BOARD_PIN,
        long_press_sec=config_loader.HARDWARE_SESSION_BUTTON_LONG_PRESS_SEC,
        debounce_sec=config_loader.HARDWARE_SESSION_BUTTON_DEBOUNCE_SEC,
        poll_interval_sec=config_loader.HARDWARE_SESSION_BUTTON_POLL_INTERVAL_SEC,
        active_low=config_loader.HARDWARE_SESSION_BUTTON_ACTIVE_LOW,
    )


def build_session_button_controller(
    on_short_press: Callable[[], None],
    on_long_press: Callable[[], None],
) -> SessionButtonController | NullSessionButtonController:
    settings = build_session_button_settings()
    if not settings.enabled:
        return NullSessionButtonController()
    return SessionButtonController(settings, on_short_press, on_long_press)
