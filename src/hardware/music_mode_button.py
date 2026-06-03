"""GPIO controller for cycling CaiTI background music modes."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("MusicModeButton")


@dataclass(frozen=True)
class MusicModeButtonSettings:
    enabled: bool = False
    board_pin: int = 35
    debounce_sec: float = 0.5
    release_sec: float = 0.2
    poll_interval_sec: float = 0.01
    active_low: bool = True


@dataclass
class _ButtonState:
    armed: bool
    last_press_at: float = -1_000_000.0
    release_started_at: float | None = None


@dataclass
class NullMusicModeButtonController:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return


@dataclass
class MusicModeButtonController:
    settings: MusicModeButtonSettings
    on_press: Callable[[], None]
    gpio_module: object | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _state: _ButtonState = field(default_factory=lambda: _ButtonState(armed=True), init=False)

    def start(self) -> None:
        if not self.settings.enabled:
            logger.info("Music mode button disabled.")
            return
        if self._thread and self._thread.is_alive():
            return
        try:
            gpio = self.gpio_module or self._load_gpio_module()
            self.gpio_module = gpio
            gpio.setmode(gpio.BOARD)
            gpio.setup(self.settings.board_pin, gpio.IN)
            self._state = _ButtonState(armed=(gpio.input(self.settings.board_pin) != self._pressed_value))
        except Exception as exc:
            logger.warning("Music mode button disabled: GPIO setup failed: %s", exc)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="caiti-music-mode-button", daemon=True)
        self._thread.start()
        logger.info("Music mode button started: BOARD %s", self.settings.board_pin)

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
                logger.warning("Music mode button polling failed: %s", exc)
                time.sleep(0.2)

    def _process_value(self, value: int, now: float) -> bool:
        pressed = value == self._pressed_value
        state = self._state
        if pressed:
            state.release_started_at = None
            if not state.armed:
                return False
            if now - state.last_press_at < self.settings.debounce_sec:
                return False
            state.armed = False
            state.last_press_at = now
            logger.info("Music mode button press detected.")
            self.on_press()
            return True

        if state.release_started_at is None:
            state.release_started_at = now
        if not state.armed and now - state.release_started_at >= self.settings.release_sec:
            state.armed = True
        return False


def build_music_mode_button_settings() -> MusicModeButtonSettings:
    return MusicModeButtonSettings(
        enabled=config_loader.HARDWARE_MUSIC_MODE_BUTTON_ENABLED,
        board_pin=config_loader.HARDWARE_MUSIC_MODE_BUTTON_BOARD_PIN,
        debounce_sec=config_loader.HARDWARE_MUSIC_MODE_BUTTON_DEBOUNCE_SEC,
        release_sec=config_loader.HARDWARE_MUSIC_MODE_BUTTON_RELEASE_SEC,
        poll_interval_sec=config_loader.HARDWARE_MUSIC_MODE_BUTTON_POLL_INTERVAL_SEC,
        active_low=config_loader.HARDWARE_MUSIC_MODE_BUTTON_ACTIVE_LOW,
    )


def build_music_mode_button_controller(
    on_press: Callable[[], None],
) -> MusicModeButtonController | NullMusicModeButtonController:
    settings = build_music_mode_button_settings()
    if not settings.enabled:
        return NullMusicModeButtonController()
    return MusicModeButtonController(settings, on_press)
