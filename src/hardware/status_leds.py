"""Status LED controller for CaiTI on the Jetson 40-pin header."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("StatusLEDs")


@dataclass(frozen=True)
class StatusLEDSettings:
    enabled: bool = False
    white_pin: int = 15
    yellow_pin: int = 16
    blue_pin: int = 18
    green_pin: int = 22
    active_low: bool = False


@dataclass
class NullStatusLEDController:
    status_monitor: object | None = None

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def mark_session_started(self) -> None:
        return

    def set_tts_active(self, active: bool) -> None:
        return

    def set_stt_active(self, active: bool) -> None:
        return



@dataclass
class StatusLEDController:
    settings: StatusLEDSettings
    gpio_module: object | None = None
    status_monitor: object | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _started: bool = field(default=False, init=False)
    _session_started: bool = field(default=False, init=False)

    def start(self) -> None:
        if not self.settings.enabled:
            logger.info("Status LEDs disabled.")
            return
        with self._lock:
            if self._started:
                return
            try:
                gpio = self.gpio_module or self._load_gpio_module()
                self.gpio_module = gpio
                gpio.setmode(gpio.BOARD)
                gpio.setup(self._pins, gpio.OUT, initial=self._off_value)
                self._started = True
                self._write_color("white", True)
                self._write_color("yellow", False)
                self._write_color("blue", False)
                self._write_color("green", False)
                logger.info(
                    "Status LEDs started: white=BOARD %s yellow=BOARD %s blue=BOARD %s green=BOARD %s",
                    self.settings.white_pin,
                    self.settings.yellow_pin,
                    self.settings.blue_pin,
                    self.settings.green_pin,
                )
            except Exception as exc:
                self._started = False
                logger.warning("Status LEDs disabled: GPIO setup failed: %s", exc)

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            try:
                for color in ("white", "yellow", "blue", "green"):
                    self._write_color(color, False)
                self.gpio_module.cleanup(self._pins)
            except Exception as exc:
                logger.warning("Status LED cleanup failed: %s", exc)
            finally:
                self._started = False

    def mark_session_started(self) -> None:
        with self._lock:
            self._session_started = True
            self._write_color("yellow", True)

    def set_tts_active(self, active: bool) -> None:
        with self._lock:
            if active:
                self._write_color("green", False)
            self._write_color("blue", active)

    def set_stt_active(self, active: bool) -> None:
        with self._lock:
            if active:
                self._write_color("blue", False)
            self._write_color("green", active)

    @staticmethod
    def _load_gpio_module():
        try:
            import RPi.GPIO as GPIO
        except ImportError:
            import Jetson.GPIO as GPIO
        return GPIO

    @property
    def _pins(self) -> list[int]:
        return [
            self.settings.white_pin,
            self.settings.yellow_pin,
            self.settings.blue_pin,
            self.settings.green_pin,
        ]

    @property
    def _color_to_pin(self) -> dict[str, int]:
        return {
            "white": self.settings.white_pin,
            "yellow": self.settings.yellow_pin,
            "blue": self.settings.blue_pin,
            "green": self.settings.green_pin,
        }

    @property
    def _on_value(self) -> int:
        gpio = self.gpio_module
        return gpio.LOW if self.settings.active_low else gpio.HIGH

    @property
    def _off_value(self) -> int:
        gpio = self.gpio_module
        return gpio.HIGH if self.settings.active_low else gpio.LOW

    def _write_color(self, color: str, enabled: bool) -> None:
        if not self._started or self.gpio_module is None:
            return
        pin = self._color_to_pin[color]
        value = self._on_value if enabled else self._off_value
        try:
            self.gpio_module.output(pin, value)
            self._publish_light_state(color, enabled)
        except Exception as exc:
            logger.warning("Status LED write failed for %s: %s", color, exc)

    def _publish_light_state(self, color: str, enabled: bool) -> None:
        if self.status_monitor is None:
            return
        method = getattr(self.status_monitor, "set_light", None)
        if not callable(method):
            return
        try:
            method(color, enabled)
        except Exception as exc:
            logger.warning("Status monitor light update failed for %s: %s", color, exc)


def build_status_led_settings() -> StatusLEDSettings:
    return StatusLEDSettings(
        enabled=config_loader.HARDWARE_STATUS_LEDS_ENABLED,
        white_pin=config_loader.HARDWARE_STATUS_LED_WHITE_BOARD_PIN,
        yellow_pin=config_loader.HARDWARE_STATUS_LED_YELLOW_BOARD_PIN,
        blue_pin=config_loader.HARDWARE_STATUS_LED_BLUE_BOARD_PIN,
        green_pin=config_loader.HARDWARE_STATUS_LED_GREEN_BOARD_PIN,
        active_low=config_loader.HARDWARE_STATUS_LED_ACTIVE_LOW,
    )


def build_status_led_controller(status_monitor=None) -> StatusLEDController | NullStatusLEDController:
    settings = build_status_led_settings()
    if not settings.enabled:
        return NullStatusLEDController(status_monitor=status_monitor)
    return StatusLEDController(settings, status_monitor=status_monitor)
