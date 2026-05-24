"""GPIO volume-button controller for the Jetson voice shell.

The expected button wiring is active-low:

    3.3V -- 10k resistor -- BOARD pin -- button -- GND

With that circuit, idle reads as 1 and pressed reads as 0. Jetson.GPIO ignores
software pull-up/pull-down requests on this platform, so the external resistor
is part of the runtime contract.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("VolumeButtons")


CommandRunner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class VolumeButtonSettings:
    enabled: bool = False
    up_pin: int = 32
    down_pin: int = 33
    step_percent: int = 5
    min_percent: int = 0
    max_percent: int = 100
    debounce_sec: float = 0.5
    release_sec: float = 0.2
    poll_interval_sec: float = 0.01
    active_low: bool = True
    pactl_command: str = "pactl"


@dataclass
class _ButtonState:
    armed: bool
    last_press_at: float = -1_000_000.0
    release_started_at: float | None = None


@dataclass
class NullVolumeButtonController:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return


@dataclass
class VolumeButtonController:
    settings: VolumeButtonSettings
    gpio_module: object | None = None
    command_runner: CommandRunner = subprocess.run
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _button_states: dict[int, _ButtonState] = field(default_factory=dict, init=False)
    _pin_to_action: dict[int, str] = field(default_factory=dict, init=False)

    def start(self) -> None:
        if not self.settings.enabled:
            logger.info("Volume buttons disabled.")
            return
        if self._thread and self._thread.is_alive():
            return
        if shutil.which(self.settings.pactl_command) is None:
            logger.warning("Volume buttons disabled: %s not found.", self.settings.pactl_command)
            return

        try:
            gpio = self.gpio_module or self._load_gpio_module()
            self.gpio_module = gpio
            self._setup_gpio(gpio)
        except Exception as exc:
            logger.warning("Volume buttons disabled: GPIO setup failed: %s", exc)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="caiti-volume-buttons", daemon=True)
        self._thread.start()
        logger.info(
            "Volume buttons started: up=BOARD %s down=BOARD %s step=%s%% range=%s-%s%%",
            self.settings.up_pin,
            self.settings.down_pin,
            self.settings.step_percent,
            self.settings.min_percent,
            self.settings.max_percent,
        )

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

    def _setup_gpio(self, gpio) -> None:
        pins = [self.settings.up_pin, self.settings.down_pin]
        self._pin_to_action = {
            self.settings.up_pin: "up",
            self.settings.down_pin: "down",
        }
        gpio.setmode(gpio.BOARD)
        for pin in pins:
            gpio.setup(pin, gpio.IN)
        pressed_value = self._pressed_value
        self._button_states = {
            pin: _ButtonState(armed=(gpio.input(pin) != pressed_value))
            for pin in pins
        }

    def _cleanup_gpio(self) -> None:
        gpio = self.gpio_module
        if gpio is None:
            return
        try:
            gpio.cleanup([self.settings.up_pin, self.settings.down_pin])
        except Exception:
            try:
                gpio.cleanup()
            except Exception:
                pass

    @property
    def _pressed_value(self) -> int:
        return 0 if self.settings.active_low else 1

    def _run(self) -> None:
        while not self._stop_event.wait(self.settings.poll_interval_sec):
            try:
                self._poll_once(time.monotonic())
            except Exception as exc:
                logger.warning("Volume button polling failed: %s", exc)
                time.sleep(0.2)

    def _poll_once(self, now: float) -> None:
        gpio = self.gpio_module
        if gpio is None:
            return
        for pin in self._pin_to_action:
            action = self._process_pin_value(pin, gpio.input(pin), now)
            if action:
                self._change_volume(action)

    def _process_pin_value(self, pin: int, value: int, now: float) -> str | None:
        state = self._button_states[pin]
        pressed = value == self._pressed_value
        if pressed:
            state.release_started_at = None
            if not state.armed:
                return None
            if now - state.last_press_at < self.settings.debounce_sec:
                return None
            state.armed = False
            state.last_press_at = now
            return self._pin_to_action.get(pin)

        if state.release_started_at is None:
            state.release_started_at = now
        if not state.armed and now - state.release_started_at >= self.settings.release_sec:
            state.armed = True
        return None

    def _change_volume(self, action: str) -> None:
        current = self._get_current_volume_percent()
        if current is None:
            logger.warning("Skipping volume_%s: current volume could not be read.", action)
            return

        if action == "up":
            target = current + self.settings.step_percent
        elif action == "down":
            target = current - self.settings.step_percent
        else:
            return

        target = max(self.settings.min_percent, min(self.settings.max_percent, target))
        if target == current:
            logger.info("Volume_%s ignored at limit: %s%%", action, current)
            return

        cmd = [self.settings.pactl_command, "set-sink-volume", "@DEFAULT_SINK@", f"{target}%"]
        logger.info("Volume_%s: %s%% -> %s%%", action, current, target)
        self.command_runner(cmd, check=False)

    def _get_current_volume_percent(self) -> int | None:
        cmd = [self.settings.pactl_command, "get-sink-volume", "@DEFAULT_SINK@"]
        completed = self.command_runner(cmd, check=False, capture_output=True, text=True)
        if getattr(completed, "returncode", 0) != 0:
            return None
        stdout = str(getattr(completed, "stdout", ""))
        matches = re.findall(r"/\s*(\d+)%", stdout)
        if not matches:
            return None
        return int(matches[0])


def build_volume_button_settings() -> VolumeButtonSettings:
    return VolumeButtonSettings(
        enabled=config_loader.HARDWARE_VOLUME_BUTTONS_ENABLED,
        up_pin=config_loader.HARDWARE_VOLUME_UP_BOARD_PIN,
        down_pin=config_loader.HARDWARE_VOLUME_DOWN_BOARD_PIN,
        step_percent=config_loader.HARDWARE_VOLUME_STEP_PERCENT,
        min_percent=config_loader.HARDWARE_VOLUME_MIN_PERCENT,
        max_percent=config_loader.HARDWARE_VOLUME_MAX_PERCENT,
        debounce_sec=config_loader.HARDWARE_VOLUME_DEBOUNCE_SEC,
        release_sec=config_loader.HARDWARE_VOLUME_RELEASE_SEC,
        poll_interval_sec=config_loader.HARDWARE_VOLUME_POLL_INTERVAL_SEC,
        active_low=config_loader.HARDWARE_VOLUME_ACTIVE_LOW,
    )


def build_volume_button_controller() -> VolumeButtonController | NullVolumeButtonController:
    settings = build_volume_button_settings()
    if not settings.enabled:
        return NullVolumeButtonController()
    return VolumeButtonController(settings)
