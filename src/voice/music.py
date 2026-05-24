"""Optional waiting-music backend for the voice shell.

Stage-one music is intentionally simple: play a local audio file only while
CaiTI is thinking between user response and next question. The voice loop stops
music before TTS and STT so the current aplay-based speech path stays stable.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from src.utils.config_loader import VOICE_MUSIC_BACKEND, VOICE_MUSIC_COMMAND, VOICE_MUSIC_PATH
from src.utils.log_util import get_logger

logger = get_logger("VoiceMusic")


class MusicBackend(Protocol):
    def start(self) -> None:
        """Start waiting music if available."""

    def stop(self) -> None:
        """Stop waiting music if it is playing."""


@dataclass
class NullMusic:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return


def format_music_command(command: str, path: str) -> str:
    quoted_path = shlex.quote(path)
    template = str(command or "").strip()
    if not template:
        template = "aplay -q {path}"
    if "{path}" in template:
        return template.replace("{path}", quoted_path)
    return f"{template} {quoted_path}"


@dataclass
class CommandMusic:
    path: str
    command: str = "aplay -q {path}"
    loop: bool = True
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen
    poll_interval_sec: float = 0.1
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _process: subprocess.Popen | None = field(default=None, init=False)

    def start(self) -> None:
        if not self.path:
            return
        if not os.path.exists(self.path):
            logger.warning("Waiting music file not found: %s", self.path)
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="caiti-waiting-music", daemon=True)
            self._thread.start()
            logger.info("Waiting music started: %s", self.path)

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            process = self._process
            self._stop_event.set()
        self._terminate_process(process)
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            if self._thread is thread:
                self._thread = None
            if self._process is process:
                self._process = None

    def _run(self) -> None:
        command = format_music_command(self.command, self.path)
        while not self._stop_event.is_set():
            process = None
            try:
                process = self.popen_factory(
                    command,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                with self._lock:
                    self._process = process
                while process.poll() is None and not self._stop_event.wait(self.poll_interval_sec):
                    pass
            except Exception as exc:
                logger.warning("Waiting music playback failed: %s", exc)
                break
            finally:
                self._terminate_process(process)
                with self._lock:
                    if self._process is process:
                        self._process = None
            if not self.loop:
                break
        logger.info("Waiting music stopped.")

    @staticmethod
    def _terminate_process(process) -> None:
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
        except Exception:
            pass


def build_music() -> MusicBackend:
    backend = VOICE_MUSIC_BACKEND.strip().lower()
    if backend in {"", "off", "none", "disabled"}:
        return NullMusic()
    if backend == "command":
        return CommandMusic(VOICE_MUSIC_PATH, VOICE_MUSIC_COMMAND)
    raise ValueError(f"Unsupported music backend: {VOICE_MUSIC_BACKEND}")
