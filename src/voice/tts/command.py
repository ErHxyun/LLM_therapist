from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

from src.utils.log_util import get_logger
from src.voice.exceptions import VoiceInterrupted

logger = get_logger("VoiceTTS")

PLAYBACK_START_MARKER = "__CAITI_TTS_PLAYBACK_START__"
PLAYBACK_END_MARKER = "__CAITI_TTS_PLAYBACK_END__"


@dataclass
class ConsoleTTS:
    prefix: str = "AI"

    def speak(self, text: str) -> None:
        chunk = str(text or "").strip()
        if chunk:
            print(f"[{self.prefix}] {chunk}", flush=True)

    def speak_stream(self, text: str) -> None:
        self.speak(text)


@dataclass
class CommandTTS:
    command: str
    timeout_sec: int = 60
    runner: Callable = subprocess.run
    _playback_status_hook: Callable[[bool], None] | None = field(default=None, init=False, repr=False)
    _interrupt_check: Callable[[], bool] | None = field(default=None, init=False, repr=False)
    last_timing: dict[str, float | int | bool] = field(default_factory=dict, init=False)

    def set_playback_status_hook(self, hook: Callable[[bool], None] | None) -> None:
        self._playback_status_hook = hook

    def set_interrupt_check(self, checker: Callable[[], bool] | None) -> None:
        self._interrupt_check = checker

    def _should_interrupt(self) -> bool:
        return bool(self._interrupt_check is not None and self._interrupt_check())

    def speak(self, text: str) -> None:
        chunk = str(text or "").strip()
        if not chunk:
            return
        if not self.command.strip():
            raise ValueError("CommandTTS requires a non-empty command.")
        if self._should_interrupt():
            raise VoiceInterrupted("TTS interrupted before playback.")
        if self._playback_status_hook is not None:
            self._speak_with_playback_markers(chunk, self._playback_status_hook)
            self._log_timing(len(chunk), used_playback_markers=True)
            logger.info("TTS command spoke block length=%s", len(chunk))
            return
        started_at = time.monotonic()
        completed = self.runner(
            self.command,
            input=chunk,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        total_duration_sec = time.monotonic() - started_at
        self.last_timing = {
            "total_duration_sec": round(total_duration_sec, 3),
            "prepare_duration_sec": 0.0,
            "playback_duration_sec": round(total_duration_sec, 3),
            "text_length": len(chunk),
            "used_playback_markers": False,
        }
        if self._should_interrupt():
            raise VoiceInterrupted("TTS interrupted after playback.")
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(f"TTS command failed with code {completed.returncode}: {stderr}")
        self._log_timing(len(chunk), used_playback_markers=False)
        logger.info("TTS command spoke block length=%s", len(chunk))

    def _speak_with_playback_markers(self, chunk: str, hook: Callable[[bool], None]) -> None:
        env = os.environ.copy()
        env["CAITI_TTS_PLAYBACK_MARKERS"] = "1"
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            env=env,
            bufsize=1,
            start_new_session=True,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        playback_active = False
        started_at = time.monotonic()
        playback_started_at: float | None = None
        playback_ended_at: float | None = None

        def set_playback_active(active: bool) -> None:
            nonlocal playback_active, playback_started_at, playback_ended_at
            if playback_active == active:
                return
            playback_active = active
            now = time.monotonic()
            if active:
                playback_started_at = now
            else:
                playback_ended_at = now
            hook(active)

        selector = selectors.DefaultSelector()
        try:
            assert process.stdin is not None
            process.stdin.write(chunk)
            process.stdin.close()
            if process.stdout is not None:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            if process.stderr is not None:
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")

            deadline = time.monotonic() + self.timeout_sec
            while selector.get_map():
                if self._should_interrupt():
                    self._terminate_process_tree(process)
                    raise VoiceInterrupted("TTS interrupted during playback.")
                if time.monotonic() > deadline:
                    self._terminate_process_tree(process, force=True)
                    raise subprocess.TimeoutExpired(self.command, self.timeout_sec)
                events = selector.select(timeout=0.1)
                if not events and process.poll() is not None:
                    continue
                for key, _mask in events:
                    line = key.fileobj.readline()
                    if line == "":
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stdout":
                        stdout_chunks.append(line)
                        continue
                    marker = line.strip()
                    if marker == PLAYBACK_START_MARKER:
                        set_playback_active(True)
                    elif marker == PLAYBACK_END_MARKER:
                        set_playback_active(False)
                    else:
                        stderr_chunks.append(line)

            returncode = process.wait(timeout=0.1)
        finally:
            selector.close()
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            if playback_active:
                set_playback_active(False)

        if returncode != 0:
            stderr = "".join(stderr_chunks).strip()
            raise RuntimeError(f"TTS command failed with code {returncode}: {stderr}")
        finished_at = time.monotonic()
        prepare_duration = (
            max(0.0, playback_started_at - started_at) if playback_started_at is not None else finished_at - started_at
        )
        playback_duration = (
            max(0.0, playback_ended_at - playback_started_at)
            if playback_started_at is not None and playback_ended_at is not None
            else 0.0
        )
        self.last_timing = {
            "total_duration_sec": round(finished_at - started_at, 3),
            "prepare_duration_sec": round(prepare_duration, 3),
            "playback_duration_sec": round(playback_duration, 3),
            "text_length": len(chunk),
            "used_playback_markers": True,
        }

    def _log_timing(self, text_length: int, used_playback_markers: bool) -> None:
        timing = self.last_timing or {}
        logger.info(
            "TTS timing total=%.3fs prepare=%.3fs playback=%.3fs text_length=%s markers=%s",
            float(timing.get("total_duration_sec", 0.0)),
            float(timing.get("prepare_duration_sec", 0.0)),
            float(timing.get("playback_duration_sec", 0.0)),
            text_length,
            used_playback_markers,
        )

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen, force: bool = False) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except Exception:
            try:
                process.kill() if force else process.terminate()
            except Exception:
                pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            process.wait(timeout=1)

    def speak_stream(self, text: str) -> None:
        self.speak(text)
