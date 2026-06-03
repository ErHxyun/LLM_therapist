"""Optional music backends for the voice shell.

The command backend is a simple waiting-music player. The mpv backend keeps
background music alive and uses mpv IPC to lower volume during TTS/STT.
"""

from __future__ import annotations

import os
import json
import signal
import shlex
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from src.utils.config_loader import (
    VOICE_MUSIC_BACKEND,
    VOICE_MUSIC_COMMAND,
    VOICE_MUSIC_DUCK_VOLUME_PERCENT,
    VOICE_MUSIC_FIREPLACE_PATH,
    VOICE_MUSIC_IPC_PATH,
    VOICE_MUSIC_PATH,
    VOICE_MUSIC_SEAWAVES_PATH,
    VOICE_MUSIC_VOLUME_PERCENT,
)
from src.utils.log_util import get_logger

logger = get_logger("VoiceMusic")


class MusicBackend(Protocol):
    def start(self) -> None:
        """Start waiting music if available."""

    def stop(self) -> None:
        """Stop waiting music if it is playing."""

    def is_playing(self) -> bool:
        """Return whether waiting music is currently active."""

    def is_background(self) -> bool:
        """Return whether music should continue through voice activity."""

    def duck(self) -> None:
        """Lower music volume for TTS or STT."""

    def restore_volume(self) -> None:
        """Restore normal music volume."""

    def pause(self) -> None:
        """Pause music without losing playback state."""

    def resume(self) -> None:
        """Resume music after pause."""

    def cycle_mode(self) -> str:
        """Switch to the next configured music mode."""


@dataclass(frozen=True)
class MusicMode:
    name: str
    path: str = ""


def build_music_modes(
    music_path: str,
    fireplace_path: str = "",
    seawaves_path: str = "",
) -> list[MusicMode]:
    return [
        MusicMode("music", music_path),
        MusicMode("fireplace", fireplace_path),
        MusicMode("seawaves", seawaves_path),
        MusicMode("off", ""),
    ]


def _normalize_music_modes(path: str, modes: list[MusicMode] | None) -> list[MusicMode]:
    if modes:
        normalized = list(modes)
    else:
        normalized = [MusicMode("music", path), MusicMode("off", "")]
    if not normalized:
        normalized = [MusicMode("off", "")]
    return normalized

@dataclass
class NullMusic:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def is_playing(self) -> bool:
        return False

    def is_background(self) -> bool:
        return False

    def duck(self) -> None:
        return

    def restore_volume(self) -> None:
        return

    def pause(self) -> None:
        return

    def resume(self) -> None:
        return

    def cycle_mode(self) -> str:
        return "off"


def format_music_command(command: str, path: str) -> str:
    quoted_path = shlex.quote(path)
    template = str(command or "").strip()
    if not template:
        template = "aplay -q {path}"
    if "{path}" in template:
        return template.replace("{path}", quoted_path)
    return f"{template} {quoted_path}"


def build_music_command_args(command: str, path: str) -> list[str]:
    return shlex.split(format_music_command(command, path))


@dataclass
class CommandMusic:
    path: str
    command: str = "aplay -q {path}"
    loop: bool = True
    modes: list[MusicMode] | None = None
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen
    poll_interval_sec: float = 0.1
    _stop_event: threading.Event | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _process: subprocess.Popen | None = field(default=None, init=False)
    _mode_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.modes = _normalize_music_modes(self.path, self.modes)
        self.path = self.modes[self._mode_index].path

    def start(self) -> None:
        if not self.path:
            return
        if not os.path.exists(self.path):
            logger.warning("Waiting music file not found: %s", self.path)
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._thread = threading.Thread(
                target=self._run,
                args=(stop_event,),
                name="caiti-waiting-music",
                daemon=True,
            )
            self._thread.start()
            logger.info("Waiting music started: %s", self.path)

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            process = self._process
            stop_event = self._stop_event
            if stop_event is not None:
                stop_event.set()
        self._terminate_process(process)
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None
                self._stop_event = None
            if self._process is process:
                self._process = None

    def is_playing(self) -> bool:
        with self._lock:
            thread = self._thread
            stop_event = self._stop_event
            return bool(
                thread is not None
                and thread.is_alive()
                and stop_event is not None
                and not stop_event.is_set()
            )

    def is_background(self) -> bool:
        return False

    def duck(self) -> None:
        self.stop()

    def restore_volume(self) -> None:
        return

    def pause(self) -> None:
        self.stop()

    def resume(self) -> None:
        self.start()

    def cycle_mode(self) -> str:
        was_playing = self.is_playing()
        was_off = not self.path
        if was_playing:
            self.stop()
        with self._lock:
            self._mode_index = (self._mode_index + 1) % len(self.modes)
            mode = self.modes[self._mode_index]
            self.path = mode.path
        logger.info("Waiting music mode changed: %s", mode.name)
        if mode.path and (was_playing or was_off):
            self.start()
        return mode.name

    def _run(self, stop_event: threading.Event) -> None:
        command_args = build_music_command_args(self.command, self.path)
        while not stop_event.is_set():
            process = None
            try:
                process = self.popen_factory(
                    command_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    text=True,
                )
                with self._lock:
                    self._process = process
                while process.poll() is None and not stop_event.wait(self.poll_interval_sec):
                    pass
                returncode = process.poll()
                if (
                    returncode not in (None, 0)
                    and not stop_event.is_set()
                ):
                    stderr = ""
                    if process.stderr is not None:
                        stderr = process.stderr.read().strip()
                    logger.warning(
                        "Waiting music player exited with code %s: %s",
                        returncode,
                        stderr or "no stderr",
                    )
                    break
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
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except Exception:
                    process.terminate()
                try:
                    process.wait(timeout=0.15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except Exception:
                        process.kill()
                    try:
                        process.wait(timeout=0.15)
                    except subprocess.TimeoutExpired:
                        pass
        except Exception:
            pass


def build_mpv_command_args(path: str, ipc_path: str, volume_percent: int) -> list[str]:
    return [
        "mpv",
        "--no-video",
        "--quiet",
        "--force-window=no",
        "--loop-file=inf",
        f"--input-ipc-server={ipc_path}",
        f"--volume={int(volume_percent)}",
        path,
    ]


@dataclass
class MPVBackgroundMusic:
    path: str
    ipc_path: str = "/tmp/caiti_mpv_music.sock"
    volume_percent: int = 30
    duck_volume_percent: int = 8
    modes: list[MusicMode] | None = None
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen
    ipc_sender: Callable[[str, list], None] | None = None
    ipc_ready_timeout_sec: float = 1.5
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _process: subprocess.Popen | None = field(default=None, init=False)
    _paused: bool = field(default=False, init=False)
    _ducked: bool = field(default=False, init=False)
    _mode_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.modes = _normalize_music_modes(self.path, self.modes)
        self.path = self.modes[self._mode_index].path

    def start(self) -> None:
        if not self.path:
            return
        if not os.path.exists(self.path):
            logger.warning("Background music file not found: %s", self.path)
            return
        with self._lock:
            if self._process and self._process.poll() is None:
                self._send_command_locked(["set_property", "pause", False])
                self._send_volume_locked(self.duck_volume_percent if self._ducked else self.volume_percent)
                self._paused = False
                return
            if shutil.which("mpv") is None and self.popen_factory is subprocess.Popen:
                logger.warning("mpv is not installed; background music is disabled.")
                return
            self._remove_stale_socket()
            initial_volume = self.duck_volume_percent if self._ducked else self.volume_percent
            command_args = build_mpv_command_args(self.path, self.ipc_path, initial_volume)
            try:
                self._process = self.popen_factory(
                    command_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    text=True,
                )
            except Exception as exc:
                logger.warning("Background music failed to start: %s", exc)
                self._process = None
                return
            self._paused = False
        self._wait_for_ipc_ready()
        logger.info("Background music started with mpv: %s", self.path)

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                self._force_stop_matching_mpv_processes()
                self._remove_stale_socket()
                return
            self._send_command_locked(["quit"])
        self._terminate_process(process)
        self._force_stop_matching_mpv_processes()
        with self._lock:
            if self._process is process:
                self._process = None
            self._paused = False
            self._ducked = False
            self._remove_stale_socket()

    def is_playing(self) -> bool:
        with self._lock:
            return bool(self._process is not None and self._process.poll() is None and not self._paused)

    def is_background(self) -> bool:
        return True

    def duck(self) -> None:
        self.start()
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            self._ducked = True
            self._send_volume_locked(self.duck_volume_percent)

    def restore_volume(self) -> None:
        with self._lock:
            self._ducked = False
            if self._process is None or self._process.poll() is not None:
                return
            self._send_volume_locked(self.volume_percent)

    def pause(self) -> None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            self._send_command_locked(["set_property", "pause", True])
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                should_start = True
            else:
                should_start = False
        if should_start:
            self.start()
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            self._send_command_locked(["set_property", "pause", False])
            self._send_volume_locked(self.duck_volume_percent if self._ducked else self.volume_percent)
            self._paused = False

    def cycle_mode(self) -> str:
        with self._lock:
            process = self._process
            was_running = bool(process is not None and process.poll() is None)
            was_paused = self._paused
            was_ducked = self._ducked
            was_off = not self.path
            self._mode_index = (self._mode_index + 1) % len(self.modes)
            mode = self.modes[self._mode_index]
            self.path = mode.path
            if was_running and mode.path:
                switched = self._send_command_locked(["loadfile", mode.path, "replace"])
                if switched:
                    self._ducked = was_ducked
                    self._send_volume_locked(self.duck_volume_percent if was_ducked else self.volume_percent)
                    self._send_command_locked(["set_property", "pause", was_paused])
                    self._paused = was_paused
                    logger.info("Background music mode changed: %s", mode.name)
                    return mode.name

        if was_running:
            self.stop()
            with self._lock:
                self._ducked = was_ducked

        logger.info("Background music mode changed: %s", mode.name)
        if mode.path and (was_running or was_off):
            self.start()
            if was_paused:
                self.pause()
        return mode.name

    def _send_volume_locked(self, volume_percent: int) -> None:
        self._send_command_locked(["set_property", "volume", int(volume_percent)])

    def _send_command_locked(self, command: list) -> bool:
        try:
            if self.ipc_sender is not None:
                self.ipc_sender(self.ipc_path, command)
                return True
            _send_mpv_ipc_command(self.ipc_path, command)
            return True
        except Exception as exc:
            logger.debug("mpv IPC command failed: %s", exc)
            return False

    def _wait_for_ipc_ready(self) -> None:
        if self.ipc_sender is not None:
            return
        deadline = time.monotonic() + self.ipc_ready_timeout_sec
        while time.monotonic() < deadline:
            if os.path.exists(self.ipc_path):
                return
            time.sleep(0.02)

    def _remove_stale_socket(self) -> None:
        try:
            if os.path.exists(self.ipc_path):
                os.unlink(self.ipc_path)
        except OSError:
            pass

    def _force_stop_matching_mpv_processes(self) -> None:
        ipc_arg = f"--input-ipc-server={self.ipc_path}"
        pids = self._find_matching_mpv_pids(ipc_arg)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            still_running = []
            for pid in pids:
                try:
                    os.kill(pid, sig)
                    still_running.append(pid)
                except ProcessLookupError:
                    continue
                except Exception as exc:
                    logger.debug("Failed to signal mpv pid %s: %s", pid, exc)
            if not still_running:
                return
            time.sleep(0.1)
            pids = [pid for pid in still_running if self._pid_is_running(pid)]

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return True

    @staticmethod
    def _find_matching_mpv_pids(ipc_arg: str) -> list[int]:
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,comm=,args="],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.debug("Unable to scan for orphan mpv processes: %s", exc)
            return []
        matches: list[int] = []
        current_pid = os.getpid()
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            command = parts[1]
            args = parts[2]
            if pid == current_pid:
                continue
            if command == "mpv" and ipc_arg in args:
                matches.append(pid)
        return matches

    @staticmethod
    def _terminate_process(process) -> None:
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except Exception:
                    process.terminate()
                try:
                    process.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except Exception:
                        process.kill()
                    try:
                        process.wait(timeout=0.3)
                    except subprocess.TimeoutExpired:
                        pass
        except Exception:
            pass


def _send_mpv_ipc_command(ipc_path: str, command: list) -> None:
    payload = json.dumps({"command": command}).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(0.2)
        client.connect(ipc_path)
        client.sendall(payload)
        try:
            client.recv(4096)
        except socket.timeout:
            pass


def build_music() -> MusicBackend:
    backend = VOICE_MUSIC_BACKEND.strip().lower()
    if backend in {"", "off", "none", "disabled"}:
        return NullMusic()
    if backend == "command":
        return CommandMusic(
            VOICE_MUSIC_PATH,
            VOICE_MUSIC_COMMAND,
            modes=build_music_modes(
                VOICE_MUSIC_PATH,
                fireplace_path=VOICE_MUSIC_FIREPLACE_PATH,
                seawaves_path=VOICE_MUSIC_SEAWAVES_PATH,
            ),
        )
    if backend == "mpv":
        return MPVBackgroundMusic(
            VOICE_MUSIC_PATH,
            ipc_path=VOICE_MUSIC_IPC_PATH,
            volume_percent=VOICE_MUSIC_VOLUME_PERCENT,
            duck_volume_percent=VOICE_MUSIC_DUCK_VOLUME_PERCENT,
            modes=build_music_modes(
                VOICE_MUSIC_PATH,
                fireplace_path=VOICE_MUSIC_FIREPLACE_PATH,
                seawaves_path=VOICE_MUSIC_SEAWAVES_PATH,
            ),
        )
    raise ValueError(f"Unsupported music backend: {VOICE_MUSIC_BACKEND}")
