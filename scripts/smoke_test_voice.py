"""Smoke test local CaiTI voice commands and hardware.

This script uses the voice backend configuration from config.yaml. By default it
plays a short Piper sentence, records one Faster-Whisper utterance, and prints
the transcript. Use --dry-run to validate configuration without touching audio.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import (  # noqa: E402
    VOICE_MUSIC_BACKEND,
    VOICE_MUSIC_COMMAND,
    VOICE_MUSIC_PATH,
    VOICE_STT_AUDIO_DEVICE,
    VOICE_STT_BACKEND,
    VOICE_STT_COMMAND,
    VOICE_STT_TIMEOUT_SEC,
    VOICE_TTS_BACKEND,
    VOICE_TTS_COMMAND,
    VOICE_TTS_TIMEOUT_SEC,
)
from src.voice.backends import build_stt  # noqa: E402
from src.voice.music import format_music_command  # noqa: E402


def command_option(command: str, option: str) -> str:
    parts = shlex.split(command)
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(option + "="):
            return part.split("=", 1)[1]
    return ""


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _shell_first_token(command: str) -> str:
    parts = shlex.split(command)
    if not parts:
        return ""
    if parts[0] == "python" and len(parts) > 1:
        return parts[1]
    return parts[0]


def check_configuration() -> list[str]:
    failures: list[str] = []
    stt_backend = VOICE_STT_BACKEND.strip().lower()
    tts_backend = VOICE_TTS_BACKEND.strip().lower()
    music_backend = VOICE_MUSIC_BACKEND.strip().lower()

    if stt_backend not in {"command", "faster_whisper", "faster-whisper", "local_whisper", "local-whisper"}:
        failures.append(
            f"voice.stt_backend is {VOICE_STT_BACKEND!r}; expected 'command' or 'faster_whisper'."
        )
    if tts_backend != "command":
        failures.append(f"voice.tts_backend is {VOICE_TTS_BACKEND!r}; expected 'command'.")
    if stt_backend == "command" and not VOICE_STT_COMMAND.strip():
        failures.append("voice.stt_command is empty.")
    if not VOICE_TTS_COMMAND.strip():
        failures.append("voice.tts_command is empty.")

    if stt_backend in {"command", "faster_whisper", "faster-whisper", "local_whisper", "local-whisper"} and not _has_module("faster_whisper"):
        failures.append("Python package faster-whisper is not importable.")

    tts_script = _shell_first_token(VOICE_TTS_COMMAND)
    stt_script = _shell_first_token(VOICE_STT_COMMAND) if stt_backend == "command" else ""
    for script in (tts_script, stt_script):
        if script.endswith(".py") and not resolve_repo_path(script).exists():
            failures.append(f"Configured script does not exist: {script}")

    tts_model = command_option(VOICE_TTS_COMMAND, "--model")
    if tts_model and not resolve_repo_path(tts_model).exists():
        failures.append(f"Piper model does not exist: {tts_model}")

    audio_device = command_option(VOICE_STT_COMMAND, "--audio-device") if stt_backend == "command" else VOICE_STT_AUDIO_DEVICE
    if audio_device and shutil.which("arecord") is None:
        failures.append("arecord is not available, but STT needs microphone recording.")

    player = command_option(VOICE_TTS_COMMAND, "--player")
    if player and shutil.which(player) is None:
        failures.append(f"Audio player is not available: {player}")

    piper_executable = command_option(VOICE_TTS_COMMAND, "--executable") or "piper"
    if VOICE_TTS_COMMAND and "piper_tts_command.py" in VOICE_TTS_COMMAND and shutil.which(piper_executable) is None:
        failures.append(f"Piper executable is not available: {piper_executable}")

    if music_backend not in {"", "off", "none", "disabled"}:
        if music_backend != "command":
            failures.append(f"voice.music_backend is {VOICE_MUSIC_BACKEND!r}; expected 'off' or 'command'.")
        if not VOICE_MUSIC_PATH.strip():
            failures.append("voice.music_path is empty.")
        elif not resolve_repo_path(VOICE_MUSIC_PATH).exists():
            failures.append(f"Waiting music file does not exist: {VOICE_MUSIC_PATH}")
        music_command = format_music_command(VOICE_MUSIC_COMMAND, VOICE_MUSIC_PATH)
        music_player = shlex.split(music_command)[0] if music_command.strip() else ""
        if music_player and shutil.which(music_player) is None:
            failures.append(f"Waiting music player is not available: {music_player}")

    return failures


def run_command(command: str, timeout_sec: int, stdin_text: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=stdin_text,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def run_tts_smoke(text: str) -> bool:
    print("\n[TTS] Playing a short Piper test sentence...")
    completed = run_command(VOICE_TTS_COMMAND, VOICE_TTS_TIMEOUT_SEC, text)
    if completed.returncode == 0:
        print("[TTS] OK")
        return True
    print("[TTS] FAIL")
    print((completed.stderr or completed.stdout or "").strip())
    return False


def run_stt_smoke() -> bool:
    if VOICE_STT_BACKEND.strip().lower() == "command":
        print("\n[STT] Recording now. Speak one short sentence after this line appears.")
        completed = run_command(VOICE_STT_COMMAND, VOICE_STT_TIMEOUT_SEC)
        transcript = (completed.stdout or "").strip()
        if completed.returncode == 0 and transcript:
            print("[STT] OK")
            print(f"Transcript: {transcript}")
            return True
        print("[STT] FAIL")
        if transcript:
            print(f"Transcript: {transcript}")
        print((completed.stderr or "").strip() or "No transcript returned.")
        return False

    try:
        stt = build_stt()
        warm_up = getattr(stt, "warm_up", None)
        if callable(warm_up):
            print("\n[STT] Loading persistent Faster-Whisper model...")
            warm_up()
        print("[STT] Recording now. Speak one short sentence after this line appears.")
        transcript = stt.listen().strip()
    except Exception as exc:
        print("[STT] FAIL")
        print(str(exc))
        return False
    if transcript:
        print("[STT] OK")
        print(f"Transcript: {transcript}")
        return True
    print("[STT] FAIL")
    print("No transcript returned.")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test CaiTI's local voice command backends.")
    parser.add_argument("--dry-run", action="store_true", help="Check config/dependencies without recording or playback.")
    parser.add_argument("--skip-stt", action="store_true", help="Skip the microphone/Faster-Whisper check.")
    parser.add_argument("--skip-tts", action="store_true", help="Skip the Piper playback check.")
    parser.add_argument(
        "--tts-text",
        default="This is a CaiTI local voice smoke test.",
        help="Sentence to speak during the TTS check.",
    )
    args = parser.parse_args(argv)

    print("CaiTI voice smoke test")
    print(f"STT backend: {VOICE_STT_BACKEND}")
    print(f"STT command: {VOICE_STT_COMMAND}")
    print(f"TTS backend: {VOICE_TTS_BACKEND}")
    print(f"TTS command: {VOICE_TTS_COMMAND}")
    print(f"Music backend: {VOICE_MUSIC_BACKEND}")
    print(f"Music path: {VOICE_MUSIC_PATH}")

    failures = check_configuration()
    if failures:
        print("\nConfiguration check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 2

    print("\nConfiguration check OK.")
    if args.dry_run:
        return 0

    ok = True
    if not args.skip_tts:
        ok = run_tts_smoke(args.tts_text) and ok
    if not args.skip_stt:
        ok = run_stt_smoke() and ok

    if ok:
        print("\nVoice smoke test passed.")
        return 0
    print("\nVoice smoke test failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
