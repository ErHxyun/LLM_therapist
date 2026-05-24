"""Command-backend adapter for local Piper TTS.

Reads text from stdin, synthesizes one temporary WAV with Piper, then plays it
with a local audio player. Designed for CAITI_TTS_COMMAND.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

PLAYBACK_START_MARKER = "__CAITI_TTS_PLAYBACK_START__"
PLAYBACK_END_MARKER = "__CAITI_TTS_PLAYBACK_END__"


def build_piper_command(
    executable: str,
    model_path: str,
    output_file: str,
    length_scale: float,
    sentence_silence: float,
) -> list[str]:
    return [
        executable,
        "--model",
        model_path,
        "--length_scale",
        str(length_scale),
        "--sentence_silence",
        str(sentence_silence),
        "--output_file",
        output_file,
    ]


def build_player_command(player: str, wav_file: str) -> list[str]:
    if player == "aplay":
        return ["aplay", "-q", wav_file]
    if player == "paplay":
        return ["paplay", wav_file]
    return [player, wav_file]


def speak_text(
    text: str,
    model_path: str,
    executable: str = "piper",
    player: str = "aplay",
    length_scale: float = 0.8,
    sentence_silence: float = 0.4,
    timeout_sec: int = 60,
    emit_playback_markers: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    text = str(text or "").strip()
    if not text:
        return
    if not model_path:
        raise ValueError("Piper model path is required.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Piper model not found: {model_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_file = tmp.name

    try:
        piper_cmd = build_piper_command(
            executable,
            model_path,
            wav_file,
            length_scale,
            sentence_silence,
        )
        completed = runner(
            piper_cmd,
            input=text,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or "").strip() or "Piper TTS failed.")
        if not Path(wav_file).exists() or Path(wav_file).stat().st_size <= 44:
            raise RuntimeError("Piper produced an empty or invalid WAV file.")

        play_cmd = build_player_command(player, wav_file)
        if emit_playback_markers:
            print(PLAYBACK_START_MARKER, file=sys.stderr, flush=True)
        try:
            completed = runner(
                play_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        finally:
            if emit_playback_markers:
                print(PLAYBACK_END_MARKER, file=sys.stderr, flush=True)
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or "").strip() or "Audio playback failed.")
    finally:
        try:
            os.remove(wav_file)
        except OSError:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read stdin, synthesize with Piper, and play WAV locally.")
    parser.add_argument("--model", default=os.environ.get("CAITI_PIPER_MODEL", ""))
    parser.add_argument("--executable", default=os.environ.get("CAITI_PIPER_EXECUTABLE", "piper"))
    parser.add_argument("--player", default=os.environ.get("CAITI_AUDIO_PLAYER", "aplay"))
    parser.add_argument("--length-scale", type=float, default=float(os.environ.get("CAITI_PIPER_LENGTH_SCALE", "0.8")))
    parser.add_argument(
        "--sentence-silence",
        type=float,
        default=float(os.environ.get("CAITI_PIPER_SENTENCE_SILENCE", "0.4")),
    )
    parser.add_argument("--timeout-sec", type=int, default=int(os.environ.get("CAITI_TTS_TIMEOUT_SEC", "60")))
    parser.add_argument(
        "--playback-markers",
        action="store_true",
        default=os.environ.get("CAITI_TTS_PLAYBACK_MARKERS", "0").strip().lower() in {"1", "true", "yes", "on"},
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        speak_text(
            sys.stdin.read(),
            model_path=args.model,
            executable=args.executable,
            player=args.player,
            length_scale=args.length_scale,
            sentence_silence=args.sentence_silence,
            timeout_sec=args.timeout_sec,
            emit_playback_markers=args.playback_markers,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
