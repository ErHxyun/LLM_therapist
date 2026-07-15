"""Probe likely ALSA playback routes for CaiTI TTS.

Run this on the Jetson while the CaiTI app service is stopped. The script
speaks a short phrase through several candidate `aplay` device routes so the
operator can note which route is actually audible.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import piper_tts_command as piper_tts  # noqa: E402
from src.utils.config_loader import VOICE_TTS_COMMAND  # noqa: E402


def command_option(command: str, option: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return ""
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(option + "="):
            return part.split("=", 1)[1]
    return ""


def command_float_option(command: str, option: str, default: float) -> float:
    value = command_option(command, option)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate


def default_players() -> list[str]:
    return [
        "aplay -q",
        "aplay -D default -q",
        "aplay -D plughw:3,0 -q",
        "aplay -D hw:3,0 -q",
        "aplay -D plughw:0,0 -q",
        "aplay -D hw:0,0 -q",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe CaiTI TTS playback devices.")
    parser.add_argument(
        "--player",
        action="append",
        dest="players",
        help="Exact player command to test, for example: --player 'aplay -D plughw:3,0 -q'",
    )
    parser.add_argument(
        "--text",
        default="This is a CaiTI playback test.",
        help="Base sentence to speak for each route.",
    )
    parser.add_argument(
        "--pause-sec",
        type=float,
        default=1.2,
        help="Pause between candidate routes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = command_option(VOICE_TTS_COMMAND, "--model")
    if not model_path:
        print("Could not find --model in voice.tts_command.", file=sys.stderr)
        return 1

    resolved_model = resolve_repo_path(model_path)
    if not resolved_model.exists():
        print(f"TTS model not found: {resolved_model}", file=sys.stderr)
        return 1

    length_scale = command_float_option(VOICE_TTS_COMMAND, "--length-scale", 0.8)
    sentence_silence = command_float_option(VOICE_TTS_COMMAND, "--sentence-silence", 0.25)
    players = args.players or default_players()

    print("Stop the CaiTI app service before running this probe:")
    print("  sudo systemctl stop caiti-app.service")
    print()
    print(f"Using model: {resolved_model}")
    print(f"Testing {len(players)} playback route(s). Listen for which one is audible.")
    print()

    for index, player in enumerate(players, start=1):
        print(f"[{index}/{len(players)}] Testing player: {player}")
        text = f"{args.text} Route {index}."
        try:
            piper_tts.speak_text(
                text=text,
                model_path=str(resolved_model),
                player=player,
                length_scale=length_scale,
                sentence_silence=sentence_silence,
                cache_dir=str(piper_tts.DEFAULT_CACHE_DIR),
            )
            print("  OK")
        except Exception as exc:
            print(f"  FAIL: {exc}")
        print()
        time.sleep(max(0.0, args.pause_sec))

    print("When you identify the audible route, set it in config.yaml for both main and intermission TTS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
