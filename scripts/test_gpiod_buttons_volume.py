"""Test GPIO buttons and control system volume through libgpiod CLI tools.

This bypasses Jetson.GPIO. It uses gpiomon for edge events and pactl for
PulseAudio volume control.

Default button wiring:
  GPIO line -> button -> GND

With that wiring, keep the default pull-up bias and falling-edge press logic.

Default Jetson Orin Nano/NX 40-pin header mapping used here:
  volume_up   BOARD 32 -> gpiochip0 line 41
  volume_down BOARD 33 -> gpiochip0 line 43
  mute        BOARD 24 -> gpiochip0 line 136
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time


BOARD_TO_LINE = {
    7: 144,
    11: 112,
    12: 50,
    13: 122,
    15: 85,
    16: 126,
    18: 125,
    19: 135,
    21: 134,
    22: 123,
    23: 133,
    24: 136,
    26: 137,
    29: 105,
    31: 106,
    32: 41,
    33: 43,
    35: 53,
    36: 113,
    37: 124,
    38: 52,
    40: 51,
}

DEFAULT_BUTTON_BOARDS = {
    "volume_up": 32,
    "volume_down": 33,
    "mute": 24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch GPIO buttons and adjust output volume.")
    parser.add_argument("--chip", default="gpiochip0")
    parser.add_argument("--volume-up-board", type=int, default=DEFAULT_BUTTON_BOARDS["volume_up"])
    parser.add_argument("--volume-down-board", type=int, default=DEFAULT_BUTTON_BOARDS["volume_down"])
    parser.add_argument("--mute-board", type=int, default=DEFAULT_BUTTON_BOARDS["mute"])
    parser.add_argument("--volume-up-line", type=int)
    parser.add_argument("--volume-down-line", type=int)
    parser.add_argument("--mute-line", type=int)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--step", default="5%", help="Volume step passed to pactl, e.g. 5%.")
    parser.add_argument("--debounce-sec", type=float, default=0.2)
    parser.add_argument("--active-high", action="store_true", help="Treat rising edges as presses.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def board_to_line(board_pin: int) -> int:
    try:
        return BOARD_TO_LINE[board_pin]
    except KeyError as exc:
        known = ", ".join(str(pin) for pin in sorted(BOARD_TO_LINE))
        raise SystemExit(f"Unknown BOARD pin {board_pin}. Known pins: {known}") from exc


def resolve_line(board_pin: int, line_override: int | None) -> int:
    return line_override if line_override is not None else board_to_line(board_pin)


def build_line_map(args: argparse.Namespace) -> dict[int, str]:
    return {
        resolve_line(args.volume_up_board, args.volume_up_line): "volume_up",
        resolve_line(args.volume_down_board, args.volume_down_line): "volume_down",
        resolve_line(args.mute_board, args.mute_line): "mute",
    }


def run_volume_action(action: str, step: str, dry_run: bool) -> None:
    if action == "volume_up":
        cmd = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"+{step}"]
    elif action == "volume_down":
        cmd = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"-{step}"]
    elif action == "mute":
        cmd = ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"]
    else:
        return

    print(f"{time.strftime('%H:%M:%S')} {action}: {' '.join(cmd)}", flush=True)
    if not dry_run:
        subprocess.run(cmd, check=False)
        subprocess.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"], check=False)


def main() -> int:
    args = parse_args()
    if shutil.which("gpiomon") is None:
        raise SystemExit("gpiomon not found. Install libgpiod tools first.")
    if shutil.which("pactl") is None:
        raise SystemExit("pactl not found. PulseAudio volume control is unavailable.")

    line_to_action = build_line_map(args)
    lines = list(line_to_action)
    press_event = "1" if args.active_high else "0"
    bias = "pull-down" if args.active_high else "pull-up"
    cmd = [
        "gpiomon",
        "--line-buffered",
        f"--bias={bias}",
        "--format=%o %e",
        args.chip,
        *[str(line) for line in lines],
    ]

    print(f"chip: {args.chip}")
    print(f"logic: {'active-high/rising-edge press' if args.active_high else 'active-low/falling-edge press'}")
    print(f"duration: {args.duration:.1f}s")
    print(f"  {'volume_up':12s} -> BOARD {args.volume_up_board}")
    print(f"  {'volume_down':12s} -> BOARD {args.volume_down_board}")
    print(f"  {'mute':12s} -> BOARD {args.mute_board}")
    for line, action in line_to_action.items():
        print(f"  {action:12s} -> gpio line {line}")
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return 0

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_at = {line: 0.0 for line in lines}
    deadline = time.monotonic() + max(0.1, args.duration)
    try:
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            parts = line.strip().split()
            print(f"event: {line.strip()}", flush=True)
            if len(parts) < 2:
                continue
            try:
                offset = int(parts[0])
            except ValueError:
                continue
            event = parts[1]
            now = time.monotonic()
            if event != press_event:
                continue
            if now - last_at.get(offset, 0.0) < args.debounce_sec:
                continue
            last_at[offset] = now
            run_volume_action(line_to_action.get(offset, ""), args.step, args.dry_run)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("button volume test done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
