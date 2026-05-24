"""Test CaiTI status LEDs through libgpiod CLI tools.

This bypasses Jetson.GPIO and talks to /dev/gpiochip0 through gpioset.

Default Jetson Orin Nano/NX 40-pin header mapping used here:
  white  BOARD 15 -> gpiochip0 line 85
  yellow BOARD 16 -> gpiochip0 line 126
  red    BOARD 18 -> gpiochip0 line 125
  green  BOARD 22 -> gpiochip0 line 123
"""

from __future__ import annotations

import argparse
import shutil
import subprocess


DEFAULT_LED_LINES = {
    "white": 85,
    "yellow": 126,
    "red": 125,
    "green": 123,
}

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

DEFAULT_LED_BOARDS = {
    "white": 15,
    "yellow": 16,
    "red": 18,
    "green": 22,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blink status LEDs with gpioset.")
    parser.add_argument("--chip", default="gpiochip0")
    parser.add_argument("--white-board", type=int, default=DEFAULT_LED_BOARDS["white"])
    parser.add_argument("--yellow-board", type=int, default=DEFAULT_LED_BOARDS["yellow"])
    parser.add_argument("--red-board", type=int, default=DEFAULT_LED_BOARDS["red"])
    parser.add_argument("--green-board", type=int, default=DEFAULT_LED_BOARDS["green"])
    parser.add_argument("--white-line", type=int)
    parser.add_argument("--yellow-line", type=int)
    parser.add_argument("--red-line", type=int)
    parser.add_argument("--green-line", type=int)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--single", choices=sorted(DEFAULT_LED_LINES), default="")
    parser.add_argument("--active-low", action="store_true")
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


def build_line_map(args: argparse.Namespace) -> dict[str, int]:
    return {
        "white": resolve_line(args.white_board, args.white_line),
        "yellow": resolve_line(args.yellow_board, args.yellow_line),
        "red": resolve_line(args.red_board, args.red_line),
        "green": resolve_line(args.green_board, args.green_line),
    }


def run_gpioset(chip: str, values: dict[int, int], seconds: float, active_low: bool, dry_run: bool) -> None:
    cmd = ["gpioset"]
    if active_low:
        cmd.append("--active-low")
    cmd.extend(["--mode=time", f"--sec={seconds:g}", chip])
    cmd.extend(f"{line}={value}" for line, value in sorted(values.items()))
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    if shutil.which("gpioset") is None:
        raise SystemExit("gpioset not found. Install libgpiod tools first.")

    line_map = build_line_map(args)
    on_value = 1
    off_value = 0
    sequence = [(args.single, line_map[args.single])] if args.single else list(line_map.items())

    print(f"chip: {args.chip}")
    print(f"logic: {'active-low' if args.active_low else 'active-high'}")
    print(f"  {'white':6s} -> BOARD {args.white_board}")
    print(f"  {'yellow':6s} -> BOARD {args.yellow_board}")
    print(f"  {'red':6s} -> BOARD {args.red_board}")
    print(f"  {'green':6s} -> BOARD {args.green_board}")
    for color, line in line_map.items():
        print(f"  {color:6s} -> gpio line {line}")

    all_off = {line: off_value for line in line_map.values()}
    for cycle in range(1, args.cycles + 1):
        for color, line in sequence:
            print(f"cycle {cycle}: {color} on", flush=True)
            values = dict(all_off)
            values[line] = on_value
            run_gpioset(args.chip, values, args.seconds, args.active_low, args.dry_run)

    if not args.single:
        print("all LEDs on", flush=True)
        run_gpioset(
            args.chip,
            {line: on_value for line in line_map.values()},
            args.seconds,
            args.active_low,
            args.dry_run,
        )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
