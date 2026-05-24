"""Test Jetson GPIO by physical BOARD pin numbers.

Examples:
  python scripts/test_jetson_gpio_board.py input --pin 32 --pull up
  python scripts/test_jetson_gpio_board.py blink --pin 15
  python scripts/test_jetson_gpio_board.py output --pin 15 --value high
"""

from __future__ import annotations

import argparse
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    import Jetson.GPIO as GPIO


PULLS = {
    "none": GPIO.PUD_OFF,
    "up": GPIO.PUD_UP,
    "down": GPIO.PUD_DOWN,
}

VALUES = {
    "high": GPIO.HIGH,
    "low": GPIO.LOW,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Jetson GPIO with BOARD pin numbers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    input_parser = subparsers.add_parser("input", help="Poll one input pin.")
    input_parser.add_argument("--pin", type=int, required=True, help="Physical BOARD pin number.")
    input_parser.add_argument("--pull", choices=sorted(PULLS), default="up")
    input_parser.add_argument("--seconds", type=float, default=20.0)
    input_parser.add_argument("--interval", type=float, default=0.2)

    output_parser = subparsers.add_parser("output", help="Set one output pin.")
    output_parser.add_argument("--pin", type=int, required=True, help="Physical BOARD pin number.")
    output_parser.add_argument("--value", choices=sorted(VALUES), required=True)
    output_parser.add_argument("--hold-seconds", type=float, default=2.0)

    blink_parser = subparsers.add_parser("blink", help="Blink one output pin.")
    blink_parser.add_argument("--pin", type=int, required=True, help="Physical BOARD pin number.")
    blink_parser.add_argument("--cycles", type=int, default=5)
    blink_parser.add_argument("--seconds", type=float, default=0.5)

    return parser.parse_args()


def setup() -> None:
    GPIO.setwarnings(True)
    GPIO.setmode(GPIO.BOARD)


def poll_input(pin: int, pull: str, seconds: float, interval: float) -> None:
    GPIO.setup(pin, GPIO.IN, pull_up_down=PULLS[pull])
    deadline = time.monotonic() + seconds
    print(f"Reading BOARD {pin} with pull-{pull}. Press Ctrl+C to stop.", flush=True)
    while time.monotonic() < deadline:
        print(f"{time.strftime('%H:%M:%S')} {GPIO.input(pin)}", flush=True)
        time.sleep(interval)


def set_output(pin: int, value: str, hold_seconds: float) -> None:
    GPIO.setup(pin, GPIO.OUT, initial=VALUES[value])
    print(f"BOARD {pin} = {value}", flush=True)
    time.sleep(hold_seconds)


def blink_output(pin: int, cycles: int, seconds: float) -> None:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    for cycle in range(1, cycles + 1):
        print(f"cycle {cycle}: high", flush=True)
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(seconds)
        print(f"cycle {cycle}: low", flush=True)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(seconds)


def main() -> int:
    args = parse_args()
    setup()
    try:
        if args.command == "input":
            poll_input(args.pin, args.pull, args.seconds, args.interval)
        elif args.command == "output":
            set_output(args.pin, args.value, args.hold_seconds)
        elif args.command == "blink":
            blink_output(args.pin, args.cycles, args.seconds)
    finally:
        GPIO.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
