#!/usr/bin/env python3
"""Print large session-button events from a Jetson BOARD pin."""

from __future__ import annotations

import argparse
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    import Jetson.GPIO as GPIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CaiTI's large session button.")
    parser.add_argument("--pin", type=int, default=37, help="BOARD pin connected to the large button.")
    parser.add_argument("--long-press-sec", type=float, default=3.0)
    parser.add_argument("--active-low", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--poll-sec", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pressed_value = GPIO.LOW if args.active_low else GPIO.HIGH
    released_value = GPIO.HIGH if args.active_low else GPIO.LOW
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(args.pin, GPIO.IN)

    pressed = GPIO.input(args.pin) == pressed_value
    press_started_at = time.monotonic() if pressed else None
    long_emitted = False
    print(f"Testing BOARD {args.pin}. Expected idle={released_value}, pressed={pressed_value}. Ctrl+C to stop.")

    try:
        while True:
            now = time.monotonic()
            value = GPIO.input(args.pin)
            is_pressed = value == pressed_value
            if is_pressed and not pressed:
                pressed = True
                press_started_at = now
                long_emitted = False
                print("pressed", flush=True)
            elif is_pressed and pressed and press_started_at is not None:
                if not long_emitted and now - press_started_at >= args.long_press_sec:
                    long_emitted = True
                    print("long_press", flush=True)
            elif not is_pressed and pressed:
                duration = now - (press_started_at or now)
                pressed = False
                press_started_at = None
                if not long_emitted:
                    print(f"short_press duration={duration:.2f}s", flush=True)
                else:
                    print(f"released_after_long duration={duration:.2f}s", flush=True)
            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        return 130
    finally:
        GPIO.cleanup([args.pin])


if __name__ == "__main__":
    raise SystemExit(main())
