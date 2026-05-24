"""Test CaiTI status LEDs on the Jetson 40-pin header.

Default BOARD pins:
  white  -> 15
  yellow -> 16
  red    -> 18
  green  -> 22
"""

from __future__ import annotations

import argparse
import time


DEFAULT_LED_PINS = {
    "white": 15,
    "yellow": 16,
    "red": 18,
    "green": 22,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blink the CaiTI status LEDs connected to Jetson GPIO pins.")
    parser.add_argument("--white-pin", type=int, default=DEFAULT_LED_PINS["white"])
    parser.add_argument("--yellow-pin", type=int, default=DEFAULT_LED_PINS["yellow"])
    parser.add_argument("--red-pin", type=int, default=DEFAULT_LED_PINS["red"])
    parser.add_argument("--green-pin", type=int, default=DEFAULT_LED_PINS["green"])
    parser.add_argument("--seconds", type=float, default=0.5, help="Seconds each LED stays on.")
    parser.add_argument("--cycles", type=int, default=2, help="Number of chase cycles.")
    parser.add_argument(
        "--active-low",
        action="store_true",
        help="Use this when LEDs connect to 3.3V and GPIO turns them on by pulling LOW.",
    )
    parser.add_argument(
        "--single",
        choices=sorted(DEFAULT_LED_PINS),
        default="",
        help="Only test one LED color.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the sequence without touching GPIO.")
    return parser.parse_args()


def build_pin_map(args: argparse.Namespace) -> dict[str, int]:
    return {
        "white": args.white_pin,
        "yellow": args.yellow_pin,
        "red": args.red_pin,
        "green": args.green_pin,
    }


def run_dry(pin_map: dict[str, int], seconds: float, cycles: int, active_low: bool, single: str) -> None:
    print("Dry run: no GPIO pins will be changed.")
    print(f"logic: {'active-low' if active_low else 'active-high'}")
    print("BOARD pin map:")
    for color, pin in pin_map.items():
        print(f"  {color:6s} -> pin {pin}")
    sequence = [(single, pin_map[single])] if single else list(pin_map.items())
    for cycle in range(1, cycles + 1):
        for color, pin in sequence:
            print(f"cycle {cycle}: {color} on at BOARD pin {pin} for {seconds:.2f}s")


def run_gpio(pin_map: dict[str, int], seconds: float, cycles: int, active_low: bool, single: str) -> None:
    import Jetson.GPIO as GPIO

    pins = list(pin_map.values())
    on_value = GPIO.LOW if active_low else GPIO.HIGH
    off_value = GPIO.HIGH if active_low else GPIO.LOW
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pins, GPIO.OUT, initial=off_value)

    try:
        print("Testing status LEDs. Press Ctrl+C to stop.")
        print(f"logic: {'active-low' if active_low else 'active-high'}")
        for color, pin in pin_map.items():
            print(f"  {color:6s} -> BOARD pin {pin}")

        sequence = [(single, pin_map[single])] if single else list(pin_map.items())
        for cycle in range(1, cycles + 1):
            for color, pin in sequence:
                print(f"cycle {cycle}: {color} on")
                GPIO.output(pins, off_value)
                GPIO.output(pin, on_value)
                time.sleep(seconds)

        if not single:
            print("all LEDs on")
            GPIO.output(pins, on_value)
            time.sleep(seconds)
        GPIO.output(pins, off_value)
        print("done")
    finally:
        GPIO.output(pins, off_value)
        GPIO.cleanup(pins)


def main() -> int:
    args = parse_args()
    pin_map = build_pin_map(args)
    if args.dry_run:
        run_dry(pin_map, args.seconds, args.cycles, args.active_low, args.single)
    else:
        run_gpio(pin_map, args.seconds, args.cycles, args.active_low, args.single)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
