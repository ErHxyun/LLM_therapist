"""Test three GPIO buttons on the Jetson 40-pin header.

Default wiring:
  BOARD pin -> button -> GND

With this wiring, the script uses internal pull-up resistors and treats LOW as
pressed.
"""

from __future__ import annotations

import argparse
import time


DEFAULT_BUTTON_PINS = {
    "volume_up": 35,
    "volume_down": 37,
    "mute": 38,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print GPIO button press/release events.")
    parser.add_argument("--volume-up-pin", type=int, default=DEFAULT_BUTTON_PINS["volume_up"])
    parser.add_argument("--volume-down-pin", type=int, default=DEFAULT_BUTTON_PINS["volume_down"])
    parser.add_argument("--mute-pin", type=int, default=DEFAULT_BUTTON_PINS["mute"])
    parser.add_argument("--duration", type=float, default=20.0, help="Seconds to watch buttons.")
    parser.add_argument("--poll-sec", type=float, default=0.02, help="Polling interval in seconds.")
    parser.add_argument("--debounce-sec", type=float, default=0.08, help="Minimum stable time before reporting a change.")
    parser.add_argument("--show-raw", action="store_true", help="Print every raw HIGH/LOW change before debounce.")
    parser.add_argument(
        "--active-high",
        action="store_true",
        help="Use this if the button drives the GPIO HIGH when pressed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print pin configuration without touching GPIO.")
    return parser.parse_args()


def build_pin_map(args: argparse.Namespace) -> dict[str, int]:
    return {
        "volume_up": args.volume_up_pin,
        "volume_down": args.volume_down_pin,
        "mute": args.mute_pin,
    }


def run_dry(pin_map: dict[str, int], active_high: bool, duration: float) -> None:
    print("Dry run: no GPIO pins will be changed.")
    print(f"logic: {'active-high' if active_high else 'active-low with pull-up'}")
    print(f"duration: {duration:.1f}s")
    for name, pin in pin_map.items():
        print(f"  {name:12s} -> BOARD pin {pin}")


def is_pressed(value: int, high: int, low: int, active_high: bool) -> bool:
    return value == high if active_high else value == low


def run_gpio(
    pin_map: dict[str, int],
    *,
    duration: float,
    poll_sec: float,
    debounce_sec: float,
    active_high: bool,
    show_raw: bool,
) -> None:
    import Jetson.GPIO as GPIO

    pins = list(pin_map.values())
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)

    pull = GPIO.PUD_DOWN if active_high else GPIO.PUD_UP
    GPIO.setup(pins, GPIO.IN, pull_up_down=pull)

    raw_state = {name: GPIO.input(pin) for name, pin in pin_map.items()}
    stable_state = dict(raw_state)
    last_change_at = {name: time.monotonic() for name in pin_map}
    counts = {name: 0 for name in pin_map}
    deadline = time.monotonic() + max(0.1, duration)

    try:
        print("Watching GPIO buttons. Press Ctrl+C to stop.")
        print(f"logic: {'active-high' if active_high else 'active-low with pull-up'}")
        for name, pin in pin_map.items():
            print(f"  {name:12s} -> BOARD pin {pin}")

        while time.monotonic() < deadline:
            now = time.monotonic()
            for name, pin in pin_map.items():
                value = GPIO.input(pin)
                if value != raw_state[name]:
                    raw_state[name] = value
                    last_change_at[name] = now
                    if show_raw:
                        level = "HIGH" if value == GPIO.HIGH else "LOW"
                        print(f"{time.strftime('%H:%M:%S')} raw {name} -> {level}", flush=True)
                    continue
                if value != stable_state[name] and now - last_change_at[name] >= debounce_sec:
                    stable_state[name] = value
                    pressed = is_pressed(value, GPIO.HIGH, GPIO.LOW, active_high)
                    if pressed:
                        counts[name] += 1
                    action = "PRESSED" if pressed else "released"
                    print(f"{time.strftime('%H:%M:%S')} {name} {action} count={counts[name]}", flush=True)
            time.sleep(max(0.005, poll_sec))
    finally:
        GPIO.cleanup(pins)
        print("Button test complete.")
        for name, count in counts.items():
            print(f"  {name:12s}: {count} presses")


def main() -> int:
    args = parse_args()
    pin_map = build_pin_map(args)
    if args.dry_run:
        run_dry(pin_map, args.active_high, args.duration)
    else:
        run_gpio(
            pin_map,
            duration=args.duration,
            poll_sec=args.poll_sec,
            debounce_sec=args.debounce_sec,
            active_high=args.active_high,
            show_raw=args.show_raw,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
