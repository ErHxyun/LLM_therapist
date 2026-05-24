"""Test two Jetson BOARD-pin buttons for system volume up/down.

Default wiring:
  3.3V -- 10k resistor -- GPIO pin -- button -- GND

With this active-low wiring:
  idle    = 1
  pressed = 0

Defaults:
  volume up   = BOARD 32
  volume down = BOARD 33
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    import Jetson.GPIO as GPIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test BOARD-pin volume buttons.")
    parser.add_argument("--up-pin", type=int, default=32, help="BOARD pin for volume up.")
    parser.add_argument("--down-pin", type=int, default=33, help="BOARD pin for volume down.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--debounce-sec", type=float, default=0.25)
    parser.add_argument("--step", default="5%", help="Volume step passed to pactl, e.g. 5%.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing volume.")
    return parser.parse_args()


def run_volume_action(action: str, step: str, dry_run: bool) -> None:
    if action == "up":
        cmd = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"+{step}"]
    elif action == "down":
        cmd = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"-{step}"]
    else:
        return

    print(f"{time.strftime('%H:%M:%S')} volume_{action}: {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=False)
    subprocess.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"], check=False)


def main() -> int:
    args = parse_args()
    if shutil.which("pactl") is None:
        raise SystemExit("pactl not found. Cannot control PulseAudio volume.")

    pin_to_action = {
        args.up_pin: "up",
        args.down_pin: "down",
    }

    GPIO.setmode(GPIO.BOARD)
    for pin in pin_to_action:
        GPIO.setup(pin, GPIO.IN)

    last_values = {pin: GPIO.input(pin) for pin in pin_to_action}
    last_press_at = {pin: 0.0 for pin in pin_to_action}

    print("Testing volume buttons with BOARD pins.", flush=True)
    print(f"volume up   -> BOARD {args.up_pin}", flush=True)
    print(f"volume down -> BOARD {args.down_pin}", flush=True)
    print(f"debounce    -> {args.debounce_sec:.2f}s", flush=True)
    print(f"initial     -> " + ", ".join(f"BOARD {pin}={value}" for pin, value in last_values.items()), flush=True)
    print("Press buttons now. Ctrl+C to stop.", flush=True)

    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            for pin, action in pin_to_action.items():
                value = GPIO.input(pin)
                if last_values[pin] == 1 and value == 0:
                    if now - last_press_at[pin] >= args.debounce_sec:
                        last_press_at[pin] = now
                        run_volume_action(action, args.step, args.dry_run)
                    else:
                        print(f"{time.strftime('%H:%M:%S')} BOARD {pin}: bounce ignored", flush=True)
                last_values[pin] = value
            time.sleep(0.01)
    finally:
        GPIO.cleanup()

    print("volume button test done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
