import unittest

from src.hardware.volume_buttons import VolumeButtonController, VolumeButtonSettings, _ButtonState


class _Completed:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


class VolumeButtonControllerTests(unittest.TestCase):
    def test_volume_up_clamps_to_max_percent(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[1] == "get-sink-volume":
                return _Completed("Volume: front-left: 64225 /  98% / -0.50 dB")
            return _Completed()

        controller = VolumeButtonController(
            VolumeButtonSettings(enabled=True, step_percent=5, max_percent=100),
            command_runner=runner,
        )

        controller._change_volume("up")

        self.assertEqual(calls[-1][0], ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "100%"])

    def test_volume_down_clamps_to_min_percent(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[1] == "get-sink-volume":
                return _Completed("Volume: front-left: 2000 /   2% / -50.00 dB")
            return _Completed()

        controller = VolumeButtonController(
            VolumeButtonSettings(enabled=True, step_percent=5, min_percent=0),
            command_runner=runner,
        )

        controller._change_volume("down")

        self.assertEqual(calls[-1][0], ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "0%"])

    def test_press_requires_release_before_retrigger(self):
        controller = VolumeButtonController(
            VolumeButtonSettings(enabled=True, debounce_sec=0.5, release_sec=0.2)
        )
        controller._pin_to_action = {32: "up"}
        controller._button_states = {32: _ButtonState(armed=True)}

        self.assertEqual(controller._process_pin_value(32, 0, now=1.0), "up")
        self.assertIsNone(controller._process_pin_value(32, 0, now=2.0))
        self.assertIsNone(controller._process_pin_value(32, 1, now=2.1))
        self.assertIsNone(controller._process_pin_value(32, 0, now=2.2))
        self.assertIsNone(controller._process_pin_value(32, 1, now=2.3))
        self.assertIsNone(controller._process_pin_value(32, 1, now=2.6))
        self.assertEqual(controller._process_pin_value(32, 0, now=2.7), "up")


if __name__ == "__main__":
    unittest.main()
