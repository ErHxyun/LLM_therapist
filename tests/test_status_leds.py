import unittest

from src.hardware.status_leds import StatusLEDController, StatusLEDSettings


class FakeGPIO:
    BOARD = "BOARD"
    OUT = "OUT"
    HIGH = 1
    LOW = 0

    def __init__(self):
        self.mode = None
        self.setups = []
        self.outputs = []
        self.cleaned = []

    def setmode(self, mode):
        self.mode = mode

    def setup(self, pins, direction, initial=None):
        self.setups.append((list(pins), direction, initial))

    def output(self, pin, value):
        self.outputs.append((pin, value))

    def cleanup(self, pins=None):
        self.cleaned.append(list(pins or []))


class StatusLEDControllerTests(unittest.TestCase):
    def test_start_sets_white_on_and_others_off(self):
        gpio = FakeGPIO()
        controller = StatusLEDController(
            StatusLEDSettings(enabled=True, white_pin=15, yellow_pin=16, red_pin=18, green_pin=22),
            gpio_module=gpio,
        )

        controller.start()

        self.assertEqual(gpio.mode, gpio.BOARD)
        self.assertEqual(gpio.setups, [([15, 16, 18, 22], gpio.OUT, gpio.LOW)])
        self.assertEqual(
            gpio.outputs[-4:],
            [(15, gpio.HIGH), (16, gpio.LOW), (18, gpio.LOW), (22, gpio.LOW)],
        )

    def test_status_methods_drive_expected_colors(self):
        gpio = FakeGPIO()
        controller = StatusLEDController(
            StatusLEDSettings(enabled=True, white_pin=15, yellow_pin=16, red_pin=18, green_pin=22),
            gpio_module=gpio,
        )
        controller.start()

        controller.mark_session_started()
        controller.set_tts_active(True)
        controller.set_tts_active(False)
        controller.set_stt_active(True)
        controller.set_stt_active(False)
        controller.stop()

        self.assertIn((16, gpio.HIGH), gpio.outputs)
        self.assertIn((18, gpio.HIGH), gpio.outputs)
        self.assertIn((18, gpio.LOW), gpio.outputs)
        self.assertIn((22, gpio.HIGH), gpio.outputs)
        self.assertIn((22, gpio.LOW), gpio.outputs)
        self.assertEqual(gpio.cleaned, [[15, 16, 18, 22]])

    def test_status_methods_publish_monitor_state(self):
        events = []
        gpio = FakeGPIO()
        monitor = type(
            "FakeMonitor",
            (),
            {"set_light": lambda self, color, active: events.append((color, active))},
        )()
        controller = StatusLEDController(
            StatusLEDSettings(enabled=True, white_pin=15, yellow_pin=16, red_pin=18, green_pin=22),
            gpio_module=gpio,
            status_monitor=monitor,
        )

        controller.start()
        controller.set_tts_active(True)
        controller.set_tts_active(False)
        controller.stop()

        self.assertIn(("white", True), events)
        self.assertIn(("red", True), events)
        self.assertIn(("red", False), events)
        self.assertEqual(events[-4:], [("white", False), ("yellow", False), ("red", False), ("green", False)])


if __name__ == "__main__":
    unittest.main()
