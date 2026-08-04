import unittest

from src.hardware.session_button import SessionButtonController, SessionButtonSettings


class SessionButtonControllerTests(unittest.TestCase):
    def test_default_debounce_is_half_a_second(self):
        self.assertEqual(SessionButtonSettings().debounce_sec, 0.5)

    def test_short_press_emits_short_callback_on_release(self):
        events = []
        controller = SessionButtonController(
            SessionButtonSettings(enabled=True, board_pin=37, long_press_sec=3.0, debounce_sec=0.05),
            on_short_press=lambda: events.append("short"),
            on_long_press=lambda: events.append("long"),
        )

        self.assertIsNone(controller._process_value(0, 0.0))
        self.assertEqual(controller._process_value(1, 0.2), "short")

        self.assertEqual(events, ["short"])

    def test_long_press_emits_long_once_and_no_short_on_release(self):
        events = []
        controller = SessionButtonController(
            SessionButtonSettings(enabled=True, board_pin=37, long_press_sec=3.0, debounce_sec=0.05),
            on_short_press=lambda: events.append("short"),
            on_long_press=lambda: events.append("long"),
        )

        controller._process_value(0, 1.0)
        self.assertEqual(controller._process_value(0, 4.1), "long")
        self.assertIsNone(controller._process_value(0, 4.2))
        self.assertIsNone(controller._process_value(1, 4.3))

        self.assertEqual(events, ["long"])


if __name__ == "__main__":
    unittest.main()
