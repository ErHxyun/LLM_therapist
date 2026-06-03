import unittest

from src.hardware.music_mode_button import MusicModeButtonController, MusicModeButtonSettings


class MusicModeButtonControllerTests(unittest.TestCase):
    def test_press_cycles_once_until_release_rearms(self):
        events = []
        controller = MusicModeButtonController(
            MusicModeButtonSettings(enabled=True, debounce_sec=0.5, release_sec=0.2),
            on_press=lambda: events.append("cycle"),
        )

        self.assertTrue(controller._process_value(0, now=1.0))
        self.assertFalse(controller._process_value(0, now=1.1))
        self.assertFalse(controller._process_value(1, now=1.2))
        self.assertFalse(controller._process_value(1, now=1.5))
        self.assertTrue(controller._process_value(0, now=1.6))

        self.assertEqual(events, ["cycle", "cycle"])


if __name__ == "__main__":
    unittest.main()
