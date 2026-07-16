import unittest

from src.handler_rl import OPENING_GREETING


class HandlerGreetingTests(unittest.TestCase):
    def test_opening_greeting_introduces_caiti_as_speaker(self):
        self.assertTrue(OPENING_GREETING.startswith("Hello, I am Caiti"))
        self.assertNotIn("Hello CaiTI", OPENING_GREETING)
        self.assertNotIn("daily", OPENING_GREETING.lower())


if __name__ == "__main__":
    unittest.main()
