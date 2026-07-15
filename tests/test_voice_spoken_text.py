import unittest

from src.voice.io_loop import clean_spoken_text, split_spoken_chunks


class VoiceSpokenTextTests(unittest.TestCase):
    def test_split_spoken_chunks_preserves_gap_boundaries(self):
        raw = (
            "Hello, I am Caiti, your AI therapist. Thank you for joining me today. "
            "Let's get started with a couple of questions about your recent daily life.\n\n"
            "How have your eating habits been day to day, and have you been keeping a regular eating schedule?"
        )
        chunks = split_spoken_chunks(raw)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("Hello, I am Caiti"))
        self.assertTrue(chunks[1].startswith("How have your eating habits"))

    def test_clean_spoken_text_softens_opening_and_question_boundary(self):
        raw = (
            "Hello, I am Caiti, your AI therapist. Thank you for joining me today. "
            "Let's get started with a couple of questions about your recent daily life.\n\n"
            "How have your eating habits been day to day, and have you been keeping a regular eating schedule?"
        )
        cleaned = clean_spoken_text(raw)
        self.assertNotIn("\n\n", cleaned)
        self.assertIn("your recent daily life, and How have your eating habits", cleaned)
        self.assertIn("your AI therapist, Thank you for joining me today, Let's get started", cleaned)

    def test_clean_spoken_text_strips_spoken_labels(self):
        raw = "VALIDATION: That sounds difficult.\n\nCan you tell me a little more?"
        cleaned = clean_spoken_text(raw)
        self.assertEqual(cleaned, "That sounds difficult, and Can you tell me a little more?")


if __name__ == "__main__":
    unittest.main()
