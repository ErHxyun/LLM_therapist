import unittest

from src.utils import text_generators


class TextGeneratorTests(unittest.TestCase):
    def test_prompt_preserves_question_capitalization(self):
        prompt = text_generators.generate_prompt_synonymous_sentences(
            "How's your eating? Are you eating regularly?"
        )

        self.assertIn("How's your eating? Are you eating regularly?", prompt)

    def test_clean_question_rewrite_keeps_first_question_before_answer_artifact(self):
        cleaned = text_generators.clean_question_rewrite(
            "How have your eating habits been, and are you eating regularly?\n=> I am having problem eating well."
        )

        self.assertEqual(cleaned, "How have your eating habits been, and are you eating regularly?")

    def test_invalid_question_rewrite_rejects_statement_outputs(self):
        self.assertFalse(text_generators.is_valid_question_rewrite("My mood has been unstable lately."))
        self.assertFalse(text_generators.is_valid_question_rewrite("I am having problem eating well?"))
        self.assertTrue(text_generators.is_valid_question_rewrite("How has your mood been lately?"))

    def test_generate_synonymous_sentences_falls_back_to_original_question(self):
        original_llm_complete = text_generators.llm_complete
        try:
            text_generators.llm_complete = lambda *_args: "My mood has been unstable lately."
            question = "How's your mood recently?"
            self.assertEqual(text_generators.generate_synonymous_sentences(question), question)
        finally:
            text_generators.llm_complete = original_llm_complete


if __name__ == "__main__":
    unittest.main()
