import unittest

from src.local_llm.types import GenerationResult, LLMTask
from src import response_analyzer


class ResponseAnalyzerTask1Test(unittest.TestCase):
    def test_dimension_score_classifier_uses_task1_adapter_plain_prompt(self):
        calls = []
        original = response_analyzer.llm_complete_task

        def fake_complete_task(task, system_content, user_content, max_new_tokens=None):
            calls.append((task, system_content, user_content, max_new_tokens))
            return GenerationResult(
                text="1_weight, 2",
                task=task,
                adapter="adapters/task1_response_analyzer",
                raw_text="1_weight, 2",
            )

        try:
            response_analyzer.llm_complete_task = fake_complete_task
            result = response_analyzer.classify_dimension_and_score(
                "My weight increased a lot recently.",
                "Have your weight changed significantly recently?",
            )
        finally:
            response_analyzer.llm_complete_task = original

        self.assertEqual(result, "weight, 2")
        self.assertEqual(len(calls), 1)
        task, system_content, user_content, max_new_tokens = calls[0]
        self.assertEqual(task, LLMTask.TASK1_RESPONSE_ANALYZER)
        self.assertEqual(
            user_content,
            '{"in":"My weight increased a lot recently.", "res":',
        )
        self.assertIn("To assign the user input with DIMENSION and SCORE", system_content)
        self.assertEqual(max_new_tokens, 16)

    def test_dimension_score_result_preserves_raw_output(self):
        original = response_analyzer.llm_complete_task

        def fake_complete_task(task, system_content, user_content, max_new_tokens=None):
            return GenerationResult(
                text='"weight, 2"}\n{"in":"extra"',
                task=task,
                adapter="adapters/task1_response_analyzer",
                raw_text='"weight, 2"}\n{"in":"extra"',
            )

        try:
            response_analyzer.llm_complete_task = fake_complete_task
            result = response_analyzer.classify_dimension_and_score_result(
                "My weight increased a lot recently.",
                "Have your weight changed significantly recently?",
            )
        finally:
            response_analyzer.llm_complete_task = original

        self.assertEqual(result.raw_output, '"weight, 2"}\n{"in":"extra"')
        self.assertEqual(result.normalized_output, "weight, 2")
        self.assertTrue(result.is_valid)

    def test_task1_input_escapes_user_text(self):
        self.assertEqual(
            response_analyzer._format_task1_input('He said "I gained weight."'),
            '{"in":"He said \\"I gained weight.\\"", "res":',
        )


if __name__ == "__main__":
    unittest.main()
