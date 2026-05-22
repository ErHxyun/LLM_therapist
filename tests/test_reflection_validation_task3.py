import unittest

from src.local_llm.types import GenerationResult, LLMTask
from src import reflection_validation


class ReflectionValidationTask3Test(unittest.TestCase):
    def test_parse_rv_decision_accepts_adapter_and_paper_formats(self):
        self.assertEqual(reflection_validation.parse_rv_decision("0"), "0")
        self.assertEqual(reflection_validation.parse_rv_decision("DECISION: 1"), "1")
        self.assertEqual(reflection_validation.parse_rv_decision("n/a", default="1"), "1")

    def test_rv_reasoner_uses_task3_adapter_and_normalizes_decision(self):
        calls = []
        original = reflection_validation.llm_complete_task

        def fake_complete_task(task, system_content, user_content, max_new_tokens=None):
            calls.append((task, system_content, user_content, max_new_tokens))
            return GenerationResult(
                text="0",
                task=task,
                adapter="adapters/task3_rv_reasoner",
                raw_text="0",
            )

        try:
            reflection_validation.llm_complete_task = fake_complete_task
            result = reflection_validation.rv_reasoner(
                "weight",
                "Have your weight changed significantly recently?",
                "My weight increased a lot recently.",
                "I have upcoming deadlines, so I often do stress eating.",
            )
        finally:
            reflection_validation.llm_complete_task = original

        self.assertEqual(result, "DECISION: 0")
        self.assertEqual(len(calls), 1)
        task, system_content, user_content, max_new_tokens = calls[0]
        self.assertEqual(task, LLMTask.TASK3_RV_REASONER)
        self.assertIn("conversation topic", system_content)
        self.assertIn('"Topic":', user_content)
        self.assertIn('"weight"', user_content)
        self.assertIn('"Original Response":', user_content)
        self.assertIn("My weight increased", user_content)
        self.assertIn('"Follow Up Response":', user_content)
        self.assertIn("I have upcoming deadlines", user_content)
        self.assertEqual(max_new_tokens, 8)

    def test_rv_guide_redirects_to_original_response_without_llm(self):
        events = []
        original_log = reflection_validation.log_llm_event

        try:
            reflection_validation.log_llm_event = lambda **kwargs: events.append(kwargs)
            result = reflection_validation.rv_guide(
                "weight",
                "Have your weight changed significantly recently?",
                "My weight increased a lot recently.",
                "I like painting.",
            )
        finally:
            reflection_validation.log_llm_event = original_log

        self.assertEqual(
            result,
            'Guide: Thank you for sharing that. I want to return to what you mentioned earlier: '
            '"My weight increased a lot recently." Could you tell me more about that?',
        )
        self.assertNotIn("painting", result.lower())
        self.assertEqual(events[0]["task"], "rv_guide")
        self.assertEqual(events[0]["dimension"], "weight")
        self.assertEqual(events[0]["metadata"]["mode"], "redirect")

    def test_rv_guide_routes_serious_risk_to_professional_help(self):
        events = []
        original_log = reflection_validation.log_llm_event

        try:
            reflection_validation.log_llm_event = lambda **kwargs: events.append(kwargs)
            result = reflection_validation.rv_guide(
                "mood",
                "How has your mood been recently?",
                "I want to die.",
                "I like painting.",
            )
        finally:
            reflection_validation.log_llm_event = original_log

        self.assertIn("qualified professional", result)
        self.assertIn("immediate danger", result)
        self.assertIn("988", result)
        self.assertEqual(events[0]["metadata"]["mode"], "professional_help")

    def test_rv_guide_uses_topic_when_original_response_missing(self):
        events = []
        original_log = reflection_validation.log_llm_event

        try:
            reflection_validation.log_llm_event = lambda **kwargs: events.append(kwargs)
            result = reflection_validation.rv_guide(
                "weight",
                "Have your weight changed significantly recently?",
                "",
                "I like painting.",
            )
        finally:
            reflection_validation.log_llm_event = original_log

        self.assertEqual(
            result,
            "Guide: Thank you for sharing that. I want to return to the current topic, "
            "weight. Could you tell me more about that?",
        )

    def test_rv_validation_uses_bounded_generation_and_cleans_output(self):
        calls = []
        events = []
        original_complete = reflection_validation.llm_complete
        original_log = reflection_validation.log_llm_event

        def fake_complete(system_content, user_content):
            calls.append((system_content, user_content))
            return (
                "VALIDATION: You mentioned stress eating during deadlines. "
                "That connects with the weight change you shared earlier. "
                "It took care to explain that connection. What else?"
            )

        try:
            reflection_validation.llm_complete = fake_complete
            reflection_validation.log_llm_event = lambda **kwargs: events.append(kwargs)
            result = reflection_validation.rv_validation(
                "weight",
                "Have your weight changed significantly recently?",
                "My weight increased a lot recently.",
                "I often do stress eating during deadlines.",
            )
        finally:
            reflection_validation.llm_complete = original_complete
            reflection_validation.log_llm_event = original_log

        self.assertEqual(
            result,
            "VALIDATION: You mentioned stress eating during deadlines. "
            "That connects with the weight change you shared earlier. "
            "It took care to explain that connection.",
        )
        self.assertEqual(len(calls), 1)
        system_content, user_content = calls[0]
        self.assertIn("affective reflection", system_content)
        self.assertIn("only for emotion words the client explicitly stated", system_content)
        self.assertIn('Do not say "you feel"', system_content)
        self.assertIn("Do not ask questions", system_content)
        self.assertIn('"Topic":', user_content)
        self.assertIn('"Original Response":', user_content)
        self.assertIn('"Follow-up Response":', user_content)
        self.assertNotIn("?", result)
        self.assertEqual(events[0]["task"], "rv_validation")
        self.assertEqual(events[0]["dimension"], "weight")
        self.assertEqual(events[0]["raw_llm_output"], calls and (
            "VALIDATION: You mentioned stress eating during deadlines. "
            "That connects with the weight change you shared earlier. "
            "It took care to explain that connection. What else?"
        ))
        self.assertEqual(events[0]["metadata"]["mode"], "bounded_generation")

    def test_rv_validation_cleaner_removes_unstated_emotion_sentence(self):
        result = reflection_validation.clean_rv_validation_text(
            (
                "VALIDATION: You mentioned stress eating during deadlines. "
                "Many people feel overwhelmed in that situation."
            ),
            source_texts=(
                "My weight increased a lot recently.",
                "I often do stress eating during deadlines.",
            ),
        )

        self.assertEqual(result, "VALIDATION: You mentioned stress eating during deadlines.")
        self.assertNotIn("overwhelmed", result.lower())

    def test_rv_validation_routes_serious_risk_to_professional_help(self):
        events = []
        original_log = reflection_validation.log_llm_event

        try:
            reflection_validation.log_llm_event = lambda **kwargs: events.append(kwargs)
            result = reflection_validation.rv_validation(
                "mood",
                "How has your mood been recently?",
                "I am sad recently.",
                "I want to die.",
            )
        finally:
            reflection_validation.log_llm_event = original_log

        self.assertTrue(result.startswith("VALIDATION: "))
        self.assertIn("qualified professional", result)
        self.assertIn("immediate danger", result)
        self.assertIn("988", result)
        self.assertEqual(events[0]["metadata"]["mode"], "professional_help")


if __name__ == "__main__":
    unittest.main()
