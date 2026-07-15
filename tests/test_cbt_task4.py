import unittest

from src import CBT
from src.local_llm.types import GenerationResult, LLMTask


class CBTTask4Test(unittest.TestCase):
    def test_parse_cbt_decision_accepts_adapter_and_paper_formats(self):
        self.assertEqual(CBT.parse_cbt_decision("0"), "0")
        self.assertEqual(CBT.parse_cbt_decision("DECISION: 1"), "1")
        self.assertEqual(CBT.parse_cbt_decision("n/a", default="1"), "1")

    def test_extract_cbt_choice_number_accepts_spoken_numbers(self):
        self.assertEqual(CBT.extract_cbt_choice_number("1"), 1)
        self.assertEqual(CBT.extract_cbt_choice_number("One."), 1)
        self.assertEqual(CBT.extract_cbt_choice_number("I choose first"), 1)
        self.assertEqual(CBT.extract_cbt_choice_number("twenty one"), 21)

    def test_stage1_reasoner_uses_task4_stage1_adapter(self):
        calls = []
        original = CBT.llm_complete_task

        def fake_complete_task(task, system_content, user_content, max_new_tokens=None):
            calls.append((task, system_content, user_content, max_new_tokens))
            return GenerationResult(
                text="0",
                task=task,
                adapter="adapters/task4_cbt_stage1",
                raw_text="0",
            )

        try:
            CBT.llm_complete_task = fake_complete_task
            result = CBT.stage1_reasoner(
                "I keep missing deadlines and feel like a failure.",
                "If I miss one deadline, I am useless.",
            )
        finally:
            CBT.llm_complete_task = original

        self.assertEqual(result, "DECISION: 0")
        self.assertEqual(len(calls), 1)
        task, system_content, user_content, max_new_tokens = calls[0]
        self.assertEqual(task, LLMTask.TASK4_CBT_STAGE1)
        self.assertIn("cognitive behavioural therapy", system_content)
        self.assertIn("13 possible common cognitive distortions", system_content)
        self.assertIn("Response format:\nDECISION: 0/1", system_content)
        self.assertIn("STATEMENT: I keep missing deadlines", user_content)
        self.assertIn("UNHELPFUL_THOUGHTS: If I miss one deadline", user_content)
        self.assertTrue(user_content.strip().startswith('"STATEMENT:'))
        self.assertTrue(user_content.strip().endswith(';"'))
        self.assertEqual(max_new_tokens, 8)

    def test_stage2_reasoner_uses_task4_stage2_adapter(self):
        calls = []
        original = CBT.llm_complete_task

        def fake_complete_task(task, system_content, user_content, max_new_tokens=None):
            calls.append((task, system_content, user_content, max_new_tokens))
            return GenerationResult(
                text="0",
                task=task,
                adapter="adapters/task4_cbt_stage2",
                raw_text="0",
            )

        try:
            CBT.llm_complete_task = fake_complete_task
            result = CBT.stage2_reasoner(
                "I avoid speaking in meetings.",
                "People will think my ideas are silly.",
                "I can ask whether that has happened before.",
            )
        finally:
            CBT.llm_complete_task = original

        self.assertEqual(result, "DECISION: 0")
        self.assertEqual(len(calls), 1)
        task, system_content, user_content, max_new_tokens = calls[0]
        self.assertEqual(task, LLMTask.TASK4_CBT_STAGE2)
        self.assertIn("cognitive behavioural therapy", system_content)
        self.assertIn("challenges the unhelpful thoughts", system_content)
        self.assertIn("Response format:\nDECISION: 0/1", system_content)
        self.assertIn("UNHELPFUL_THOUGHTS: People will think", user_content)
        self.assertIn("CHALLENGE: I can ask", user_content)
        self.assertTrue(user_content.strip().startswith('"STATEMENT:'))
        self.assertTrue(user_content.strip().endswith(';"'))
        self.assertEqual(max_new_tokens, 8)

    def test_stage3_reasoner_uses_task4_stage3_adapter(self):
        calls = []
        original = CBT.llm_complete_task

        def fake_complete_task(task, system_content, user_content, max_new_tokens=None):
            calls.append((task, system_content, user_content, max_new_tokens))
            return GenerationResult(
                text="0",
                task=task,
                adapter="adapters/task4_cbt_stage3",
                raw_text="0",
            )

        try:
            CBT.llm_complete_task = fake_complete_task
            result = CBT.stage3_reasoner(
                "I avoid speaking in meetings.",
                "People will think my ideas are silly.",
                "I can ask whether that has happened before.",
                "My ideas may still have value.",
            )
        finally:
            CBT.llm_complete_task = original

        self.assertEqual(result, "DECISION: 0")
        self.assertEqual(len(calls), 1)
        task, system_content, user_content, max_new_tokens = calls[0]
        self.assertEqual(task, LLMTask.TASK4_CBT_STAGE3)
        self.assertIn("cognitive behavioural therapy", system_content)
        self.assertIn("reframes the unhelpful thoughts properly", system_content)
        self.assertIn("Response format:\nDECISION: 0/1", system_content)
        self.assertIn("CHALLENGE: I can ask", user_content)
        self.assertIn("REFRAME: My ideas may still have value.", user_content)
        self.assertTrue(user_content.strip().startswith('"STATEMENT:'))
        self.assertTrue(user_content.strip().endswith(';"'))
        self.assertEqual(max_new_tokens, 8)

    def test_cbt_failure_message_recommends_professional_help(self):
        self.assertIn("mental health professional", CBT.CBT_PROFESSIONAL_HELP_MESSAGE)

    def test_cbt_guides_use_bounded_generation(self):
        original = CBT.llm_complete
        events = []
        original_log = CBT.log_llm_event
        calls = []

        def fake_complete(system_content, user_content):
            calls.append((system_content, user_content))
            if "Stage 1" in system_content:
                return "GUIDE: Try naming the specific thought that shows up when this happens. Please answer in your own words."
            if "Stage 2" in system_content:
                return "GUIDE: Look at whether that thought is always true or whether there is another balanced view. Write your own challenge."
            return "GUIDE: Use your challenge to write a more balanced thought in your own words. Extra sentence. Another extra sentence."

        try:
            CBT.llm_complete = fake_complete
            CBT.log_llm_event = lambda **kwargs: events.append(kwargs)
            guide1 = CBT.stage1_guide("I miss deadlines.")
            guide2 = CBT.stage2_guide("I miss deadlines.", "I am failing.")
            guide3 = CBT.stage3_guide(
                "I miss deadlines.",
                "I am failing.",
                "Missing one deadline does not mean I am failing.",
            )
        finally:
            CBT.llm_complete = original
            CBT.log_llm_event = original_log

        self.assertEqual(len(calls), 3)
        self.assertIn("Do not write the unhelpful thought for the client", calls[0][0])
        self.assertIn("Do not ask what the statement means", calls[0][0])
        self.assertIn("Focus only on the thought, belief, or self-talk", calls[0][0])
        self.assertIn('Prefer asking for "one unhelpful thought"', calls[0][0])
        self.assertIn("Do not write the challenge for the client", calls[1][0])
        self.assertIn("do not provide content-specific evidence", calls[1][0])
        self.assertIn("Do not ask for examples or past situations", calls[1][0])
        self.assertIn("Do not write the reframe for the client", calls[2][0])
        self.assertIn("Do not provide a possible balanced thought", calls[2][0])
        self.assertIn("Do not ask for examples or other situations", calls[2][0])
        self.assertIn("Do not mention worth, value, competence, identity", calls[2][0])
        self.assertIn("STATEMENT: I miss deadlines.", calls[0][1])
        self.assertIn("UNHELPFUL_THOUGHTS: I am failing.", calls[1][1])
        self.assertIn("CHALLENGE: Missing one deadline", calls[2][1])
        self.assertIn("specific thought", guide1)
        self.assertIn("always true", guide2)
        self.assertIn("more balanced thought", guide3)
        self.assertNotIn("Another extra sentence", guide3)
        self.assertEqual([event["task"] for event in events], [
            "cbt_stage1_guide",
            "cbt_stage2_guide",
            "cbt_stage3_guide",
        ])
        self.assertEqual([event["metadata"]["mode"] for event in events], [
            "bounded_generation",
            "bounded_generation",
            "bounded_generation",
        ])

    def test_stage3_guide_falls_back_when_generation_adds_reassurance_content(self):
        original = CBT.llm_complete
        original_log = CBT.log_llm_event
        events = []

        try:
            CBT.llm_complete = lambda *_args: (
                "GUIDE: Can you write a balanced thought that says one missed deadline "
                "does not define your worth as a person?"
            )
            CBT.log_llm_event = lambda **kwargs: events.append(kwargs)
            guide = CBT.stage3_guide(
                "I miss deadlines.",
                "I am failing.",
                "Missing one deadline does not mean I am failing.",
            )
        finally:
            CBT.llm_complete = original
            CBT.log_llm_event = original_log

        self.assertEqual(
            guide,
            "GUIDE: Please use your challenge to write one balanced sentence in your own words.",
        )
        self.assertNotIn("worth", guide.lower())
        self.assertEqual(events[0]["normalized_output"], guide)

    def test_stage3_recap_mirrors_user_challenge_without_llm(self):
        original = CBT.llm_complete
        events = []
        original_log = CBT.log_llm_event

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("CBT recap should not call the LLM")

        try:
            CBT.llm_complete = fail_if_called
            CBT.log_llm_event = lambda **kwargs: events.append(kwargs)
            recap = CBT.recap_stage3_challenge(
                "I miss deadlines.",
                "I am failing.",
                "Missing one deadline does not mean I am failing.",
            )
        finally:
            CBT.llm_complete = original
            CBT.log_llm_event = original_log

        self.assertEqual(
            recap,
            'You challenged the thought by saying: "Missing one deadline does not mean I am failing."',
        )
        self.assertEqual(events[0]["task"], "cbt_stage3_recap")

    def test_stage0_prompter_returns_cleaned_model_question(self):
        original = CBT._chat_complete
        try:
            CBT._chat_complete = lambda *_args: "QUESTION: Which topic would you like to work on?"
            generated = CBT.stage0_prompter("screening history")
            CBT._chat_complete = lambda *_args: ""
            empty = CBT.stage0_prompter("screening history")
        finally:
            CBT._chat_complete = original

        self.assertEqual(generated, "Which topic would you like to work on?")
        self.assertEqual(empty, "")

    def test_run_cbt_uses_stage0_fallback_for_empty_generation(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "weight",
                    "name": "Maintaining stable weight",
                    "question": ["Have your weight changed significantly recently?"],
                    "score": [2],
                    "notes": [],
                }
            }
        }
        responses = iter([
            "1",
            "If I gain weight, I am failing.",
            "Gaining weight does not mean I am failing.",
            "I can work on regular meals without judging myself.",
        ])
        logged = []
        originals = {
            "stage0_prompter": CBT.stage0_prompter,
            "log_question": CBT.log_question,
            "log_system_message": CBT.log_system_message,
            "set_question_prefix": CBT.set_question_prefix,
            "get_resp_log": CBT.get_resp_log,
            "stage1_reasoner": CBT.stage1_reasoner,
            "stage2_reasoner": CBT.stage2_reasoner,
            "stage3_reasoner": CBT.stage3_reasoner,
        }
        try:
            CBT.stage0_prompter = lambda _history: ""
            CBT.log_question = lambda text: logged.append(str(text))
            CBT.log_system_message = lambda text: logged.append(str(text))
            CBT.set_question_prefix = lambda _text: None
            CBT.get_resp_log = lambda: next(responses)
            CBT.stage1_reasoner = lambda *_args, **_kwargs: "DECISION: 0"
            CBT.stage2_reasoner = lambda *_args, **_kwargs: "DECISION: 0"
            CBT.stage3_reasoner = lambda *_args, **_kwargs: "DECISION: 0"
            CBT.run_cbt(question_lib)
        finally:
            for name, value in originals.items():
                setattr(CBT, name, value)

        self.assertIn(
            "Thank you for answering the questions. Based on your earlier responses",
            logged[0],
        )

    def test_stage0_history_and_statement_use_all_rv_responses(self):
        entry = {
            "label": "weight",
            "name": "Maintaining stable weight",
            "question": [
                "Have your weight changed significantly recently?",
                "What have you noticed about your weight lately?",
            ],
            "score": [2],
            "notes": [
                [
                    "original_resp: My weight increased recently.",
                    "followup_resp: I like painting.",
                    "followup_resp_1: I stress eat during deadlines.",
                    "rv_validation: That connects to the weight change.",
                ],
                ["CBT_stage: old_session"],
            ],
        }

        history = CBT.build_cbt_dimension_history(entry)
        statement = CBT.build_cbt_statement(entry)

        self.assertIn("Have your weight changed significantly recently?", history)
        self.assertIn("What have you noticed about your weight lately?", history)
        self.assertIn("My weight increased recently.", history)
        self.assertIn("I like painting.", history)
        self.assertIn("I stress eat during deadlines.", history)
        self.assertNotIn("CBT_stage", history)
        self.assertEqual(statement, "My weight increased recently. I like painting. I stress eat during deadlines.")


if __name__ == "__main__":
    unittest.main()
