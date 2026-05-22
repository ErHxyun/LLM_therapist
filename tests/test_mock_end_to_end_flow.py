import os
import sqlite3
import tempfile
import unittest

from src import CBT, questioner, reflection_validation
from src.local_llm.types import GenerationResult
from src.utils import response_bridge
from src.utils.llm_output_contracts import normalize_task1_output


class MockEndToEndFlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "events.sqlite3")
        self.old_db_path = os.environ.get("CAITI_STRUCTURED_LOG_DB")
        self.old_session_id = os.environ.get("CAITI_SESSION_ID")
        os.environ["CAITI_STRUCTURED_LOG_DB"] = self.db_path
        os.environ["CAITI_SESSION_ID"] = "mock-e2e-session"

    def tearDown(self):
        if self.old_db_path is None:
            os.environ.pop("CAITI_STRUCTURED_LOG_DB", None)
        else:
            os.environ["CAITI_STRUCTURED_LOG_DB"] = self.old_db_path
        if self.old_session_id is None:
            os.environ.pop("CAITI_SESSION_ID", None)
        else:
            os.environ["CAITI_SESSION_ID"] = self.old_session_id
        self.tmp.cleanup()

    def _events(self):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(
                """
                SELECT task, adapter, dimension, score, segment_text, raw_llm_output, normalized_output
                FROM session_events
                ORDER BY id
                """
            ).fetchall()
        finally:
            conn.close()

    def test_screening_rv_and_cbt_complete_with_structured_logs(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "weight",
                    "name": "Maintaining stable weight",
                    "question": ["Have your weight changed significantly recently?"],
                    "score": [],
                    "notes": [],
                    "Yes": 2,
                    "No": 0,
                }
            }
        }
        question_text = question_lib["1"]["1"]["question"][0]
        user_segments = ["My weight increased a lot recently."]
        rv_responses = iter([
            "I like painting.",
            "I often do stress eating during deadlines.",
        ])
        rv_decisions = iter(["1", "0"])
        cbt_responses = iter([
            "1",
            "If I gain weight, I am failing.",
            "Deadlines affect my eating, but gaining weight does not mean I am failing.",
            "I can plan regular meals during deadlines and treat this as something to work on.",
        ])
        logged_questions = []
        prefixes = []

        originals = {
            "response_bridge.classify_dimension_and_score_result": (
                response_bridge.classify_dimension_and_score_result
            ),
            "reflection_validation.llm_complete_task": reflection_validation.llm_complete_task,
            "questioner.generate_change": questioner.generate_change,
            "questioner.get_resp_log": questioner.get_resp_log,
            "questioner.log_question": questioner.log_question,
            "questioner.rv_guide": questioner.rv_guide,
            "questioner.rv_validation": questioner.rv_validation,
            "questioner.set_question_prefix": questioner.set_question_prefix,
            "CBT.get_resp_log": CBT.get_resp_log,
            "CBT.log_question": CBT.log_question,
            "CBT.set_question_prefix": CBT.set_question_prefix,
            "CBT.llm_complete_task": CBT.llm_complete_task,
            "CBT.recap_stage3_challenge": CBT.recap_stage3_challenge,
        }

        def fake_rv_complete_task(task, system_content, user_content, max_new_tokens=None):
            raw = next(rv_decisions)
            return GenerationResult(
                text=raw,
                task=task,
                adapter="adapters/task3_rv_reasoner",
                raw_text=raw,
            )

        def fake_cbt_complete_task(task, system_content, user_content, max_new_tokens=None):
            return GenerationResult(
                text="0",
                task=task,
                adapter=f"adapters/{task.value}",
                raw_text="0",
            )

        try:
            response_bridge.classify_dimension_and_score_result = (
                lambda _answer, _question: normalize_task1_output("1_weight, 2")
            )
            reflection_validation.llm_complete_task = fake_rv_complete_task
            questioner.generate_change = lambda text: text
            questioner.get_resp_log = lambda: next(rv_responses)
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.rv_guide = lambda *_args: "Guide: Please return to the weight change."
            questioner.rv_validation = lambda *_args: "VALIDATION: That connects to the weight change."
            questioner.set_question_prefix = lambda text: prefixes.append(text)

            dla_result = questioner.classify_segments(user_segments, question_text, "weight")
            valid, terminate, previous, updated = questioner.evaluate_result(
                question_lib,
                dla_result,
                1,
                "1",
                user_segments,
                question_text,
            )

            CBT.get_resp_log = lambda: next(cbt_responses)
            CBT.log_question = lambda text: logged_questions.append(text)
            CBT.set_question_prefix = lambda text: prefixes.append(text)
            CBT.llm_complete_task = fake_cbt_complete_task
            CBT.recap_stage3_challenge = lambda *_args: "You challenged the thought by naming another possibility."
            CBT.run_cbt(updated)
        finally:
            response_bridge.classify_dimension_and_score_result = originals[
                "response_bridge.classify_dimension_and_score_result"
            ]
            reflection_validation.llm_complete_task = originals[
                "reflection_validation.llm_complete_task"
            ]
            questioner.generate_change = originals["questioner.generate_change"]
            questioner.get_resp_log = originals["questioner.get_resp_log"]
            questioner.log_question = originals["questioner.log_question"]
            questioner.rv_guide = originals["questioner.rv_guide"]
            questioner.rv_validation = originals["questioner.rv_validation"]
            questioner.set_question_prefix = originals["questioner.set_question_prefix"]
            CBT.get_resp_log = originals["CBT.get_resp_log"]
            CBT.log_question = originals["CBT.log_question"]
            CBT.set_question_prefix = originals["CBT.set_question_prefix"]
            CBT.llm_complete_task = originals["CBT.llm_complete_task"]
            CBT.recap_stage3_challenge = originals["CBT.recap_stage3_challenge"]

        self.assertEqual(dla_result, [("weight", 2)])
        self.assertEqual(valid, 1)
        self.assertEqual(terminate, 0)
        self.assertIn("Can you tell me more", previous)
        self.assertEqual(question_lib["1"]["1"]["score"], [2])

        rv_note = question_lib["1"]["1"]["notes"][-2]
        self.assertIn("rv_decision: 0", rv_note)
        self.assertIn("rv_decision_raw: DECISION: 1 | DECISION: 0", rv_note)
        self.assertIn("followup_resp_1: I often do stress eating during deadlines.", rv_note)

        cbt_note = question_lib["1"]["1"]["notes"][-1]
        self.assertIn("CBT_dimension: weight", cbt_note)
        self.assertIn("CBT_stage: success", cbt_note)

        events = self._events()
        tasks = [row[0] for row in events]
        self.assertEqual(
            tasks,
            [
                "task1_response_analyzer",
                "reflective_summarizer",
                "task3_rv_reasoner",
                "task3_rv_reasoner",
                "task4_cbt_stage1",
                "task4_cbt_stage2",
                "task4_cbt_stage3",
            ],
        )
        self.assertEqual([row[2] for row in events], ["weight"] * 7)
        self.assertEqual([row[3] for row in events], ["2", "2", "1", "0", "0", "0", "0"])
        self.assertEqual(events[0][6], "weight, 2")
        self.assertEqual(events[-1][6], "DECISION: 0")


if __name__ == "__main__":
    unittest.main()
