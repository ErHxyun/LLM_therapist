import os
import sqlite3
import tempfile
import unittest

from src import CBT, reflection_validation
from src.local_llm.types import GenerationResult
from src.utils import response_bridge
from src.utils.llm_output_contracts import normalize_task1_output, normalize_task2_output


class StructuredLoggingHooksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "events.sqlite3")
        self.old_db_path = os.environ.get("CAITI_STRUCTURED_LOG_DB")
        self.old_session_id = os.environ.get("CAITI_SESSION_ID")
        os.environ["CAITI_STRUCTURED_LOG_DB"] = self.db_path
        os.environ["CAITI_SESSION_ID"] = "hook-test-session"

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
                SELECT task, adapter, dimension, score, segment_text,
                       raw_llm_output, normalized_output
                FROM session_events
                ORDER BY id
                """
            ).fetchall()
        finally:
            conn.close()

    def test_response_bridge_logs_task1_event(self):
        original = response_bridge.classify_dimension_and_score_result
        try:
            response_bridge.classify_dimension_and_score_result = (
                lambda _answer, _question: normalize_task1_output("1_weight, 2")
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "My weight increased a lot recently.",
                    "Have your weight changed significantly recently?",
                    "weight",
                ),
                ("weight", 2),
            )
        finally:
            response_bridge.classify_dimension_and_score_result = original

        rows = self._events()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "task1_response_analyzer")
        self.assertEqual(rows[0][1], "adapters/task1_response_analyzer")
        self.assertEqual(rows[0][2], "weight")
        self.assertEqual(rows[0][3], "2")
        self.assertEqual(rows[0][5], "1_weight, 2")
        self.assertEqual(rows[0][6], "weight, 2")

    def test_response_bridge_logs_task2_event(self):
        original = response_bridge.classify_general_response_result
        try:
            response_bridge.classify_general_response_result = (
                lambda _answer, default=None: normalize_task2_output("No", default=default)
            )
            self.assertEqual(response_bridge.classify_with_task2("Nope.", "alcohol"), ("alcohol", "No"))
        finally:
            response_bridge.classify_general_response_result = original

        rows = self._events()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "task2_general_response")
        self.assertEqual(rows[0][1], "adapters/task2_general_response")
        self.assertEqual(rows[0][2], "alcohol")
        self.assertEqual(rows[0][3], "No")
        self.assertEqual(rows[0][5], "No")
        self.assertEqual(rows[0][6], "alcohol, No")

    def test_rv_reasoner_logs_task3_event(self):
        original = reflection_validation.llm_complete_task

        def fake_complete_task(task, system_content, user_content, max_new_tokens=None):
            return GenerationResult(
                text="0",
                task=task,
                adapter="adapters/task3_rv_reasoner",
                raw_text="0",
            )

        try:
            reflection_validation.llm_complete_task = fake_complete_task
            self.assertEqual(
                reflection_validation.rv_reasoner(
                    "weight",
                    "Have your weight changed significantly recently?",
                    "My weight increased a lot recently.",
                    "I often do stress eating during deadlines.",
                ),
                "DECISION: 0",
            )
        finally:
            reflection_validation.llm_complete_task = original

        rows = self._events()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "task3_rv_reasoner")
        self.assertEqual(rows[0][1], "adapters/task3_rv_reasoner")
        self.assertEqual(rows[0][2], "weight")
        self.assertEqual(rows[0][3], "0")
        self.assertEqual(rows[0][5], "0")
        self.assertEqual(rows[0][6], "DECISION: 0")

    def test_cbt_reasoners_log_task4_events(self):
        original = CBT.llm_complete_task

        def fake_complete_task(task, system_content, user_content, max_new_tokens=None):
            return GenerationResult(
                text="0",
                task=task,
                adapter=f"adapters/{task.value}",
                raw_text="0",
            )

        try:
            CBT.llm_complete_task = fake_complete_task
            self.assertEqual(
                CBT.stage1_reasoner(
                    "I keep missing deadlines and feel like a failure.",
                    "If I miss one deadline, I am useless.",
                    dimension="productivity",
                ),
                "DECISION: 0",
            )
            self.assertEqual(
                CBT.stage2_reasoner(
                    "I keep missing deadlines and feel like a failure.",
                    "If I miss one deadline, I am useless.",
                    "One deadline does not mean I am useless.",
                    dimension="productivity",
                ),
                "DECISION: 0",
            )
            self.assertEqual(
                CBT.stage3_reasoner(
                    "I keep missing deadlines and feel like a failure.",
                    "If I miss one deadline, I am useless.",
                    "One deadline does not mean I am useless.",
                    "I can learn from this and plan the next task.",
                    dimension="productivity",
                ),
                "DECISION: 0",
            )
        finally:
            CBT.llm_complete_task = original

        rows = self._events()
        self.assertEqual([row[0] for row in rows], [
            "task4_cbt_stage1",
            "task4_cbt_stage2",
            "task4_cbt_stage3",
        ])
        self.assertEqual([row[2] for row in rows], ["productivity", "productivity", "productivity"])
        self.assertEqual([row[3] for row in rows], ["0", "0", "0"])
        self.assertEqual([row[6] for row in rows], ["DECISION: 0", "DECISION: 0", "DECISION: 0"])


if __name__ == "__main__":
    unittest.main()
