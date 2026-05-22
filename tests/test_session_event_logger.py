import json
import os
import sqlite3
import tempfile
import unittest

from src.local_llm.types import LLMTask
from src.utils.session_event_logger import log_llm_event


class SessionEventLoggerTest(unittest.TestCase):
    def test_log_llm_event_writes_sqlite_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "events.sqlite3")
            log_llm_event(
                task=LLMTask.TASK1_RESPONSE_ANALYZER,
                dimension="weight",
                score=2,
                segment_text="My weight increased a lot recently.",
                question_text="Have your weight changed significantly recently?",
                raw_llm_output="1_weight, 2",
                normalized_output="weight, 2",
                metadata={"source": "unit_test"},
                session_id="test-session",
                db_path=db_path,
            )

            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    """
                    SELECT
                        session_id,
                        task,
                        adapter,
                        dimension,
                        score,
                        segment_text,
                        raw_llm_output,
                        normalized_output,
                        metadata_json
                    FROM session_events
                    """
                ).fetchone()
            finally:
                conn.close()

        self.assertEqual(row[0], "test-session")
        self.assertEqual(row[1], "task1_response_analyzer")
        self.assertEqual(row[2], "adapters/task1_response_analyzer")
        self.assertEqual(row[3], "weight")
        self.assertEqual(row[4], "2")
        self.assertEqual(row[5], "My weight increased a lot recently.")
        self.assertEqual(row[6], "1_weight, 2")
        self.assertEqual(row[7], "weight, 2")
        self.assertEqual(json.loads(row[8]), {"source": "unit_test"})


if __name__ == "__main__":
    unittest.main()
