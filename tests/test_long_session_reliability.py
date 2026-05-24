import unittest
import json
import tempfile
from pathlib import Path

from scripts import long_session_reliability as reliability


class LongSessionReliabilityScriptTest(unittest.TestCase):
    def test_scripted_user_responses_cover_flow_prompts(self):
        user = reliability.ScriptedUser(record_path=None, stop_event=None)

        self.assertEqual(user.response_for("Which dimension would you like to work on today?"), "1")
        self.assertIn("failing", user.response_for("Can you try to identify any unhelpful thoughts?").lower())
        self.assertIn("difficult week", user.response_for("Now, how could you challenge those thoughts?"))
        self.assertIn("problem to work on", user.response_for("Finally, can you reframe the thought?"))
        self.assertIn("deadlines", user.response_for('You mentioned "x". Can you tell me more about that?'))

    def test_analyze_question_lib_counts_scores_and_cbt_success(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "weight",
                    "score": [2],
                    "notes": [["original_resp: hard", "CBT_stage: success"]],
                }
            },
            "2": {
                "1": {
                    "label": "work",
                    "score": [0],
                    "notes": [["original_resp: no issue"]],
                }
            },
        }

        summary = reliability.analyze_question_lib(question_lib)

        self.assertEqual(summary["dimension_count"], 2)
        self.assertEqual(summary["score_count"], 2)
        self.assertEqual(summary["score2_count"], 1)
        self.assertTrue(summary["cbt_success"])
        self.assertEqual(summary["dimensions_with_scores"], ["weight", "work"])

    def test_read_event_counts_returns_empty_for_missing_db(self):
        self.assertEqual(reliability.read_event_counts(__import__("pathlib").Path("/tmp/missing-caiti-events.sqlite3")), {})

    def test_reset_question_lib_file_clears_scores_and_notes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "question_lib.json"
            path.write_text(
                json.dumps(
                    {
                        "1": {
                            "1": {
                                "label": "weight",
                                "score": [2],
                                "notes": [["old"]],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            reliability.reset_question_lib_file(path)
            result = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(result["1"]["1"]["score"], [])
        self.assertEqual(result["1"]["1"]["notes"], [])


if __name__ == "__main__":
    unittest.main()
