import csv
import os
import tempfile
import unittest

from src.utils.io_question_lib import generate_results


class SessionReportFormatTest(unittest.TestCase):
    def test_generate_results_uses_clinical_note_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_file = os.path.join(tmp, "report.csv")
            notes_file = os.path.join(tmp, "notes.csv")
            question_lib = {
                "1": {
                    "1": {
                        "label": "weight",
                        "score": [2],
                        "notes": [
                            [
                                "original_question: Have your weight changed significantly recently?",
                                "original_resp: My weight increased a lot recently.",
                            ],
                            [
                                "followup_resp: I like painting.",
                                "followup_resp_1: I often do stress eating during deadlines.",
                                "rv_decision: 0",
                                "rv_validation: VALIDATION: Thank you for sharing more.",
                            ],
                            [
                                "CBT_unhelpful_thoughts: If I gain weight, I am failing.",
                                "CBT_challenge: Deadlines affect my eating, but gaining weight does not mean I am failing.",
                                "CBT_reframe: I can plan regular meals during deadlines.",
                                "CBT_stage: success",
                            ],
                        ],
                    }
                }
            }

            generate_results(question_lib, [], report_file=report_file, notes_file=notes_file)

            with open(report_file, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(rows[0], ["Score", "Responses", "Analysis"])
        self.assertEqual(rows[1][0], "2")
        self.assertNotIn("rv_decision", rows[1][0])
        self.assertNotIn("CBT_stage", rows[1][0])
        self.assertIn("My weight increased a lot recently.", rows[1][1])
        self.assertIn("I often do stress eating during deadlines.", rows[1][1])
        self.assertNotIn("rv_validation", rows[1][1])
        self.assertNotIn("CBT_stage", rows[1][1])
        self.assertIn("Dimension: weight", rows[1][2])
        self.assertIn("rv_decision: 0", rows[1][2])
        self.assertIn("rv_validation: VALIDATION: Thank you for sharing more.", rows[1][2])
        self.assertIn("CBT_stage: success", rows[1][2])


if __name__ == "__main__":
    unittest.main()
