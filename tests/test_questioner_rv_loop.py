import unittest

from src import questioner


class QuestionerRVLoopTest(unittest.TestCase):
    def test_invalid_followup_routes_through_guide_before_validation(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "weight",
                    "question": ["Have your weight changed significantly recently?"],
                    "score": [],
                    "notes": [],
                }
            }
        }
        responses = iter([
            "I like painting.",
            "I have upcoming deadlines, so I often do stress eating.",
        ])
        decisions = iter(["DECISION: 1", "DECISION: 0"])
        logged_questions = []
        prefixes = []

        originals = {
            "generate_change": questioner.generate_change,
            "get_resp_log": questioner.get_resp_log,
            "log_question": questioner.log_question,
            "rv_reasoner": questioner.rv_reasoner,
            "rv_guide": questioner.rv_guide,
            "rv_validation": questioner.rv_validation,
            "set_question_prefix": questioner.set_question_prefix,
        }

        try:
            questioner.generate_change = lambda text: text
            questioner.get_resp_log = lambda: next(responses)
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.rv_reasoner = lambda *_args: next(decisions)
            questioner.rv_guide = lambda *_args: "Guide: Please return to the weight change."
            questioner.rv_validation = lambda *_args: "VALIDATION: That connects to the weight change."
            questioner.set_question_prefix = lambda text: prefixes.append(text)

            valid, terminate, previous, updated = questioner.evaluate_result(
                question_lib,
                [("weight", 2)],
                1,
                "1",
                ["My weight increased a lot recently."],
                "Have your weight changed significantly recently?",
            )
        finally:
            for name, value in originals.items():
                setattr(questioner, name, value)

        self.assertEqual(valid, 1)
        self.assertEqual(terminate, 0)
        self.assertIn("Can you tell me more", previous)
        self.assertEqual(question_lib["1"]["1"]["score"], [2])
        self.assertEqual(len(logged_questions), 2)
        self.assertEqual(logged_questions[1], "Guide: Please return to the weight change.")
        self.assertEqual(prefixes, ["VALIDATION: That connects to the weight change."])

        rv_note = updated["1"]["1"]["notes"][-1]
        self.assertIn("rv_decision: 0", rv_note)
        self.assertIn("rv_decision_raw: DECISION: 1 | DECISION: 0", rv_note)
        self.assertIn("rv_guide: Guide: Please return to the weight change.", rv_note)
        self.assertIn("followup_resp: I like painting.", rv_note)
        self.assertIn(
            "followup_resp_1: I have upcoming deadlines, so I often do stress eating.",
            rv_note,
        )
        self.assertIn("rv_validation: VALIDATION: That connects to the weight change.", rv_note)

    def test_score_two_reflective_followup_uses_original_response_without_llm_rewrite(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "weight",
                    "question": ["Have your weight changed significantly recently?"],
                    "score": [],
                    "notes": [],
                }
            }
        }

        valid, terminate, followup, updated = questioner._if_valid_response(
            [("weight", 2)],
            1,
            "1",
            ["My weight increased a lot recently."],
            "Have your weight changed significantly recently?",
            question_lib,
        )

        self.assertEqual(valid, 1)
        self.assertEqual(terminate, 0)
        self.assertEqual(
            followup,
            'You mentioned "My weight increased a lot recently." Can you tell me more about that?',
        )
        self.assertEqual(updated["1"]["1"]["score"], [2])


if __name__ == "__main__":
    unittest.main()
