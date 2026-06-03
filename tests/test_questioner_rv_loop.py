import unittest

from src.emotion import clear_emotion_results_for_tests, register_emotion_result
from src.emotion.followup import EmotionFollowupDecision, EmotionFollowupSettings
from src import questioner


class QuestionerRVLoopTest(unittest.TestCase):
    def setUp(self):
        clear_emotion_results_for_tests()
        self._original_build_emotion_followup_settings = questioner.build_emotion_followup_settings
        questioner.build_emotion_followup_settings = lambda: EmotionFollowupSettings(False, 0.0, 50, 60, 45)

    def tearDown(self):
        questioner.build_emotion_followup_settings = self._original_build_emotion_followup_settings
        clear_emotion_results_for_tests()

    def test_ask_question_uses_library_question_without_runtime_rewrite(self):
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
        logged_questions = []

        originals = {
            "generate_synonymous_sentences": questioner.generate_synonymous_sentences,
            "get_answer": questioner.get_answer,
            "log_question": questioner.log_question,
            "classify_segments": questioner.classify_segments,
        }

        try:
            questioner.generate_synonymous_sentences = lambda _text: (_ for _ in ()).throw(
                AssertionError("rewrite should not be called")
            )
            questioner.get_answer = lambda: ("", ["My weight has been stable."])
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.classify_segments = lambda *_args: [("weight", 0)]

            reward, terminate, previous = questioner.ask_question(question_lib, 1)
        finally:
            for name, value in originals.items():
                setattr(questioner, name, value)

        self.assertEqual(reward, 0.0)
        self.assertEqual(terminate, 0)
        self.assertEqual(previous, "")
        self.assertEqual(logged_questions, ["Have your weight changed significantly recently?"])

    def test_ask_question_randomly_selects_one_library_variant(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "weight",
                    "question": [
                        "Can you tell me about recent weight changes, and has your weight changed recently?",
                        "What have you noticed about your weight, and has it changed more than usual?",
                        "How has your weight been lately, and has it changed significantly?",
                    ],
                    "score": [],
                    "notes": [],
                }
            }
        }
        logged_questions = []

        originals = {
            "get_answer": questioner.get_answer,
            "log_question": questioner.log_question,
            "classify_segments": questioner.classify_segments,
            "randint": questioner.np.random.randint,
        }

        try:
            questioner.get_answer = lambda: ("", ["My weight has been stable."])
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.classify_segments = lambda *_args: [("weight", 0)]
            questioner.np.random.randint = lambda _upper_bound: 2

            reward, terminate, previous = questioner.ask_question(question_lib, 1)
        finally:
            questioner.np.random.randint = originals["randint"]
            for name, value in originals.items():
                if name != "randint":
                    setattr(questioner, name, value)

        self.assertEqual(reward, 0.0)
        self.assertEqual(terminate, 0)
        self.assertEqual(previous, "")
        self.assertEqual(logged_questions, ["How has your weight been lately, and has it changed significantly?"])

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

    def test_emotion_followup_does_not_change_score_zero(self):
        questioner.build_emotion_followup_settings = lambda: EmotionFollowupSettings(True, 0.0, 50, 60, 45)
        register_emotion_result(
            {
                "status": "ok",
                "request": {"transcript": "I'm fine."},
                "response": {
                    "audio_emotion": [0, -1],
                    "context_emotion": [0, 1],
                    "audio_scores": {"tension": 75, "hesitation": 30, "stability": 45},
                    "emotion_comparison": {
                        "audio_vs_context_consistent": False,
                        "valence_conflict": True,
                    },
                    "final_assessment": {
                        "confidence": 85,
                        "credibility_risk": 70,
                        "quality_flags": [],
                    },
                    "final_result": {"credibility_risk": 70, "risk_level": "Moderate"},
                },
            }
        )
        question_lib = {
            "1": {
                "1": {
                    "label": "mood",
                    "question": ["How has your mood been lately?"],
                    "score": [],
                    "notes": [],
                }
            }
        }

        valid, terminate, followup, updated = questioner._if_valid_response(
            [("mood", 0)],
            1,
            "1",
            ["I'm fine."],
            "How has your mood been lately?",
            question_lib,
        )

        self.assertEqual(valid, 1)
        self.assertEqual(terminate, 0)
        self.assertEqual(updated["1"]["1"]["score"], [0])
        self.assertIn("gently check", followup)
        self.assertIn("emotion_followup_reason: emotion_distress_hidden_by_low_content_score", updated["1"]["1"]["notes"][-1])

    def test_score_two_positive_voice_uses_emotion_confirmation_followup(self):
        questioner.build_emotion_followup_settings = lambda: EmotionFollowupSettings(True, 0.0, 50, 60, 45)
        register_emotion_result(
            {
                "status": "ok",
                "request": {"transcript": "I barely sleep and feel exhausted."},
                "response": {
                    "audio_emotion": [0, 1],
                    "context_emotion": [0, -1],
                    "audio_scores": {"tension": 20, "hesitation": 20, "stability": 85},
                    "emotion_comparison": {"audio_vs_context_consistent": True},
                    "final_assessment": {
                        "confidence": 80,
                        "credibility_risk": 20,
                        "quality_flags": [],
                    },
                    "final_result": {"credibility_risk": 20, "risk_level": "Low"},
                },
            }
        )
        question_lib = {
            "1": {
                "1": {
                    "label": "sleep",
                    "question": ["How have you been sleeping lately?"],
                    "score": [],
                    "notes": [],
                }
            }
        }

        valid, terminate, followup, updated = questioner._if_valid_response(
            [("sleep", 2)],
            1,
            "1",
            ["I barely sleep and feel exhausted."],
            "How have you been sleeping lately?",
            question_lib,
        )

        self.assertEqual(valid, 1)
        self.assertEqual(terminate, 0)
        self.assertEqual(updated["1"]["1"]["score"], [2])
        self.assertIn("lighter tone", followup)

    def test_late_emotion_followup_is_appended_before_next_question(self):
        appended = []
        logged = []
        decision = EmotionFollowupDecision(
            True,
            "emotion_distress_acoustic_cue",
            "You said things are okay, and I just want to gently check in.",
            {
                "score": 0,
                "dimension": "mood",
                "user_text": "I'm fine.",
                "question_text": "How has your mood been lately?",
            },
        )

        originals = {
            "pop_late_emotion_followup": questioner.pop_late_emotion_followup,
            "append_question_prefix": questioner.append_question_prefix,
            "log_llm_event": questioner.log_llm_event,
        }

        try:
            questioner.pop_late_emotion_followup = lambda: decision
            questioner.append_question_prefix = lambda text: appended.append(text)
            questioner.log_llm_event = lambda **kwargs: logged.append(kwargs)

            questioner._apply_late_emotion_followup_prefix()
        finally:
            for name, value in originals.items():
                setattr(questioner, name, value)

        self.assertEqual(appended, ["You said things are okay, and I just want to gently check in."])
        self.assertEqual(logged[0]["task"], "emotion_late_followup")
        self.assertEqual(logged[0]["dimension"], "mood")

    def test_stop_token_only_terminates_for_explicit_stop_request(self):
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
            [("weight", "Stop")],
            1,
            "1",
            ["I stopped eating snacks recently."],
            "Have your weight changed significantly recently?",
            question_lib,
        )

        self.assertEqual(valid, 0)
        self.assertEqual(terminate, 0)
        self.assertEqual(followup, "")
        self.assertEqual(updated["1"]["1"]["score"], [])

        valid, terminate, followup, updated = questioner._if_valid_response(
            [("weight", "Stop")],
            1,
            "1",
            ["I want to stop."],
            "Have your weight changed significantly recently?",
            question_lib,
        )

        self.assertEqual(valid, 1)
        self.assertEqual(terminate, 1)
        self.assertEqual(followup, "")

    def test_rv_loop_continues_guiding_until_followup_is_valid(self):
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
            "Still painting.",
            "I have upcoming deadlines, so I often do stress eating.",
        ])
        decisions = iter(["DECISION: 1", "DECISION: 1", "DECISION: 0"])
        logged_questions = []
        prefixes = []

        originals = {
            "get_resp_log": questioner.get_resp_log,
            "log_question": questioner.log_question,
            "rv_reasoner": questioner.rv_reasoner,
            "rv_guide": questioner.rv_guide,
            "rv_validation": questioner.rv_validation,
            "set_question_prefix": questioner.set_question_prefix,
        }

        try:
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
        self.assertEqual(logged_questions.count("Guide: Please return to the weight change."), 2)
        self.assertEqual(prefixes, ["VALIDATION: That connects to the weight change."])

        rv_note = updated["1"]["1"]["notes"][-1]
        self.assertIn("rv_decision: 0", rv_note)
        self.assertIn("rv_decision_raw: DECISION: 1 | DECISION: 1 | DECISION: 0", rv_note)
        self.assertIn(
            "followup_resp_1: I have upcoming deadlines, so I often do stress eating.",
            rv_note,
        )
        self.assertIn("rv_validation: VALIDATION: That connects to the weight change.", rv_note)


if __name__ == "__main__":
    unittest.main()
