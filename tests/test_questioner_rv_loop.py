import unittest

from src.emotion import clear_emotion_results_for_tests, register_emotion_result
from src.emotion.followup import EmotionFollowupDecision, EmotionFollowupSettings
from src import questioner


class QuestionerRVLoopTest(unittest.TestCase):
    def setUp(self):
        clear_emotion_results_for_tests()
        self._original_build_emotion_followup_settings = questioner.build_emotion_followup_settings
        questioner.build_emotion_followup_settings = lambda: EmotionFollowupSettings(False, 0.0, 50, 60, 45)
        questioner._PENDING_NEXT_QUESTION_INTRO = ""
        questioner._PENDING_VALIDATION_TEXT = ""

    def tearDown(self):
        questioner.build_emotion_followup_settings = self._original_build_emotion_followup_settings
        questioner._PENDING_NEXT_QUESTION_INTRO = ""
        questioner._PENDING_VALIDATION_TEXT = ""
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

            result = questioner.ask_question(question_lib, 1)
        finally:
            for name, value in originals.items():
                setattr(questioner, name, value)

        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.terminate, 0)
        self.assertEqual(result.previous_question, "")
        self.assertEqual(result.covered_item_ids, {1})
        self.assertTrue(result.current_answered)
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

            result = questioner.ask_question(question_lib, 1)
        finally:
            questioner.np.random.randint = originals["randint"]
            for name, value in originals.items():
                if name != "randint":
                    setattr(questioner, name, value)

        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.terminate, 0)
        self.assertEqual(result.previous_question, "")
        self.assertEqual(result.covered_item_ids, {1})
        self.assertTrue(result.current_answered)
        self.assertEqual(logged_questions, ["How has your weight been lately, and has it changed significantly?"])


    def test_ask_question_reports_current_and_cross_covered_dimensions(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "sleep",
                    "question": ["Have you been sleeping enough recently?"],
                    "score": [],
                    "notes": [],
                }
            },
            "2": {
                "1": {
                    "label": "eat",
                    "question": ["Have you been eating at regular times?"],
                    "score": [],
                    "notes": [],
                }
            },
        }
        originals = {
            "get_answer": questioner.get_answer,
            "log_question": questioner.log_question,
            "classify_segments": questioner.classify_segments,
        }

        try:
            questioner.get_answer = lambda: (
                "",
                [
                    "I have been sleeping well.",
                    "I have been eating at regular times.",
                ],
            )
            questioner.log_question = lambda _text: None
            questioner.classify_segments = lambda *_args: [
                ("sleep", 0),
                ("eat", 0),
            ]

            result = questioner.ask_question(question_lib, 1)
        finally:
            for name, value in originals.items():
                setattr(questioner, name, value)

        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.terminate, 0)
        self.assertEqual(result.covered_item_ids, {1, 2})
        self.assertTrue(result.current_answered)
        self.assertEqual(question_lib["1"]["1"]["score"], [0])
        self.assertEqual(question_lib["2"]["1"]["score"], [0])
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

        originals = {
            "generate_change": questioner.generate_change,
            "get_resp_log": questioner.get_resp_log,
            "log_question": questioner.log_question,
            "rv_reasoner": questioner.rv_reasoner,
            "rv_guide": questioner.rv_guide,
            "rv_validation": questioner.rv_validation,
        }

        try:
            questioner.generate_change = lambda text: text
            questioner.get_resp_log = lambda: next(responses)
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.rv_reasoner = lambda *_args: next(decisions)
            questioner.rv_guide = lambda *_args: "Guide: Please return to the weight change."
            questioner.rv_validation = lambda *_args: "VALIDATION: That connects to the weight change."

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
        self.assertEqual(
            questioner._pop_pending_next_question_intro(),
            "Thank you for your clarification. Let's continue our questions.",
        )
        self.assertEqual(
            questioner._pop_pending_validation_text(),
            "VALIDATION: That connects to the weight change.",
        )

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

    def test_score_two_remains_provisional_until_rv_runs(self):
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

        outcome = questioner._if_valid_response(
            [("weight", 2)],
            1,
            "1",
            ["My weight increased a lot recently."],
            "Have your weight changed significantly recently?",
            question_lib,
        )

        self.assertFalse(outcome.current_answered)
        self.assertFalse(outcome.terminate)
        self.assertEqual(outcome.current_followup, "")
        self.assertEqual(outcome.covered_item_ids, set())
        self.assertEqual([item.item_id for item in outcome.score2_queue], [1])
        self.assertEqual(question_lib["1"]["1"]["score"], [])

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

        outcome = questioner._if_valid_response(
            [("mood", 0)],
            1,
            "1",
            ["I'm fine."],
            "How has your mood been lately?",
            question_lib,
        )

        self.assertTrue(outcome.current_answered)
        self.assertFalse(outcome.terminate)
        self.assertEqual(question_lib["1"]["1"]["score"], [0])
        self.assertIn("gently check", outcome.current_followup)
        self.assertIn("emotion_followup_reason: emotion_distress_hidden_by_low_content_score", question_lib["1"]["1"]["notes"][-1])

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

        logged_questions = []
        originals = {
            "get_resp_log": questioner.get_resp_log,
            "log_question": questioner.log_question,
            "rv_reasoner": questioner.rv_reasoner,
            "rv_validation": questioner.rv_validation,
        }
        try:
            questioner.get_resp_log = lambda: "My sleep really has been difficult."
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.rv_reasoner = lambda *_args: "DECISION: 0"
            questioner.rv_validation = lambda *_args: "VALIDATION: Thank you for clarifying."

            valid, terminate, _previous, _updated = questioner.evaluate_result(
                question_lib,
                [("sleep", 2)],
                1,
                "1",
                ["I barely sleep and feel exhausted."],
                "How have you been sleeping lately?",
            )
        finally:
            for name, value in originals.items():
                setattr(questioner, name, value)

        self.assertEqual(valid, 1)
        self.assertEqual(terminate, 0)
        self.assertEqual(question_lib["1"]["1"]["score"], [2])
        self.assertIn("lighter tone", logged_questions[0])

    def test_late_emotion_followup_runs_as_combined_validation_followup_before_next_question(self):
        asked = []
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
            "log_question": questioner.log_question,
            "_get_resp_log_with_control": questioner._get_resp_log_with_control,
            "log_llm_event": questioner.log_llm_event,
        }

        try:
            questioner.pop_late_emotion_followup = lambda: decision
            questioner.log_question = lambda text: asked.append(text)
            questioner._get_resp_log_with_control = lambda _session_control=None: "It was exaggeration."
            questioner.log_llm_event = lambda **kwargs: logged.append(kwargs)
            questioner._set_pending_validation_text(
                "VALIDATION: That connects to the mood changes you just described."
            )
            questioner._set_pending_next_question_intro(
                "Thank you for your clarification. Let's continue our questions."
            )

            should_stop = questioner._run_late_emotion_followup_before_question()
        finally:
            for name, value in originals.items():
                setattr(questioner, name, value)

        self.assertFalse(should_stop)
        self.assertEqual(
            asked,
            [
                "That connects to the mood changes you just described. You said things are okay, and I just want to gently check in."
            ],
        )
        self.assertEqual(
            questioner._pop_pending_next_question_intro(),
            "Thank you for your clarification. Let's continue our questions.",
        )
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

        non_stop_outcome = questioner._if_valid_response(
            [("weight", "Stop")],
            1,
            "1",
            ["I stopped eating snacks recently."],
            "Have your weight changed significantly recently?",
            question_lib,
        )

        self.assertFalse(non_stop_outcome.current_answered)
        self.assertFalse(non_stop_outcome.terminate)
        self.assertEqual(non_stop_outcome.current_followup, "")
        self.assertEqual(question_lib["1"]["1"]["score"], [])

        stop_outcome = questioner._if_valid_response(
            [("weight", "Stop")],
            1,
            "1",
            ["I want to stop."],
            "Have your weight changed significantly recently?",
            question_lib,
        )

        self.assertFalse(stop_outcome.current_answered)
        self.assertTrue(stop_outcome.terminate)
        self.assertEqual(stop_outcome.current_followup, "")

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

        originals = {
            "get_resp_log": questioner.get_resp_log,
            "log_question": questioner.log_question,
            "rv_reasoner": questioner.rv_reasoner,
            "rv_guide": questioner.rv_guide,
            "rv_validation": questioner.rv_validation,
        }

        try:
            questioner.get_resp_log = lambda: next(responses)
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.rv_reasoner = lambda *_args: next(decisions)
            questioner.rv_guide = lambda *_args: "Guide: Please return to the weight change."
            questioner.rv_validation = lambda *_args: "VALIDATION: That connects to the weight change."

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
        self.assertEqual(
            questioner._pop_pending_next_question_intro(),
            "Thank you for your clarification. Let's continue our questions.",
        )
        self.assertEqual(
            questioner._pop_pending_validation_text(),
            "VALIDATION: That connects to the weight change.",
        )

        rv_note = updated["1"]["1"]["notes"][-1]
        self.assertIn("rv_decision: 0", rv_note)
        self.assertIn("rv_decision_raw: DECISION: 1 | DECISION: 1 | DECISION: 0", rv_note)
        self.assertIn(
            "followup_resp_1: I have upcoming deadlines, so I often do stress eating.",
            rv_note,
        )
        self.assertIn("rv_validation: VALIDATION: That connects to the weight change.", rv_note)
        self.assertIn("rv_retry_count: 2", rv_note)
        self.assertIn("rv_retry_exhausted: false", rv_note)

    def test_rv_loop_stops_after_retry_limit_and_moves_on(self):
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
            "Nothing else to add.",
        ])
        logged_questions = []

        originals = {
            "get_resp_log": questioner.get_resp_log,
            "log_question": questioner.log_question,
            "rv_reasoner": questioner.rv_reasoner,
            "rv_guide": questioner.rv_guide,
            "rv_validation": questioner.rv_validation,
        }

        try:
            questioner.get_resp_log = lambda: next(responses)
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.rv_reasoner = lambda *_args: "DECISION: 1"
            questioner.rv_guide = lambda *_args: "Guide: Please return to the weight change."
            questioner.rv_validation = lambda *_args: "VALIDATION: That connects to the weight change."

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

        self.assertEqual(valid, 0)
        self.assertEqual(terminate, 0)
        self.assertIn("Can you tell me more", previous)
        self.assertEqual(logged_questions.count("Guide: Please return to the weight change."), 2)
        self.assertEqual(questioner._pop_pending_next_question_intro(), "")
        self.assertEqual(questioner._pop_pending_validation_text(), "")
        self.assertEqual(question_lib["1"]["1"]["score"], [])

        rv_note = updated["1"]["1"]["notes"][-1]
        self.assertIn("rv_decision: 1", rv_note)
        self.assertIn("rv_retry_count: 2", rv_note)
        self.assertIn("rv_retry_exhausted: true", rv_note)
        self.assertIn("followup_resp_1: Nothing else to add.", rv_note)
        self.assertIn("rv_validation: ", rv_note)
        self.assertIn("rv_completed: false", rv_note)
        self.assertIn("score_committed: false", rv_note)

    def test_score_two_stop_during_rv_remains_uncommitted(self):
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
            "get_resp_log": questioner.get_resp_log,
            "log_question": questioner.log_question,
            "rv_reasoner": questioner.rv_reasoner,
            "rv_validation": questioner.rv_validation,
        }

        try:
            questioner.get_resp_log = lambda: "I want to stop."
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.rv_reasoner = lambda *_args: (_ for _ in ()).throw(
                AssertionError("reasoner must not run after a stop request")
            )
            questioner.rv_validation = lambda *_args: (_ for _ in ()).throw(
                AssertionError("validator must not run after a stop request")
            )

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

        self.assertEqual(valid, 0)
        self.assertEqual(terminate, 1)
        self.assertIn("Can you tell me more", previous)
        self.assertEqual(len(logged_questions), 1)
        self.assertEqual(question_lib["1"]["1"]["score"], [])

        rv_note = updated["1"]["1"]["notes"][-1]
        self.assertIn("rv_completed: false", rv_note)
        self.assertIn("score_committed: false", rv_note)

    def test_next_question_uses_pending_validation_and_intro_in_same_prompt(self):
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
            "get_answer": questioner.get_answer,
            "log_question": questioner.log_question,
            "classify_segments": questioner.classify_segments,
        }

        try:
            questioner._set_pending_validation_text(
                "VALIDATION: That connects to what you just described."
            )
            questioner._set_pending_next_question_intro(
                "Thank you for your clarification. Let's continue our questions."
            )
            questioner.get_answer = lambda: ("", ["My weight has been stable."])
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.classify_segments = lambda *_args: [("weight", 0)]

            result = questioner.ask_question(question_lib, 1)
        finally:
            for name, value in originals.items():
                setattr(questioner, name, value)

        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.terminate, 0)
        self.assertEqual(result.previous_question, "")
        self.assertEqual(result.covered_item_ids, {1})
        self.assertTrue(result.current_answered)
        self.assertEqual(
            logged_questions,
            [
                "That connects to what you just described. Thank you for your clarification. "
                "Let's continue our questions. Have your weight changed significantly recently?"
            ],
        )

    def test_multi_segment_response_scores_current_and_cross_dimension(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "sleep",
                    "question": ["Have you been sleeping enough recently?"],
                    "score": [],
                    "notes": [],
                }
            },
            "2": {
                "1": {
                    "label": "eat",
                    "question": ["Have you been eating at regular times?"],
                    "score": [],
                    "notes": [],
                }
            },
        }
        followup_calls = []
        original = questioner._build_followup_for_score

        def fake_followup(**kwargs):
            followup_calls.append(
                (kwargs["dimension"], kwargs["score"], kwargs["answer_text"])
            )
            return "Could you tell me more about your sleep?", None

        try:
            questioner._build_followup_for_score = fake_followup
            outcome = questioner._if_valid_response(
                [("sleep", 2), ("eat", 0)],
                1,
                "1",
                ["I haven't slept well", "I've been eating regularly"],
                "Have you been sleeping enough recently?",
                question_lib,
            )
        finally:
            questioner._build_followup_for_score = original

        self.assertFalse(outcome.current_answered)
        self.assertFalse(outcome.terminate)
        self.assertEqual(outcome.current_followup, "")
        self.assertEqual(outcome.covered_item_ids, {2})
        self.assertEqual(len(outcome.assessments), 2)
        self.assertEqual([item.item_id for item in outcome.score2_queue], [1])
        self.assertEqual(outcome.unresolved_segments, [])
        self.assertEqual(question_lib["1"]["1"]["score"], [])
        self.assertEqual(question_lib["2"]["1"]["score"], [0])
        self.assertEqual(followup_calls, [])
        self.assertIn("cross_dimension: true", question_lib["2"]["1"]["notes"][0])

    def test_cross_dimension_only_keeps_score_but_retries_current_dimension(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "sleep",
                    "question": ["Have you been sleeping enough recently?"],
                    "score": [],
                    "notes": [],
                }
            },
            "2": {
                "1": {
                    "label": "eat",
                    "question": ["Have you been eating at regular times?"],
                    "score": [],
                    "notes": [],
                }
            },
        }

        outcome = questioner._if_valid_response(
            [("eat", 0)],
            1,
            "1",
            ["I've been eating regularly"],
            "Have you been sleeping enough recently?",
            question_lib,
        )

        self.assertFalse(outcome.current_answered)
        self.assertFalse(outcome.terminate)
        self.assertEqual(outcome.current_followup, "")
        self.assertEqual(outcome.covered_item_ids, {2})
        self.assertEqual(len(outcome.assessments), 1)
        self.assertEqual(outcome.unresolved_segments, [])
        self.assertEqual(question_lib["1"]["1"]["score"], [])
        self.assertEqual(question_lib["2"]["1"]["score"], [0])


    def test_score_two_queue_deduplicates_dimensions_and_prioritizes_current(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "sleep",
                    "question": ["Have you been sleeping enough recently?"],
                    "score": [],
                    "notes": [],
                }
            },
            "2": {
                "1": {
                    "label": "eat",
                    "question": ["Have you been eating at regular times?"],
                    "score": [],
                    "notes": [],
                }
            },
        }
        followup_calls = []
        original = questioner._build_followup_for_score

        def fake_followup(**kwargs):
            followup_calls.append(
                (kwargs["dimension"], kwargs["score"], kwargs["answer_text"])
            )
            return "Could you tell me more about your sleep?", None

        try:
            questioner._build_followup_for_score = fake_followup
            outcome = questioner._if_valid_response(
                [("eat", 2), ("sleep", 2), ("sleep", 2)],
                1,
                "1",
                [
                    "I have not been eating regularly",
                    "I barely slept last night",
                    "My sleep has been poor all week",
                ],
                "Have you been sleeping enough recently?",
                question_lib,
            )
        finally:
            questioner._build_followup_for_score = original

        self.assertFalse(outcome.current_answered)
        self.assertEqual(len(outcome.assessments), 3)
        self.assertEqual(
            [item.item_id for item in outcome.assessments],
            [2, 1, 1],
        )
        self.assertEqual(outcome.covered_item_ids, set())
        self.assertEqual(
            [item.item_id for item in outcome.score2_queue],
            [1, 2],
        )
        self.assertEqual(question_lib["1"]["1"]["score"], [])
        self.assertEqual(question_lib["2"]["1"]["score"], [])
        self.assertEqual(followup_calls, [])


    def test_multiple_score_two_items_run_rv_current_first_and_commit_sequentially(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "sleep",
                    "question": ["Have you been sleeping enough recently?"],
                    "score": [],
                    "notes": [],
                }
            },
            "2": {
                "1": {
                    "label": "eat",
                    "question": ["Have you been eating at regular times?"],
                    "score": [],
                    "notes": [],
                }
            },
        }
        responses = iter([
            "My sleep has been poor all week.",
            "My eating schedule has also been irregular.",
        ])
        logged_questions = []
        reasoner_snapshots = []
        followup_calls = []
        originals = {
            "_build_followup_for_score": questioner._build_followup_for_score,
            "get_resp_log": questioner.get_resp_log,
            "log_question": questioner.log_question,
            "rv_reasoner": questioner.rv_reasoner,
            "rv_validation": questioner.rv_validation,
        }

        def fake_followup(**kwargs):
            followup_calls.append(kwargs["dimension"])
            return f"FOLLOWUP {kwargs['dimension']}", None

        def fake_reasoner(topic, *_args):
            reasoner_snapshots.append(
                (
                    topic,
                    list(question_lib["1"]["1"]["score"]),
                    list(question_lib["2"]["1"]["score"]),
                )
            )
            return "DECISION: 0"

        try:
            questioner._build_followup_for_score = fake_followup
            questioner.get_resp_log = lambda: next(responses)
            questioner.log_question = lambda text: logged_questions.append(text)
            questioner.rv_reasoner = fake_reasoner
            questioner.rv_validation = (
                lambda topic, *_args: f"VALIDATION: {topic} accepted."
            )

            valid, terminate, previous, updated = questioner.evaluate_result(
                question_lib,
                [("eat", 2), ("sleep", 2), ("sleep", 2)],
                1,
                "1",
                [
                    "My eating schedule has been irregular.",
                    "I barely slept last night.",
                    "My sleep has been poor all week.",
                ],
                "Have you been sleeping enough recently?",
            )
        finally:
            for name, value in originals.items():
                setattr(questioner, name, value)

        self.assertEqual(valid, 1)
        self.assertEqual(terminate, 0)
        self.assertEqual(previous, "FOLLOWUP sleep")
        self.assertEqual(followup_calls, ["sleep", "eat"])
        self.assertEqual(
            reasoner_snapshots,
            [
                ("sleep", [], []),
                ("eat", [2], []),
            ],
        )
        self.assertEqual(
            logged_questions,
            [
                "FOLLOWUP sleep",
                "sleep accepted. FOLLOWUP eat",
            ],
        )
        self.assertEqual(updated["1"]["1"]["score"], [2])
        self.assertEqual(updated["2"]["1"]["score"], [2])
        self.assertIn("score_committed: true", updated["1"]["1"]["notes"][-1])
        self.assertIn("score_committed: true", updated["2"]["1"]["notes"][-1])
        self.assertEqual(
            questioner._pop_pending_validation_text(),
            "VALIDATION: eat accepted.",
        )

    def test_unresolved_segment_is_retained_while_valid_cross_score_is_saved(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "sleep",
                    "question": ["Have you been sleeping enough recently?"],
                    "score": [],
                    "notes": [],
                }
            },
            "2": {
                "1": {
                    "label": "eat",
                    "question": ["Have you been eating at regular times?"],
                    "score": [],
                    "notes": [],
                }
            },
        }

        outcome = questioner._if_valid_response(
            [("NA", 99), ("eat", 0)],
            1,
            "1",
            ["I am not sure what you mean", "I have been eating regularly"],
            "Have you been sleeping enough recently?",
            question_lib,
        )

        self.assertFalse(outcome.current_answered)
        self.assertEqual(len(outcome.assessments), 2)
        self.assertFalse(outcome.assessments[0].valid)
        self.assertTrue(outcome.assessments[1].valid)
        self.assertEqual(outcome.unresolved_segments, ["I am not sure what you mean"])
        self.assertEqual(outcome.covered_item_ids, {2})
        self.assertEqual(question_lib["1"]["1"]["score"], [])
        self.assertEqual(question_lib["2"]["1"]["score"], [0])


    def test_unique_support_labels_route_to_separate_items(self):
        question_lib = {
            "1": {
                "1": {
                    "label": "sleep",
                    "question": ["Have you been sleeping enough recently?"],
                    "score": [],
                    "notes": [],
                }
            },
            "18": {
                "1": {
                    "label": "family_support",
                    "question": ["Have you felt supported by your family?"],
                    "score": [],
                    "notes": [],
                }
            },
            "26": {
                "1": {
                    "label": "social_support",
                    "question": ["Do you have someone else in your support network?"],
                    "score": [],
                    "notes": [],
                }
            },
        }

        outcome = questioner._if_valid_response(
            [("family_support", 2), ("social_support", 0)],
            1,
            "1",
            [
                "My family has not been there for me",
                "I have close friends I can rely on",
            ],
            "Have you been sleeping enough recently?",
            question_lib,
        )

        self.assertFalse(outcome.current_answered)
        self.assertEqual(outcome.covered_item_ids, {26})
        self.assertEqual(
            [assessment.dimension for assessment in outcome.assessments],
            ["family_support", "social_support"],
        )
        self.assertEqual(
            [assessment.item_id for assessment in outcome.score2_queue],
            [18],
        )
        self.assertEqual(question_lib["18"]["1"]["score"], [])
        self.assertEqual(question_lib["26"]["1"]["score"], [0])


if __name__ == "__main__":
    unittest.main()
