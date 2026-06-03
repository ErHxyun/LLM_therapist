import unittest
import time

from src.emotion.followup import (
    EmotionFollowupSettings,
    assess_emotion_followup,
    clear_emotion_results_for_tests,
    pop_late_emotion_followup,
    queue_late_emotion_followup_request,
    register_emotion_result,
    wait_for_emotion_result,
)


def emotion_record(
    *,
    transcript="I'm fine.",
    confidence=80,
    risk=70,
    audio_emotion=None,
    context_emotion=None,
    comparison=None,
    audio_scores=None,
    quality_flags=None,
):
    return {
        "status": "ok",
        "request": {"transcript": transcript},
        "response": {
            "audio_emotion": audio_emotion if audio_emotion is not None else [0, -1],
            "context_emotion": context_emotion if context_emotion is not None else [0, 1],
            "audio_scores": audio_scores if audio_scores is not None else {"tension": 70, "hesitation": 50, "stability": 45},
            "emotion_comparison": comparison
            if comparison is not None
            else {"audio_vs_context_consistent": False, "valence_conflict": True},
            "final_assessment": {
                "confidence": confidence,
                "credibility_risk": risk,
                "quality_flags": quality_flags or [],
            },
            "final_result": {
                "credibility_risk": risk,
                "risk_level": "Moderate",
            },
        },
    }


class EmotionFollowupTest(unittest.TestCase):
    def setUp(self):
        clear_emotion_results_for_tests()
        self.settings = EmotionFollowupSettings(
            enabled=True,
            wait_timeout_sec=0.0,
            min_confidence=50,
            risk_threshold=60,
            light_risk_threshold=45,
        )

    def tearDown(self):
        clear_emotion_results_for_tests()

    def test_score_zero_distressed_mismatch_triggers_followup_without_score_change(self):
        decision = assess_emotion_followup(
            score=0,
            user_text="I'm fine.",
            record=emotion_record(),
            settings=self.settings,
        )

        self.assertTrue(decision.should_follow_up)
        self.assertEqual(decision.reason, "emotion_distress_hidden_by_low_content_score")
        self.assertIn("gently check", decision.followup_text)

    def test_score_zero_distressed_voice_triggers_even_without_explicit_mismatch(self):
        decision = assess_emotion_followup(
            score=0,
            user_text="I'm fine.",
            record=emotion_record(
                confidence=65,
                risk=45,
                audio_emotion=[-1, 1],
                context_emotion=[-1, 0],
                comparison={"audio_vs_context_consistent": True},
                audio_scores={"tension": 50, "hesitation": 85, "stability": 35},
            ),
            settings=self.settings,
        )

        self.assertTrue(decision.should_follow_up)
        self.assertEqual(decision.reason, "emotion_distress_acoustic_cue")
        self.assertIn("strained", decision.followup_text)

    def test_score_two_positive_voice_triggers_confirmation_followup(self):
        decision = assess_emotion_followup(
            score=2,
            user_text="I barely sleep and feel exhausted.",
            record=emotion_record(
                transcript="I barely sleep and feel exhausted.",
                risk=20,
                audio_emotion=[0, 1],
                context_emotion=[0, -1],
                comparison={"audio_vs_context_consistent": True},
                audio_scores={"tension": 30, "hesitation": 20, "stability": 80},
            ),
            settings=self.settings,
        )

        self.assertTrue(decision.should_follow_up)
        self.assertEqual(decision.reason, "emotion_positive_tone_with_high_content_score")
        self.assertIn("lighter tone", decision.followup_text)

    def test_unreliable_result_does_not_trigger_followup(self):
        decision = assess_emotion_followup(
            score=0,
            user_text="I'm fine.",
            record=emotion_record(confidence=20, quality_flags=["audio_quality_ok:false"]),
            settings=self.settings,
        )

        self.assertFalse(decision.should_follow_up)
        self.assertEqual(decision.reason, "unreliable_emotion_result")

    def test_registry_matches_current_transcript_only_from_memory(self):
        register_emotion_result(emotion_record(transcript="No, I am not getting enough sleep."))

        matched = wait_for_emotion_result("No, I am not getting enough sleep.", 0.0)
        missing = wait_for_emotion_result("My appetite is normal.", 0.0)

        self.assertIsNotNone(matched)
        self.assertIsNone(missing)

    def test_late_emotion_result_queues_followup(self):
        queue_late_emotion_followup_request(
            dimension="mood",
            score=0,
            user_text="I'm fine.",
            question_text="How has your mood been lately?",
            settings=self.settings,
            started_at=time.monotonic(),
        )
        register_emotion_result(
            emotion_record(
                transcript="I'm fine.",
                comparison={"audio_vs_context_consistent": True},
                audio_scores={"tension": 50, "hesitation": 85, "stability": 35},
                risk=50,
            )
        )

        decision = pop_late_emotion_followup()

        self.assertIsNotNone(decision)
        self.assertTrue(decision.should_follow_up)
        self.assertEqual(decision.reason, "emotion_distress_acoustic_cue")
        self.assertEqual(decision.metadata["mode"], "late_emotion_followup")

    def test_late_request_matches_existing_emotion_result(self):
        register_emotion_result(
            emotion_record(
                transcript="I'm fine.",
                comparison={"audio_vs_context_consistent": True},
                audio_scores={"tension": 50, "hesitation": 85, "stability": 35},
                risk=50,
            )
        )
        queue_late_emotion_followup_request(
            dimension="mood",
            score=0,
            user_text="I'm fine.",
            question_text="How has your mood been lately?",
            settings=self.settings,
            started_at=time.monotonic(),
        )

        decision = pop_late_emotion_followup()

        self.assertIsNotNone(decision)
        self.assertTrue(decision.should_follow_up)
        self.assertEqual(decision.reason, "emotion_distress_acoustic_cue")

    def test_expired_late_emotion_result_does_not_queue_followup(self):
        queue_late_emotion_followup_request(
            dimension="mood",
            score=0,
            user_text="I'm fine.",
            question_text="How has your mood been lately?",
            settings=self.settings,
            started_at=time.monotonic() - 20.0,
        )
        register_emotion_result(emotion_record(transcript="I'm fine."))

        self.assertIsNone(pop_late_emotion_followup())


if __name__ == "__main__":
    unittest.main()
