"""Optional side-channel emotion analysis integration."""

from src.emotion.client import EmotionSideChannel, NullEmotionSideChannel, build_emotion_side_channel
from src.emotion.followup import (
    assess_emotion_followup,
    build_emotion_followup_settings,
    clear_emotion_session_state,
    clear_emotion_results_for_tests,
    pop_late_emotion_followup,
    queue_late_emotion_followup_request,
    register_emotion_result,
    wait_for_emotion_result,
)

__all__ = [
    "EmotionSideChannel",
    "NullEmotionSideChannel",
    "build_emotion_side_channel",
    "assess_emotion_followup",
    "build_emotion_followup_settings",
    "clear_emotion_session_state",
    "clear_emotion_results_for_tests",
    "pop_late_emotion_followup",
    "queue_late_emotion_followup_request",
    "register_emotion_result",
    "wait_for_emotion_result",
]
