"""Optional side-channel emotion analysis integration."""

from src.emotion.client import EmotionSideChannel, NullEmotionSideChannel, build_emotion_side_channel

__all__ = ["EmotionSideChannel", "NullEmotionSideChannel", "build_emotion_side_channel"]
