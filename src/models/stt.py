"""Faster-Whisper STT wrapper used by the local CaiTI voice loop."""
import json
import os

from src.utils.config_loader import (
    STT_BEAM_SIZE,
    STT_BEST_OF,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_MODEL_PATH,
    STT_WITHOUT_TIMESTAMPS,
)
from src.utils.log_util import get_logger

logger = get_logger("STTGenerator")


class STTGenerator:
    def __init__(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install it before starting the voice app."
            ) from exc

        logger.info(
            "Loading Faster-Whisper model '%s' on %s (%s)",
            STT_MODEL_PATH,
            STT_DEVICE,
            STT_COMPUTE_TYPE,
        )
        self.model = WhisperModel(
            STT_MODEL_PATH,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
            num_workers=1,
        )

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        segments, _info = self.model.transcribe(
            audio_path,
            beam_size=STT_BEAM_SIZE,
            best_of=STT_BEST_OF,
            condition_on_previous_text=False,
            without_timestamps=STT_WITHOUT_TIMESTAMPS,
        )
        text = " ".join(segment.text for segment in segments).strip()
        logger.info("Transcription: %s", text)
        return json.dumps({"transcript": text, "detected_emotion": "neutral"})
