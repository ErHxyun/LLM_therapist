import os
import time
from typing import Optional

import pandas as pd

from src.utils.config_loader import RECORD_CSV, VOICE_EMPTY_TRANSCRIPT_RETRIES
from src.utils.io_record import HEADER
from src.utils.log_util import get_logger
from src.voice.backends import STTBackend, TTSBackend

logger = get_logger("VoiceIOLoop")


def _read_record() -> pd.DataFrame:
    return pd.read_csv(
        RECORD_CSV,
        dtype={"Question": str, "Question_Lock": "int64", "Resp": str, "Resp_Lock": "int64"},
    )


def _write_record(df: pd.DataFrame) -> None:
    folder = os.path.dirname(RECORD_CSV)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    tmp_path = RECORD_CSV + ".tmp"
    df.to_csv(tmp_path, columns=HEADER, index=False)
    os.replace(tmp_path, RECORD_CSV)


def _collect_transcript(stt: STTBackend, tts: TTSBackend, empty_retries: int) -> str:
    attempts = 0
    while True:
        transcript = stt.listen().strip()
        if transcript:
            logger.info("Captured STT transcript length=%s", len(transcript))
            return transcript
        attempts += 1
        if attempts > empty_retries:
            logger.warning("STT returned an empty transcript after %s attempts.", attempts)
            return ""
        tts.speak("I didn't catch that. Please say your answer again.")


def run_voice_io_loop(
    stt: STTBackend,
    tts: TTSBackend,
    poll_interval_sec: float = 0.1,
    empty_transcript_retries: Optional[int] = None,
) -> None:
    """
    Bridge CaiTI's record.csv question/response protocol to local STT/TTS.

    This is intentionally an I/O shell. It does not analyze audio or feed audio
    features to LLM modules, preserving the paper's text-only semantic analysis.
    """
    empty_retries = VOICE_EMPTY_TRANSCRIPT_RETRIES if empty_transcript_retries is None else empty_transcript_retries

    while True:
        time.sleep(poll_interval_sec)
        try:
            df = _read_record()
        except Exception:
            continue

        if int(df.loc[0, "Question_Lock"]) != 1:
            continue

        question = str(df.loc[0, "Question"])
        df.loc[0, "Question_Lock"] = 0
        _write_record(df)

        logger.info("Speaking question/response length=%s", len(question))
        tts.speak_stream(question)

        transcript = _collect_transcript(stt, tts, empty_retries)
        df = _read_record()
        df.loc[0, "Resp"] = transcript
        df.loc[0, "Resp_Lock"] = 0
        _write_record(df)
