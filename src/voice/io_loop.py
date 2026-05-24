import os
import re
import threading
import time
from typing import Optional

import pandas as pd

from src.utils.config_loader import RECORD_CSV, VOICE_EMPTY_TRANSCRIPT_RETRIES
from src.utils.io_record import HEADER, NO_RESPONSE_PREFIX
from src.utils.log_util import get_logger
from src.voice.backends import STTBackend, TTSBackend
from src.voice.music import MusicBackend
from src.voice.sentence_stream import split_for_tts

logger = get_logger("VoiceIOLoop")
MUSIC_STOP_SETTLE_SEC = 0.15
_SPOKEN_LABEL_RE = re.compile(r"(?im)^\s*(guide|validation)\s*:\s*")


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


def _listen_once(stt: STTBackend, music: Optional[MusicBackend]) -> str:
    listen_with_waiting_music = getattr(stt, "listen_with_waiting_music", None)
    if music is not None and callable(listen_with_waiting_music):
        return str(listen_with_waiting_music(music)).strip()
    return stt.listen().strip()


def _set_led_status(status_leds, method_name: str, *args) -> None:
    if status_leds is None:
        return
    method = getattr(status_leds, method_name, None)
    if not callable(method):
        return
    try:
        method(*args)
    except Exception as exc:
        logger.warning("Status LED hook failed for %s: %s", method_name, exc)


def _speak_chunk_with_status(tts: TTSBackend, text: str, status_leds=None, on_playback_start=None) -> None:
    set_hook = getattr(tts, "set_playback_status_hook", None)
    if status_leds is not None and callable(set_hook):
        playback_start_called = False
        tts_active = False

        def playback_status_hook(active: bool) -> None:
            nonlocal playback_start_called, tts_active
            if active and not playback_start_called:
                playback_start_called = True
                if callable(on_playback_start):
                    on_playback_start()
            tts_active = active
            _set_led_status(status_leds, "set_tts_active", active)

        set_hook(playback_status_hook)
        try:
            tts.speak(text)
        finally:
            set_hook(None)
            if tts_active:
                _set_led_status(status_leds, "set_tts_active", False)
        return

    if callable(on_playback_start):
        on_playback_start()
    _set_led_status(status_leds, "set_tts_active", True)
    try:
        tts.speak(text)
    finally:
        _set_led_status(status_leds, "set_tts_active", False)


def _speak_chunk_and_hold_stream_status(tts: TTSBackend, text: str, on_playback_start=None) -> None:
    set_hook = getattr(tts, "set_playback_status_hook", None)
    if callable(set_hook):

        def playback_status_hook(active: bool) -> None:
            if active and callable(on_playback_start):
                on_playback_start()

        set_hook(playback_status_hook)
        try:
            tts.speak(text)
        finally:
            set_hook(None)
        return

    if callable(on_playback_start):
        on_playback_start()
    tts.speak(text)


def _speak_stream_with_status(tts: TTSBackend, text: str, status_leds=None) -> None:
    if status_leds is None:
        tts.speak_stream(text)
        return

    chunks = split_for_tts(text)
    if not chunks:
        return

    session_start_marked = False
    stream_playback_started = False

    def mark_stream_playback_started_once() -> None:
        nonlocal session_start_marked, stream_playback_started
        if session_start_marked:
            if not stream_playback_started:
                stream_playback_started = True
                _set_led_status(status_leds, "set_tts_active", True)
            return
        session_start_marked = True
        _set_led_status(status_leds, "mark_session_started")
        if not stream_playback_started:
            stream_playback_started = True
            _set_led_status(status_leds, "set_tts_active", True)

    try:
        for chunk in chunks:
            _speak_chunk_and_hold_stream_status(tts, chunk, mark_stream_playback_started_once)
    finally:
        if stream_playback_started:
            _set_led_status(status_leds, "set_tts_active", False)


def _collect_transcript(
    stt: STTBackend,
    tts: TTSBackend,
    empty_retries: int,
    music: Optional[MusicBackend] = None,
    status_leds=None,
) -> str:
    attempts = 0
    while True:
        _set_led_status(status_leds, "set_stt_active", True)
        try:
            transcript = _listen_once(stt, music)
        finally:
            _set_led_status(status_leds, "set_stt_active", False)
        if transcript:
            logger.info("Captured STT transcript length=%s", len(transcript))
            return transcript
        attempts += 1
        if attempts > empty_retries:
            logger.warning("STT returned an empty transcript after %s attempts.", attempts)
            return ""
        if music is not None:
            music.stop()
            time.sleep(MUSIC_STOP_SETTLE_SEC)
        _speak_chunk_with_status(tts, "I didn't catch that. Please say your answer again.", status_leds)


def clean_spoken_text(text: str) -> str:
    return _SPOKEN_LABEL_RE.sub("", str(text or "")).strip()


def parse_voice_prompt(text: str) -> tuple[str, bool]:
    raw = str(text or "")
    expects_response = True
    if raw.startswith(NO_RESPONSE_PREFIX):
        raw = raw[len(NO_RESPONSE_PREFIX):].lstrip("\r\n")
        expects_response = False
    return raw, expects_response


def process_voice_turn(
    stt: STTBackend,
    tts: TTSBackend,
    music: Optional[MusicBackend] = None,
    empty_transcript_retries: Optional[int] = None,
    activity_event: Optional[threading.Event] = None,
    status_leds=None,
) -> bool:
    """
    Process one record.csv voice turn if a question is ready.

    Returns True when it spoke a question and wrote a transcript, otherwise False.
    """
    empty_retries = VOICE_EMPTY_TRANSCRIPT_RETRIES if empty_transcript_retries is None else empty_transcript_retries

    try:
        df = _read_record()
    except Exception:
        return False

    if int(df.loc[0, "Question_Lock"]) != 1:
        return False

    if activity_event is not None:
        activity_event.clear()

    try:
        if music is not None:
            music.stop()
            time.sleep(MUSIC_STOP_SETTLE_SEC)

        question = str(df.loc[0, "Question"])
        spoken_question, expects_response = parse_voice_prompt(question)
        spoken_question = clean_spoken_text(spoken_question)
        df.loc[0, "Question_Lock"] = 0
        _write_record(df)

        logger.info(
            "Speaking question/response length=%s expects_response=%s",
            len(question),
            expects_response,
        )
        _speak_stream_with_status(tts, spoken_question, status_leds)

        df = _read_record()
        if expects_response:
            transcript = _collect_transcript(stt, tts, empty_retries, music, status_leds)
            df = _read_record()
            df.loc[0, "Resp"] = transcript
            df.loc[0, "Resp_Lock"] = 0
            _write_record(df)
            if music is not None:
                music.start()
        else:
            df.loc[0, "Resp"] = ""
            df.loc[0, "Resp_Lock"] = 1
            _write_record(df)
            logger.info("Spoke system message without collecting a response.")
        return True
    finally:
        if activity_event is not None:
            activity_event.set()


def wait_for_voice_io_drain(
    activity_event: Optional[threading.Event] = None,
    timeout_sec: float = 90.0,
    poll_interval_sec: float = 0.1,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            df = _read_record()
            question_pending = int(df.loc[0, "Question_Lock"]) == 1
        except Exception:
            question_pending = False
        idle = activity_event is None or activity_event.is_set()
        if not question_pending and idle:
            return True
        time.sleep(poll_interval_sec)
    logger.warning("Timed out waiting for voice I/O to drain.")
    return False


def run_voice_io_loop(
    stt: STTBackend,
    tts: TTSBackend,
    music: Optional[MusicBackend] = None,
    poll_interval_sec: float = 0.1,
    empty_transcript_retries: Optional[int] = None,
    activity_event: Optional[threading.Event] = None,
    status_leds=None,
) -> None:
    """
    Bridge CaiTI's record.csv question/response protocol to local STT/TTS.

    This is intentionally an I/O shell. It does not analyze audio or feed audio
    features to LLM modules, preserving the paper's text-only semantic analysis.
    """
    if activity_event is not None:
        activity_event.set()
    while True:
        time.sleep(poll_interval_sec)
        process_voice_turn(stt, tts, music, empty_transcript_retries, activity_event, status_leds)
