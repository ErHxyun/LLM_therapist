import os
import re
import threading
import time
from contextlib import nullcontext
from typing import Optional

import pandas as pd

from src.utils.config_loader import RECORD_CSV, VOICE_EMPTY_TRANSCRIPT_RETRIES
from src.utils.io_record import HEADER, LONG_RESPONSE_PREFIX, NO_RESPONSE_PREFIX, RECORD_LOCK
from src.utils.log_util import get_logger
from src.voice.backends import STTBackend, TTSBackend, VoiceInterrupted
from src.voice.music import MusicBackend

logger = get_logger("VoiceIOLoop")
MUSIC_STOP_SETTLE_SEC = 0.15
SPOKEN_CHUNK_GAP_SEC = 0.55
_SPOKEN_LABEL_RE = re.compile(r"(?im)^\s*(guide|validation)\s*:\s*")
_SPOKEN_PARAGRAPH_BREAK_RE = re.compile(r"\s*\n\s*\n+\s*")
_SPOKEN_PARAGRAPH_TOKEN = "__CAITI_PARAGRAPH_BREAK__"
_SPOKEN_SOFT_BOUNDARY_PATTERNS = [
    (re.compile(r"\.\s+(Let's)\b"), r", \1"),
    (re.compile(r"\.\s+(Thank you)\b"), r", \1"),
    (re.compile(r"\.\s+(Can you|Could you|Would you)\b"), r", \1"),
    (re.compile(r"\.\s+(How|What|When|Where|Why)\b"), r", and \1"),
    (re.compile(r"\.\s+(Have you|Are you|Do you|Did you)\b"), r", and \1"),
]


def _read_record() -> pd.DataFrame:
    with RECORD_LOCK:
        return pd.read_csv(
            RECORD_CSV,
            dtype={"Question": str, "Question_Lock": "int64", "Resp": str, "Resp_Lock": "int64"},
        )


def _write_record(df: pd.DataFrame) -> None:
    with RECORD_LOCK:
        folder = os.path.dirname(RECORD_CSV)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        tmp_path = f"{RECORD_CSV}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
        df.to_csv(tmp_path, columns=HEADER, index=False)
        os.replace(tmp_path, RECORD_CSV)


def _listen_once(stt: STTBackend, music: Optional[MusicBackend]) -> str:
    if _music_is_background(music):
        return stt.listen().strip()
    listen_with_waiting_music = getattr(stt, "listen_with_waiting_music", None)
    if music is not None and callable(listen_with_waiting_music):
        return str(listen_with_waiting_music(music)).strip()
    return stt.listen().strip()


def _recent_listen_duration(stt: STTBackend, fallback_sec: float) -> float:
    value = getattr(stt, "last_audio_duration_sec", None)
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        return duration
    return max(0.0, fallback_sec)


def _music_is_background(music: Optional[MusicBackend]) -> bool:
    method = getattr(music, "is_background", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception as exc:
        logger.warning("Music background-state check failed: %s", exc)
        return False


def _music_is_playing(music: Optional[MusicBackend]) -> bool:
    method = getattr(music, "is_playing", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception as exc:
        logger.warning("Music playing-state check failed: %s", exc)
        return False


def _start_music(music: Optional[MusicBackend]) -> None:
    method = getattr(music, "start", None)
    if callable(method):
        method()


def _stop_music(music: Optional[MusicBackend]) -> None:
    method = getattr(music, "stop", None)
    if callable(method):
        method()


def _duck_music(music: Optional[MusicBackend]) -> None:
    method = getattr(music, "duck", None)
    if callable(method):
        method()
        return
    _stop_music(music)


def _restore_music_volume(music: Optional[MusicBackend]) -> None:
    method = getattr(music, "restore_volume", None)
    if callable(method):
        method()


def _suspend_music_for_spoken_audio(music: Optional[MusicBackend]) -> None:
    if music is None:
        return
    if _music_is_background(music):
        _stop_music(music)
        time.sleep(MUSIC_STOP_SETTLE_SEC)
        return
    _stop_music(music)
    time.sleep(MUSIC_STOP_SETTLE_SEC)


def _prepare_background_music_for_listening(music: Optional[MusicBackend]) -> None:
    if not _music_is_background(music):
        return
    _start_music(music)
    _duck_music(music)


def _session_should_interrupt(session_control=None) -> bool:
    if session_control is None:
        return False
    method = getattr(session_control, "should_interrupt_voice", None)
    if callable(method):
        try:
            return bool(method())
        except Exception as exc:
            logger.warning("Session control voice interrupt check failed: %s", exc)
    for method_name in ("is_shutdown_requested", "is_paused"):
        method = getattr(session_control, method_name, None)
        if callable(method):
            try:
                if method():
                    return True
            except Exception as exc:
                logger.warning("Session control interrupt check failed for %s: %s", method_name, exc)
    return False


def _should_discard_interrupted_voice_turn(session_control=None) -> bool:
    method = getattr(session_control, "should_discard_interrupted_voice_turn", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception as exc:
        logger.warning("Session control discard check failed: %s", exc)
        return False


def _should_keep_music_on_interrupted_voice_turn(session_control=None) -> bool:
    method = getattr(session_control, "should_keep_music_on_interrupted_voice_turn", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception as exc:
        logger.warning("Session control music-continuity check failed: %s", exc)
        return False


def _clear_pending_voice_turn() -> None:
    try:
        df = _read_record()
        df.loc[0, "Question"] = ""
        df.loc[0, "Question_Lock"] = 0
        df.loc[0, "Resp"] = ""
        df.loc[0, "Resp_Lock"] = 1
        _write_record(df)
    except Exception as exc:
        logger.warning("Failed to clear interrupted voice turn: %s", exc)


def _raise_if_interrupted(should_interrupt=None) -> None:
    if callable(should_interrupt) and should_interrupt():
        raise VoiceInterrupted("Voice turn interrupted by session control.")


def _set_interrupt_check(backend, checker) -> None:
    method = getattr(backend, "set_interrupt_check", None)
    if callable(method):
        method(checker)


def _voice_access_context(voice_access_lock=None):
    return voice_access_lock if voice_access_lock is not None else nullcontext()


def _set_response_profile(stt, profile: str) -> None:
    method = getattr(stt, "set_response_profile", None)
    if callable(method):
        method(profile)


def _wait_if_paused(session_control=None) -> None:
    if session_control is None:
        return
    method = getattr(session_control, "wait_while_paused", None)
    if callable(method):
        method(poll_interval_sec=0.05)
        return
    while _session_should_interrupt(session_control):
        shutdown_method = getattr(session_control, "is_shutdown_requested", None)
        if callable(shutdown_method) and shutdown_method():
            return
        time.sleep(0.05)


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


def _speak_chunk_with_status(
    tts: TTSBackend,
    text: str,
    status_leds=None,
    on_playback_start=None,
    should_interrupt=None,
) -> None:
    _raise_if_interrupted(should_interrupt)
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
        _raise_if_interrupted(should_interrupt)
        return

    if callable(on_playback_start):
        on_playback_start()
    _set_led_status(status_leds, "set_tts_active", True)
    try:
        tts.speak(text)
    finally:
        _set_led_status(status_leds, "set_tts_active", False)
    _raise_if_interrupted(should_interrupt)


def _speak_stream_with_status(tts: TTSBackend, text: str, status_leds=None, should_interrupt=None) -> None:
    block = str(text or "").strip()
    if not block:
        return
    if status_leds is None and should_interrupt is None:
        tts.speak_stream(block)
        return

    session_start_marked = False

    def mark_stream_playback_started_once() -> None:
        nonlocal session_start_marked
        if session_start_marked:
            return
        session_start_marked = True
        _set_led_status(status_leds, "mark_session_started")

    _speak_chunk_with_status(
        tts,
        block,
        status_leds=status_leds,
        on_playback_start=mark_stream_playback_started_once,
        should_interrupt=should_interrupt,
    )


def _sleep_interruptibly(duration_sec: float, should_interrupt=None, poll_interval_sec: float = 0.05) -> None:
    deadline = time.monotonic() + max(0.0, float(duration_sec))
    while time.monotonic() < deadline:
        _raise_if_interrupted(should_interrupt)
        time.sleep(min(poll_interval_sec, max(0.0, deadline - time.monotonic())))


def split_spoken_chunks(text: str) -> list[str]:
    raw = str(text or "")
    parts = [part for part in _SPOKEN_PARAGRAPH_BREAK_RE.split(raw) if str(part or "").strip()]
    if not parts:
        cleaned = clean_spoken_text(raw)
        return [cleaned] if cleaned else []
    chunks = [clean_spoken_text(part) for part in parts]
    return [chunk for chunk in chunks if chunk]


def _speak_stream_chunks_with_status(
    tts: TTSBackend,
    chunks: list[str],
    status_leds=None,
    should_interrupt=None,
    gap_sec: float = SPOKEN_CHUNK_GAP_SEC,
) -> None:
    for index, chunk in enumerate(chunks):
        _speak_stream_with_status(tts, chunk, status_leds, should_interrupt=should_interrupt)
        if index < len(chunks) - 1:
            _sleep_interruptibly(gap_sec, should_interrupt=should_interrupt)


def _collect_transcript(
    stt: STTBackend,
    tts: TTSBackend,
    empty_retries: int,
    music: Optional[MusicBackend] = None,
    status_leds=None,
    should_interrupt=None,
) -> tuple[str, float]:
    attempts = 0
    while True:
        _raise_if_interrupted(should_interrupt)
        if _music_is_background(music):
            _prepare_background_music_for_listening(music)
        _set_led_status(status_leds, "set_stt_active", True)
        listen_started = time.monotonic()
        try:
            transcript = _listen_once(stt, music)
        finally:
            listen_elapsed = time.monotonic() - listen_started
            _set_led_status(status_leds, "set_stt_active", False)
            if _music_is_background(music):
                _restore_music_volume(music)
        _raise_if_interrupted(should_interrupt)
        listen_duration_sec = _recent_listen_duration(stt, listen_elapsed)
        if transcript:
            logger.info(
                "Captured STT transcript length=%s listen_duration=%.2fs",
                len(transcript),
                listen_duration_sec,
            )
            return transcript, listen_duration_sec
        attempts += 1
        if attempts > empty_retries:
            logger.warning("STT returned an empty transcript after %s attempts.", attempts)
            return "", listen_duration_sec
        if music is not None and not _music_is_background(music):
            _stop_music(music)
            time.sleep(MUSIC_STOP_SETTLE_SEC)
        if _music_is_background(music):
            _suspend_music_for_spoken_audio(music)
        _speak_chunk_with_status(
            tts,
            "I didn't catch that. Please say your answer again.",
            status_leds,
            should_interrupt=should_interrupt,
        )


def clean_spoken_text(text: str) -> str:
    cleaned = _SPOKEN_LABEL_RE.sub("", str(text or "")).strip()
    cleaned = _SPOKEN_PARAGRAPH_BREAK_RE.sub(f" {_SPOKEN_PARAGRAPH_TOKEN} ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    for pattern, replacement in _SPOKEN_SOFT_BOUNDARY_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(rf"\.\s*{_SPOKEN_PARAGRAPH_TOKEN}\s*", f" {_SPOKEN_PARAGRAPH_TOKEN} ", cleaned)
    cleaned = re.sub(
        rf"{_SPOKEN_PARAGRAPH_TOKEN}\s+(How|What|When|Where|Why|Have you|Are you|Do you|Did you|Can you|Could you|Would you)\b",
        r", and \1",
        cleaned,
    )
    cleaned = re.sub(rf"\s*{_SPOKEN_PARAGRAPH_TOKEN}\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+([,.;?!])", r"\1", cleaned)
    return cleaned.strip()


def parse_voice_prompt_metadata(text: str) -> tuple[str, bool, str]:
    raw = str(text or "")
    expects_response = True
    response_profile = "standard"
    while True:
        if raw.startswith(NO_RESPONSE_PREFIX):
            raw = raw[len(NO_RESPONSE_PREFIX):].lstrip("\r\n")
            expects_response = False
            continue
        if raw.startswith(LONG_RESPONSE_PREFIX):
            raw = raw[len(LONG_RESPONSE_PREFIX):].lstrip("\r\n")
            response_profile = "long"
            continue
        break
    return raw, expects_response, response_profile


def parse_voice_prompt(text: str) -> tuple[str, bool]:
    spoken_text, expects_response, _response_profile = parse_voice_prompt_metadata(text)
    return spoken_text, expects_response


def process_voice_turn(
    stt: STTBackend,
    tts: TTSBackend,
    music: Optional[MusicBackend] = None,
    empty_transcript_retries: Optional[int] = None,
    activity_event: Optional[threading.Event] = None,
    status_leds=None,
    session_control=None,
    intermission_runner=None,
    voice_access_lock=None,
) -> bool:
    """
    Process one record.csv voice turn if a question is ready.

    Returns True when it spoke a question and wrote a transcript, otherwise False.
    """
    empty_retries = VOICE_EMPTY_TRANSCRIPT_RETRIES if empty_transcript_retries is None else empty_transcript_retries

    if _session_should_interrupt(session_control):
        if _should_discard_interrupted_voice_turn(session_control):
            _clear_pending_voice_turn()
        else:
            _wait_if_paused(session_control)
        return False

    try:
        with RECORD_LOCK:
            df = _read_record()
            if int(df.loc[0, "Question_Lock"]) != 1:
                return False
            original_question = str(df.loc[0, "Question"])
            df.loc[0, "Question_Lock"] = 0
            _write_record(df)
    except Exception:
        return False

    if activity_event is not None:
        activity_event.clear()

    try:
        with _voice_access_context(voice_access_lock):
            should_interrupt = (lambda: _session_should_interrupt(session_control)) if session_control is not None else None
            _set_interrupt_check(tts, should_interrupt)
            _set_interrupt_check(stt, should_interrupt)
            background_music_was_playing = _music_is_background(music) and _music_is_playing(music)
            try:
                if music is not None:
                    _suspend_music_for_spoken_audio(music)

                question = original_question
                spoken_question, expects_response, response_profile = parse_voice_prompt_metadata(question)
                _set_response_profile(stt, response_profile)
                spoken_chunks = split_spoken_chunks(spoken_question)
                logger.info(
                    "Speaking question/response length=%s expects_response=%s response_profile=%s spoken_chunks=%s",
                    len(question),
                    expects_response,
                    response_profile,
                    len(spoken_chunks),
                )
                _speak_stream_chunks_with_status(tts, spoken_chunks, status_leds, should_interrupt=should_interrupt)

                if expects_response:
                    transcript, listen_duration_sec = _collect_transcript(
                        stt,
                        tts,
                        empty_retries,
                        music,
                        status_leds,
                        should_interrupt,
                    )
                    with RECORD_LOCK:
                        df = _read_record()
                        df.loc[0, "Resp"] = transcript
                        df.loc[0, "Resp_Lock"] = 0
                        _write_record(df)
                    if transcript.strip():
                        _run_intermission_until_next_question(
                            intermission_runner,
                            should_interrupt,
                            user_speech_duration_sec=listen_duration_sec,
                        )
                    else:
                        logger.info("Skipping intermission because the main transcript is empty.")
                    if music is not None:
                        _start_music(music)
                else:
                    if background_music_was_playing:
                        _start_music(music)
                        _restore_music_volume(music)
                    with RECORD_LOCK:
                        df = _read_record()
                        df.loc[0, "Resp"] = ""
                        df.loc[0, "Resp_Lock"] = 1
                        _write_record(df)
                    logger.info("Spoke system message without collecting a response.")
                return True
            except VoiceInterrupted:
                discard_interrupted_turn = _should_discard_interrupted_voice_turn(session_control)
                if discard_interrupted_turn:
                    logger.info("Voice turn interrupted by workflow override; clearing pending question.")
                else:
                    logger.info("Voice turn interrupted; restoring pending question for replay after resume.")
                try:
                    with RECORD_LOCK:
                        df = _read_record()
                        df.loc[0, "Question"] = "" if discard_interrupted_turn else original_question
                        df.loc[0, "Question_Lock"] = 0 if discard_interrupted_turn else 1
                        df.loc[0, "Resp"] = ""
                        df.loc[0, "Resp_Lock"] = 1
                        _write_record(df)
                except Exception as exc:
                    logger.warning("Failed to restore interrupted voice turn: %s", exc)
                if music is not None:
                    if _music_is_background(music):
                        if _should_keep_music_on_interrupted_voice_turn(session_control):
                            _start_music(music)
                            _restore_music_volume(music)
                        else:
                            pause_method = getattr(music, "pause", None)
                            if callable(pause_method):
                                pause_method()
                            else:
                                _stop_music(music)
                    else:
                        _stop_music(music)
                _set_led_status(status_leds, "set_tts_active", False)
                _set_led_status(status_leds, "set_stt_active", False)
                if not discard_interrupted_turn:
                    _wait_if_paused(session_control)
                return False
            finally:
                _set_interrupt_check(tts, None)
                _set_interrupt_check(stt, None)
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


def _next_question_ready() -> bool:
    try:
        df = _read_record()
    except Exception:
        return False
    try:
        return int(df.loc[0, "Question_Lock"]) == 1
    except Exception:
        return False


def _run_intermission_until_next_question(
    intermission_runner,
    should_interrupt=None,
    user_speech_duration_sec: float | None = None,
) -> None:
    if intermission_runner is None:
        return
    method = getattr(intermission_runner, "run_until_ready", None)
    if not callable(method):
        return
    try:
        method(
            _next_question_ready,
            should_stop=should_interrupt,
            user_speech_duration_sec=user_speech_duration_sec,
        )
    except TypeError:
        method(_next_question_ready, should_stop=should_interrupt)


def run_voice_io_loop(
    stt: STTBackend,
    tts: TTSBackend,
    music: Optional[MusicBackend] = None,
    poll_interval_sec: float = 0.1,
    empty_transcript_retries: Optional[int] = None,
    activity_event: Optional[threading.Event] = None,
    status_leds=None,
    session_control=None,
    intermission_runner=None,
    voice_access_lock=None,
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
        if _session_should_interrupt(session_control):
            if _should_discard_interrupted_voice_turn(session_control):
                _clear_pending_voice_turn()
            else:
                _wait_if_paused(session_control)
            continue
        process_voice_turn(
            stt,
            tts,
            music,
            empty_transcript_retries,
            activity_event,
            status_leds,
            session_control=session_control,
            intermission_runner=intermission_runner,
            voice_access_lock=voice_access_lock,
        )
