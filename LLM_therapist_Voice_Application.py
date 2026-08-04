import threading
import time
import os
from contextlib import nullcontext
from dataclasses import replace

from src.handler_rl import HandlerRL
from src.emotion import clear_emotion_session_state
from src.hardware.music_mode_button import build_music_mode_button_controller
from src.hardware.session_button import build_session_button_controller
from src.hardware.status_leds import build_status_led_controller
from src.hardware.volume_buttons import build_volume_button_controller
from src.intermission import build_intermission_runner
from src.questioner import reset_questioner_session_state
from src.runtime.poweroff import clear_system_poweroff_request, request_system_poweroff
from src.runtime.session_context import (
    activate_session_context,
    complete_current_session,
    create_new_session,
    deactivate_session_context,
    find_resumable_session,
    interrupt_current_session,
    is_protected_participant,
    update_session_status,
)
from src.runtime.user_context import build_guest_user_id, normalize_spoken_user_id
from src.runtime.status_monitor import build_status_monitor, get_active_status_monitor, set_active_status_monitor
from src.session.control import SessionShutdownRequested, build_session_control
from src.utils import config_loader
from src.utils.io_record import reset_record_state
from src.utils.llm_client import preload_llm_runtime
from src.utils.log_util import get_logger
from src.utils.session_event_logger import build_session_id, set_session_id
from src.voice.backends import VoiceInterrupted, build_stt, build_tts
from src.voice.io_loop import run_voice_io_loop, wait_for_voice_io_drain
from src.voice.music import build_music

logger = get_logger("VoiceApplication")
SHUTDOWN_SPOKEN_MESSAGE = "Caiti is shutting down."
USER_ID_PROMPT = "Before we begin, please say your participant ID."
USER_ID_RETRY_PROMPT = "I did not catch the participant ID. Please say the participant ID again."
USER_ID_FALLBACK_MESSAGE = "I did not catch an ID, so I will use a temporary guest ID for this session."
USER_ID_CONFIRM_PROMPT = "I heard your participant ID as {value}. Is that correct? Please say yes or no."
USER_ID_CONFIRM_RETRY_PROMPT = "Please say yes if that participant ID is correct, or say no if you want to try again."
USER_ID_REENTER_PROMPT = "Okay, please say your participant ID again."
PROTECTED_ID_MESSAGE = (
    "That participant ID belongs to the completed one through twenty-five data collection set "
    "and cannot be changed. Please enter a new participant ID."
)
RESUME_SESSION_PROMPT = (
    "I found an unfinished session for participant {value}. "
    "Please say resume to continue it, or say new to start a new session."
)
RESUME_SESSION_RETRY_PROMPT = "Please say resume or new."
USER_NAME_PROMPT = "Thank you. Now please say your name."
USER_NAME_RETRY_PROMPT = "I did not catch the name. Please say your name again."
USER_NAME_CONFIRM_PROMPT = "I heard your name as {value}. Is that correct? Please say yes or no."
USER_NAME_CONFIRM_RETRY_PROMPT = "Please say yes if that name is correct, or say no if you want to try again."
USER_NAME_REENTER_PROMPT = "Okay, please say your name again."
_PERSISTENT_APP_LOOP_ENV = "CAITI_PERSISTENT_APP_LOOP"


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _event_is_set(event) -> bool:
    method = getattr(event, "is_set", None)
    if not callable(method):
        return False
    return bool(method())


def _music_is_playing(music) -> bool:
    method = getattr(music, "is_playing", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception as exc:
        logger.warning("Waiting music state check failed: %s", exc)
        return False


def _music_is_background(music) -> bool:
    method = getattr(music, "is_background", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception as exc:
        logger.warning("Waiting music background-state check failed: %s", exc)
        return False


def _session_is_active(session_control) -> bool:
    for method_name in ("is_paused", "is_shutdown_requested"):
        method = getattr(session_control, method_name, None)
        if not callable(method):
            continue
        try:
            if method():
                return False
        except Exception as exc:
            logger.warning("Session state check failed for %s: %s", method_name, exc)
            return False
    return True


def _pause_music(music) -> None:
    method = getattr(music, "pause", None)
    if callable(method):
        method()
        return
    music.stop()


def _resume_music(music) -> None:
    method = getattr(music, "resume", None)
    if callable(method):
        method()
        return
    music.start()


def _resume_music_when_voice_idle(session_control, music, voice_idle) -> None:
    def resume_if_safe() -> bool:
        if not _session_is_active(session_control):
            return True
        if not _event_is_set(voice_idle):
            return False
        _resume_music(music)
        return True

    if resume_if_safe():
        return

    def wait_and_resume() -> None:
        while not resume_if_safe():
            time.sleep(0.05)

    threading.Thread(target=wait_and_resume, name="caiti-music-resume", daemon=True).start()


def _wait_for_voice_idle(activity_event, timeout_sec: float = 2.0) -> None:
    if activity_event is None or _event_is_set(activity_event):
        return
    wait = getattr(activity_event, "wait", None)
    if callable(wait):
        wait(timeout=timeout_sec)


def _set_status_led(status_leds, method_name: str, *args) -> None:
    method = getattr(status_leds, method_name, None)
    if not callable(method):
        return
    try:
        method(*args)
    except Exception as exc:
        logger.warning("Status LED hook failed for %s: %s", method_name, exc)


def _set_interrupt_check(backend, checker) -> None:
    method = getattr(backend, "set_interrupt_check", None)
    if callable(method):
        method(checker)


def _voice_access_context(voice_access_lock=None):
    return voice_access_lock if voice_access_lock is not None else nullcontext()


def _duck_music_for_tts(music) -> None:
    if _music_is_background(music):
        stop_method = getattr(music, "stop", None)
        if callable(stop_method):
            stop_method()
            time.sleep(0.15)
            return
    method = getattr(music, "duck", None)
    if callable(method):
        method()
        return
    _pause_music(music)


def _restore_music_volume(music) -> None:
    method = getattr(music, "restore_volume", None)
    if callable(method):
        method()


def _prepare_music_for_stt(music) -> None:
    if _music_is_background(music):
        _pause_music(music)
        return
    _duck_music_for_tts(music)


def _speak_shutdown_message(
    tts,
    music,
    status_leds,
    message: str = SHUTDOWN_SPOKEN_MESSAGE,
    should_interrupt=None,
    voice_access_lock=None,
) -> bool:
    text = str(message or "").strip()
    if not text:
        return True
    with _voice_access_context(voice_access_lock):
        try:
            _set_interrupt_check(tts, should_interrupt)
            _duck_music_for_tts(music)
            _set_status_led(status_leds, "set_tts_active", True)
            tts.speak(text)
            return True
        except VoiceInterrupted:
            logger.info("Shutdown message interrupted.")
        except Exception as exc:
            logger.warning("Failed to speak shutdown message: %s", exc)
        finally:
            _set_status_led(status_leds, "set_tts_active", False)
            _set_interrupt_check(tts, None)
    return False


def _handle_short_press_with_music(session_control, music, voice_idle, restore_music_after_pause=None) -> None:
    was_paused = session_control.is_paused()
    music_was_playing = _music_is_playing(music)
    session_control.handle_short_press()
    is_paused = session_control.is_paused()
    if is_paused:
        if restore_music_after_pause is not None:
            if music_was_playing:
                restore_music_after_pause.set()
            else:
                restore_music_after_pause.clear()
        _pause_music(music)
    elif was_paused:
        if restore_music_after_pause is None:
            if _event_is_set(voice_idle):
                _resume_music(music)
            return
        should_restore_music = restore_music_after_pause.is_set()
        restore_music_after_pause.clear()
        if should_restore_music:
            _resume_music_when_voice_idle(session_control, music, voice_idle)


def _warm_up_stt(stt) -> None:
    warm_up = getattr(stt, "warm_up", None)
    if not callable(warm_up):
        return
    try:
        warm_up()
    except Exception as exc:
        logger.warning("STT warm-up failed; first listen will retry. error=%s", exc)


def _warm_up_tts(tts, role: str = "primary") -> None:
    warm_up = getattr(tts, "warm_up", None)
    if not callable(warm_up):
        return
    try:
        warm_up()
    except Exception as exc:
        logger.warning("%s TTS warm-up failed; first playback will retry. error=%s", role, exc)


def _warm_up_intermission_tts(intermission_runner, primary_tts) -> None:
    intermission_tts = getattr(intermission_runner, "tts", None)
    if intermission_tts is None or intermission_tts is primary_tts:
        return
    _warm_up_tts(intermission_tts, "intermission")


def _preload_llm_runtime() -> None:
    logger.info("Preloading local CaiTI LLM runtime before first spoken turn.")
    preload_llm_runtime()
    logger.info("Local CaiTI LLM runtime ready.")


def _persistent_app_loop_enabled() -> bool:
    return _bool_env(_PERSISTENT_APP_LOOP_ENV, default=False)


def _reset_session_runtime(
    session_control,
    session_id: str | None = None,
    subject_id: str | None = None,
) -> str:
    resolved_session_id = session_id or build_session_id(subject_id)
    set_session_id(resolved_session_id)
    reset_record_state()
    reset_questioner_session_state()
    clear_emotion_session_state()
    return resolved_session_id


def _wait_for_session_start(
    session_control,
    status_monitor,
    persistent_loop: bool,
) -> bool:
    if persistent_loop:
        set_phase = getattr(session_control, "set_phase", None)
        if callable(set_phase):
            set_phase("ready_idle")
        elif status_monitor is not None:
            status_monitor.set_phase("ready_idle")
    return session_control.wait_for_start()


def _should_auto_poweroff_after_session_complete() -> bool:
    return bool(config_loader.SESSION_AUTO_POWEROFF_ON_COMPLETE)


def _request_configured_system_poweroff(reason: str) -> bool:
    return request_system_poweroff(reason)


def _speak_prompt(
    tts,
    music,
    status_leds,
    message: str,
    should_interrupt=None,
    source: str = "system",
    expects_response: bool = False,
    voice_access_lock=None,
) -> None:
    monitor = get_active_status_monitor()
    if monitor is not None:
        setter = getattr(monitor, "set_prompt", None)
        if callable(setter):
            try:
                setter(text=str(message or ""), source=source, expects_response=bool(expects_response))
            except Exception:
                pass
    _speak_shutdown_message(
        tts,
        music,
        status_leds,
        message=message,
        should_interrupt=should_interrupt,
        voice_access_lock=voice_access_lock,
    )


def _listen_with_stt(stt, music, status_leds, should_interrupt=None, voice_access_lock=None) -> str:
    with _voice_access_context(voice_access_lock):
        restore_background_music = _music_is_background(music) and _music_is_playing(music)
        try:
            _set_interrupt_check(stt, should_interrupt)
            _prepare_music_for_stt(music)
            _set_status_led(status_leds, "set_stt_active", True)
            return str(stt.listen() or "").strip()
        finally:
            _set_status_led(status_leds, "set_stt_active", False)
            if restore_background_music:
                _resume_music(music)
            else:
                _restore_music_volume(music)
            _set_interrupt_check(stt, None)


def _collect_identity_field(
    *,
    stt,
    tts,
    music,
    status_leds,
    initial_prompt: str,
    retry_prompt: str,
    should_interrupt=None,
    max_attempts: int = 2,
    voice_access_lock=None,
) -> str:
    prompt = initial_prompt
    for _attempt in range(max(1, int(max_attempts))):
        _speak_prompt(
            tts,
            music,
            status_leds,
            prompt,
            should_interrupt=should_interrupt,
            source="identity",
            expects_response=True,
            voice_access_lock=voice_access_lock,
        )
        response = _listen_with_stt(
            stt,
            music,
            status_leds,
            should_interrupt=should_interrupt,
            voice_access_lock=voice_access_lock,
        )
        if response:
            return response
        prompt = retry_prompt
    return ""


def _is_affirmative_response(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    tokens = {token.strip(".,!?;:") for token in normalized.split()}
    return normalized in {
        "yes",
        "yeah",
        "yep",
        "correct",
        "that is correct",
        "right",
        "affirmative",
        "是",
        "对",
        "对的",
        "没错",
    } or bool(tokens & {"yes", "yeah", "yep", "correct", "right", "是", "对"})


def _is_negative_response(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    tokens = {token.strip(".,!?;:") for token in normalized.split()}
    return normalized in {
        "no",
        "nope",
        "negative",
        "incorrect",
        "not correct",
        "不是",
        "不对",
        "错了",
        "不",
    } or bool(tokens & {"no", "nope", "incorrect", "不是", "不对", "错了"})


def _confirm_identity_candidate(
    *,
    stt,
    tts,
    music,
    status_leds,
    candidate: str,
    confirm_prompt_template: str,
    retry_prompt: str,
    should_interrupt=None,
    max_attempts: int = 2,
    voice_access_lock=None,
) -> bool:
    heard = " ".join(str(candidate or "").strip().split())
    if not heard:
        return False
    prompt = str(confirm_prompt_template or "").format(value=heard)
    for _attempt in range(max(1, int(max_attempts))):
        _speak_prompt(
            tts,
            music,
            status_leds,
            prompt,
            should_interrupt=should_interrupt,
            source="identity_confirm",
            expects_response=True,
            voice_access_lock=voice_access_lock,
        )
        response = _listen_with_stt(
            stt,
            music,
            status_leds,
            should_interrupt=should_interrupt,
            voice_access_lock=voice_access_lock,
        )
        if _is_affirmative_response(response):
            return True
        if _is_negative_response(response):
            return False
        prompt = retry_prompt
    return False


def _collect_confirmed_identity_field(
    *,
    stt,
    tts,
    music,
    status_leds,
    initial_prompt: str,
    retry_prompt: str,
    confirm_prompt_template: str,
    confirm_retry_prompt: str,
    reenter_prompt: str,
    normalizer=None,
    should_interrupt=None,
    max_capture_rounds: int = 3,
    voice_access_lock=None,
) -> str:
    prompt = initial_prompt
    for _round in range(max(1, int(max_capture_rounds))):
        candidate = _collect_identity_field(
            stt=stt,
            tts=tts,
            music=music,
            status_leds=status_leds,
            initial_prompt=prompt,
            retry_prompt=retry_prompt,
            should_interrupt=should_interrupt,
            max_attempts=2,
            voice_access_lock=voice_access_lock,
        )
        value = candidate
        if callable(normalizer):
            value = normalizer(candidate)
        value = " ".join(str(value or "").strip().split())
        if not value:
            prompt = retry_prompt
            continue
        if _confirm_identity_candidate(
            stt=stt,
            tts=tts,
            music=music,
            status_leds=status_leds,
            candidate=value,
            confirm_prompt_template=confirm_prompt_template,
            retry_prompt=confirm_retry_prompt,
            should_interrupt=should_interrupt,
            max_attempts=2,
            voice_access_lock=voice_access_lock,
        ):
            return value
        prompt = reenter_prompt
    return ""


def _refresh_runtime_user_dependent_state(stt) -> None:
    if hasattr(stt, "_emotion_side_channel"):
        stt._emotion_side_channel = None


def _run_user_intake(session_control, stt, tts, music, status_leds, voice_access_lock=None) -> tuple[str, str]:
    set_phase = getattr(session_control, "set_phase", None)
    if callable(set_phase):
        set_phase("user_intake")
    should_interrupt = getattr(session_control, "is_shutdown_requested", None)
    raw_user_id = _collect_confirmed_identity_field(
        stt=stt,
        tts=tts,
        music=music,
        status_leds=status_leds,
        initial_prompt=USER_ID_PROMPT,
        retry_prompt=USER_ID_RETRY_PROMPT,
        confirm_prompt_template=USER_ID_CONFIRM_PROMPT,
        confirm_retry_prompt=USER_ID_CONFIRM_RETRY_PROMPT,
        reenter_prompt=USER_ID_REENTER_PROMPT,
        normalizer=normalize_spoken_user_id,
        should_interrupt=should_interrupt,
        voice_access_lock=voice_access_lock,
    )
    normalized_user_id = normalize_spoken_user_id(raw_user_id)
    if not normalized_user_id:
        normalized_user_id = build_guest_user_id()
        _speak_prompt(
            tts,
            music,
            status_leds,
            USER_ID_FALLBACK_MESSAGE,
            should_interrupt=should_interrupt,
            source="identity",
            voice_access_lock=voice_access_lock,
        )
    display_name = _collect_confirmed_identity_field(
        stt=stt,
        tts=tts,
        music=music,
        status_leds=status_leds,
        initial_prompt=USER_NAME_PROMPT,
        retry_prompt=USER_NAME_RETRY_PROMPT,
        confirm_prompt_template=USER_NAME_CONFIRM_PROMPT,
        confirm_retry_prompt=USER_NAME_CONFIRM_RETRY_PROMPT,
        reenter_prompt=USER_NAME_REENTER_PROMPT,
        should_interrupt=should_interrupt,
        voice_access_lock=voice_access_lock,
    )
    return normalized_user_id, display_name


def _choose_resume_session(
    *,
    participant_id: str,
    session_control,
    stt,
    tts,
    music,
    status_leds,
    voice_access_lock=None,
) -> bool:
    should_interrupt = getattr(session_control, "is_shutdown_requested", None)
    prompt = RESUME_SESSION_PROMPT.format(value=participant_id)
    for _attempt in range(2):
        _speak_prompt(
            tts,
            music,
            status_leds,
            prompt,
            should_interrupt=should_interrupt,
            source="session_resume",
            expects_response=True,
            voice_access_lock=voice_access_lock,
        )
        response = _listen_with_stt(
            stt,
            music,
            status_leds,
            should_interrupt=should_interrupt,
            voice_access_lock=voice_access_lock,
        )
        normalized = str(response or "").strip().lower()
        tokens = {token.strip(".,!?;:") for token in normalized.split()}
        if normalized in {"resume", "continue", "resume session"} or tokens & {"resume", "continue"}:
            return True
        if normalized in {"new", "new session", "start new"} or tokens & {"new"}:
            return False
        prompt = RESUME_SESSION_RETRY_PROMPT
    return False


def _prepare_user_session(session_control, stt, tts, music, status_leds, intermission_runner, voice_access_lock=None):
    while True:
        raw_user_id, display_name = _run_user_intake(
            session_control,
            stt,
            tts,
            music,
            status_leds,
            voice_access_lock=voice_access_lock,
        )
        if not is_protected_participant(raw_user_id):
            break
        _speak_prompt(
            tts,
            music,
            status_leds,
            PROTECTED_ID_MESSAGE,
            source="identity",
            voice_access_lock=voice_access_lock,
        )

    resumable = find_resumable_session(raw_user_id)
    if resumable is not None and _choose_resume_session(
        participant_id=raw_user_id,
        session_control=session_control,
        stt=stt,
        tts=tts,
        music=music,
        status_leds=status_leds,
        voice_access_lock=voice_access_lock,
    ):
        context = replace(
            resumable,
            display_name=display_name or resumable.display_name,
            resumed=True,
        )
    else:
        if resumable is not None:
            update_session_status(
                resumable,
                "ABANDONED",
                abandonment_reason="participant chose a new session",
            )
        context = create_new_session(raw_user_id, display_name)

    activate_session_context(context)
    _refresh_runtime_user_dependent_state(stt)
    session_id = _reset_session_runtime(session_control, context.session_id)
    begin_intermission = getattr(intermission_runner, "begin_session", None)
    if callable(begin_intermission):
        begin_intermission(
            db_path=context.structured_log_db_path,
            results_json_path=context.intermission_results_json_path,
            session_id=context.session_id,
            resume=context.resumed,
        )
    monitor = get_active_status_monitor()
    if monitor is not None:
        setter = getattr(monitor, "set_session", None)
        if callable(setter):
            setter(session_id)
    logger.info(
        "Prepared new CaiTI session context: %s user=%s name=%s",
        session_id,
        context.participant_id,
        context.display_name or "",
    )
    acknowledgement = f"Thank you, {context.display_name}. Let's begin." if context.display_name else "Thank you. Let's begin."
    _speak_prompt(tts, music, status_leds, acknowledgement, source="system", voice_access_lock=voice_access_lock)
    return context


def _run_session(
    *,
    session_control,
) -> None:
    set_phase = getattr(session_control, "set_phase", None)
    if callable(set_phase):
        set_phase("loading")
    session_control.checkpoint("loading")
    session_control.mark_screening()
    HandlerRL(session_control=session_control).run()


def _cleanup_after_session_cycle(
    *,
    session_control,
    music,
    voice_idle,
    status_leds=None,
    status_monitor=None,
    intermission_runner=None,
    reset_for_next_session: bool = True,
) -> None:
    set_phase = getattr(session_control, "set_phase", None)
    if callable(set_phase):
        set_phase("cleanup")
    wait_for_voice_io_drain(voice_idle, timeout_sec=90.0)
    music.stop()
    end_intermission = getattr(intermission_runner, "end_session", None)
    if callable(end_intermission):
        end_intermission()
    reset_record_state()
    clear_emotion_session_state()
    reset_questioner_session_state()
    deactivate_session_context()
    if reset_for_next_session:
        reset_method = getattr(session_control, "reset_for_next_session", None)
        if callable(reset_method):
            reset_method()
        reset_leds = getattr(status_leds, "reset_for_idle", None)
        if callable(reset_leds):
            reset_leds()
        reset_monitor = getattr(status_monitor, "reset_for_idle", None)
        if callable(reset_monitor):
            reset_monitor()


def main():
    """
    Voice entry point for CaiTI.

    STT and TTS are local I/O adapters around the existing text pipeline:
    microphone/audio -> STT text -> CaiTI LLM/RL modules -> generated text -> TTS audio.
    """
    clear_system_poweroff_request()
    status_monitor = build_status_monitor()
    set_active_status_monitor(status_monitor)
    stt = build_stt()
    tts = build_tts()
    music = build_music()
    status_leds = build_status_led_controller(status_monitor=status_monitor)
    voice_access_lock = threading.RLock()
    intermission_runner = build_intermission_runner(
        stt=stt,
        primary_tts=tts,
        music=music,
        status_leds=status_leds,
        voice_access_lock=voice_access_lock,
    )
    session_control = build_session_control(status_monitor=status_monitor)
    set_start_callback = getattr(status_monitor, "set_start_session_callback", None)
    if callable(set_start_callback):
        set_start_callback(lambda: session_control.request_start("monitor"))
    restore_music_after_pause = threading.Event()
    shutdown_message_spoken = threading.Event()
    shutdown_voice_lock = threading.Lock()

    def speak_shutdown_once() -> bool:
        if shutdown_message_spoken.is_set():
            return True
        with shutdown_voice_lock:
            if shutdown_message_spoken.is_set():
                return True
            spoken = _speak_shutdown_message(tts, music, status_leds, voice_access_lock=voice_access_lock)
            if spoken is not False:
                shutdown_message_spoken.set()
                return True
        return False

    def handle_short_press() -> None:
        _handle_short_press_with_music(session_control, music, voice_idle, restore_music_after_pause)

    def handle_long_press() -> None:
        restore_music_after_pause.clear()
        session_control.handle_long_press()
        if session_control.is_shutdown_requested():
            music.stop()
            threading.Thread(target=speak_shutdown_once, name="caiti-shutdown-message", daemon=True).start()

    def handle_music_mode_press() -> None:
        cycle_mode = getattr(music, "cycle_mode", None)
        if not callable(cycle_mode):
            logger.info("Music mode button ignored: current music backend has no mode switch.")
            return
        cycle_mode()

    session_button = build_session_button_controller(
        handle_short_press,
        handle_long_press,
    )
    volume_buttons = build_volume_button_controller()
    music_mode_button = build_music_mode_button_controller(handle_music_mode_press)
    voice_idle = threading.Event()
    voice_idle.set()

    io_thread = threading.Thread(
        target=run_voice_io_loop,
        args=(stt, tts, music),
        kwargs={
            "activity_event": voice_idle,
            "status_leds": status_leds,
            "session_control": session_control,
            "intermission_runner": intermission_runner,
            "voice_access_lock": voice_access_lock,
        },
        daemon=True,
    )
    io_thread.start()
    logger.info("Voice I/O loop started.")
    persistent_loop = _persistent_app_loop_enabled()
    poweroff_requested = False
    poweroff_reason = ""

    try:
        if status_monitor.start():
            logger.info("Open CaiTI monitor at %s", status_monitor.url)
        status_leds.start()
        volume_buttons.start()
        if not persistent_loop:
            status_monitor.set_phase("waiting_start")
        music_mode_button.start()
        session_button_started = session_button.start()
        if not session_button_started and session_control.settings.enabled and not persistent_loop:
            logger.warning("Button unavailable; starting CaiTI without button gating.")
            session_control.request_start("session button unavailable")
        elif not session_button_started and session_control.settings.enabled and persistent_loop:
            logger.warning("Session button unavailable; use the monitor or scripts/caiti_control.py start.")
        if persistent_loop:
            session_control.set_phase("preloading")
            _preload_llm_runtime()
            _warm_up_tts(tts)
            _warm_up_intermission_tts(intermission_runner, tts)
            _warm_up_stt(stt)
        while True:
            logger.info("Waiting for button to start CaiTI.")
            if not _wait_for_session_start(session_control, status_monitor, persistent_loop):
                break
            try:
                _prepare_user_session(
                    session_control,
                    stt,
                    tts,
                    music,
                    status_leds,
                    intermission_runner,
                    voice_access_lock=voice_access_lock,
                )
            except VoiceInterrupted:
                if session_control.is_shutdown_requested():
                    raise SessionShutdownRequested("Session shutdown requested during user intake.")
                raise
            if not persistent_loop:
                if not _music_is_background(music):
                    music.start()
                _preload_llm_runtime()
                _warm_up_tts(tts)
                _warm_up_intermission_tts(intermission_runner, tts)
                _warm_up_stt(stt)
            else:
                if not _music_is_background(music):
                    music.start()
            _run_session(
                session_control=session_control,
            )
            complete_current_session()
            auto_poweroff = _should_auto_poweroff_after_session_complete()
            if auto_poweroff or persistent_loop:
                _cleanup_after_session_cycle(
                    session_control=session_control,
                    music=music,
                    voice_idle=voice_idle,
                    status_leds=status_leds,
                    status_monitor=status_monitor,
                    intermission_runner=intermission_runner,
                    reset_for_next_session=not auto_poweroff,
                )
            if auto_poweroff:
                poweroff_requested = True
                poweroff_reason = "session complete"
                break
            if not persistent_loop:
                break
    except SessionShutdownRequested:
        interrupt_current_session("session-button shutdown request")
        logger.info("Voice application closing after session-button shutdown request.")
        _wait_for_voice_idle(voice_idle)
        speak_shutdown_once()
        _cleanup_after_session_cycle(
            session_control=session_control,
            music=music,
            voice_idle=voice_idle,
            status_leds=status_leds,
            status_monitor=status_monitor,
            intermission_runner=intermission_runner,
            reset_for_next_session=False,
        )
        poweroff_requested = True
        poweroff_reason = "button shutdown"
    except Exception:
        interrupt_current_session("unexpected voice application failure")
        raise
    finally:
        session_control.mark_closing()
        try:
            if not session_control.is_shutdown_requested() and not poweroff_requested:
                wait_for_voice_io_drain(voice_idle, timeout_sec=90.0)
        except KeyboardInterrupt:
            logger.info("Voice application interrupted during voice I/O drain.")
        finally:
            music.stop()
            session_button.stop()
            music_mode_button.stop()
            volume_buttons.stop()
            status_leds.stop()
            status_monitor.stop()
            set_active_status_monitor(None)
            time.sleep(0.3)
    if poweroff_requested:
        if not _request_configured_system_poweroff(poweroff_reason):
            logger.warning(
                "System poweroff was requested but marker creation failed. reason=%s marker=%s",
                poweroff_reason,
                config_loader.SESSION_POWEROFF_REQUEST_PATH,
            )
            raise SystemExit(config_loader.SESSION_POWEROFF_REQUEST_FAILURE_EXIT_CODE)


if __name__ == "__main__":
    main()
