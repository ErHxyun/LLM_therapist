import threading
import time
import os

from src.handler_rl import HandlerRL
from src.emotion import clear_emotion_session_state
from src.hardware.music_mode_button import build_music_mode_button_controller
from src.hardware.session_button import build_session_button_controller
from src.hardware.status_leds import build_status_led_controller
from src.hardware.volume_buttons import build_volume_button_controller
from src.intermission import build_intermission_runner
from src.questioner import reset_questioner_session_state
from src.runtime.user_context import activate_user_context, build_guest_user_id, normalize_spoken_user_id
from src.runtime.status_monitor import build_status_monitor, get_active_status_monitor, set_active_status_monitor
from src.session.control import SessionShutdownRequested, build_session_control
from src.utils.io_record import reset_record_state
from src.utils.llm_client import preload_llm_runtime
from src.utils.log_util import get_logger
from src.utils.session_event_logger import build_session_id, set_session_id
from src.voice.backends import VoiceInterrupted, build_stt, build_tts
from src.voice.io_loop import run_voice_io_loop, wait_for_voice_io_drain
from src.voice.music import build_music

logger = get_logger("VoiceApplication")
SHUTDOWN_SPOKEN_MESSAGE = "Okay, closing Caiti now."
BUTTON_SHUTDOWN_CONFIRMATION_PROMPT = (
    "Do you want to close Caiti now? Please say yes to close, say no to keep going, "
    "or press and hold the button again for three seconds."
)
USER_ID_PROMPT = "Before we begin, please say your participant ID."
USER_ID_RETRY_PROMPT = "I did not catch the participant ID. Please say the participant ID again."
USER_ID_FALLBACK_MESSAGE = "I did not catch an ID, so I will use a temporary guest ID for this session."
USER_ID_CONFIRM_PROMPT = "I heard your participant ID as {value}. Is that correct? Please say yes or no."
USER_ID_CONFIRM_RETRY_PROMPT = "Please say yes if that participant ID is correct, or say no if you want to try again."
USER_ID_REENTER_PROMPT = "Okay, please say your participant ID again."
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
) -> None:
    text = str(message or "").strip()
    if not text:
        return
    try:
        _set_interrupt_check(tts, should_interrupt)
        _duck_music_for_tts(music)
        _set_status_led(status_leds, "set_tts_active", True)
        tts.speak(text)
    except VoiceInterrupted:
        logger.info("Shutdown message interrupted.")
    except Exception as exc:
        logger.warning("Failed to speak shutdown message: %s", exc)
    finally:
        _set_status_led(status_leds, "set_tts_active", False)
        _set_interrupt_check(tts, None)


def _speak_shutdown_confirmation_prompt(session_control, tts, music, status_leds) -> None:
    should_interrupt = getattr(session_control, "is_shutdown_requested", None)
    if not callable(should_interrupt):
        should_interrupt = None
    _speak_shutdown_message(
        tts,
        music,
        status_leds,
        BUTTON_SHUTDOWN_CONFIRMATION_PROMPT,
        should_interrupt=should_interrupt,
    )


def _listen_for_shutdown_confirmation(session_control, stt, music, status_leds) -> str:
    try:
        _set_interrupt_check(stt, session_control.is_shutdown_requested)
        _prepare_music_for_stt(music)
        _set_status_led(status_leds, "set_stt_active", True)
        return str(stt.listen() or "").strip()
    except VoiceInterrupted:
        logger.info("Shutdown confirmation listening interrupted.")
        return ""
    except Exception as exc:
        logger.warning("Shutdown confirmation STT failed: %s", exc)
        return ""
    finally:
        _set_status_led(status_leds, "set_stt_active", False)
        _restore_music_volume(music)
        _set_interrupt_check(stt, None)


def _run_shutdown_confirmation(session_control, stt, tts, music, status_leds, voice_idle, speak_shutdown_once) -> None:
    if not session_control.begin_shutdown_confirmation():
        return
    _wait_for_voice_idle(voice_idle)
    _speak_shutdown_confirmation_prompt(session_control, tts, music, status_leds)
    if session_control.is_shutdown_requested():
        speak_shutdown_once()
        return
    response = _listen_for_shutdown_confirmation(session_control, stt, music, status_leds)
    result = session_control.handle_shutdown_confirmation_response(response)
    if result == "shutdown":
        music.stop()
        speak_shutdown_once()
        return
    if result == "cancelled":
        _speak_shutdown_message(
            tts,
            music,
            status_leds,
            getattr(session_control.settings, "close_cancel_message", "Okay, we will keep going."),
        )


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


def _reset_session_runtime(session_control, subject_id: str | None = None) -> str:
    session_id = build_session_id(subject_id)
    set_session_id(session_id)
    reset_record_state()
    reset_questioner_session_state()
    clear_emotion_session_state()
    return session_id


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


def _speak_prompt(
    tts,
    music,
    status_leds,
    message: str,
    should_interrupt=None,
    source: str = "system",
    expects_response: bool = False,
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
    )


def _listen_with_stt(stt, music, status_leds, should_interrupt=None) -> str:
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
        )
        response = _listen_with_stt(stt, music, status_leds, should_interrupt=should_interrupt)
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
        )
        response = _listen_with_stt(stt, music, status_leds, should_interrupt=should_interrupt)
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
        ):
            return value
        prompt = reenter_prompt
    return ""


def _refresh_runtime_user_dependent_state(stt) -> None:
    if hasattr(stt, "_emotion_side_channel"):
        stt._emotion_side_channel = None


def _run_user_intake(session_control, stt, tts, music, status_leds) -> tuple[str, str]:
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
    )
    normalized_user_id = normalize_spoken_user_id(raw_user_id)
    if not normalized_user_id:
        normalized_user_id = build_guest_user_id()
        _speak_prompt(tts, music, status_leds, USER_ID_FALLBACK_MESSAGE, should_interrupt=should_interrupt, source="identity")
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
    )
    return normalized_user_id, display_name


def _prepare_user_session(session_control, stt, tts, music, status_leds) -> str:
    raw_user_id, display_name = _run_user_intake(session_control, stt, tts, music, status_leds)
    context = activate_user_context(raw_user_id, display_name)
    _refresh_runtime_user_dependent_state(stt)
    session_id = _reset_session_runtime(session_control, subject_id=context.subject_id)
    monitor = get_active_status_monitor()
    if monitor is not None:
        setter = getattr(monitor, "set_session", None)
        if callable(setter):
            setter(session_id)
    logger.info(
        "Prepared new CaiTI session context: %s user=%s name=%s",
        session_id,
        context.subject_id,
        context.display_name or "",
    )
    acknowledgement = f"Thank you, {context.display_name}. Let's begin." if context.display_name else "Thank you. Let's begin."
    _speak_prompt(tts, music, status_leds, acknowledgement, source="system")
    return session_id


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
) -> None:
    set_phase = getattr(session_control, "set_phase", None)
    if callable(set_phase):
        set_phase("cleanup")
    wait_for_voice_io_drain(voice_idle, timeout_sec=90.0)
    music.stop()
    reset_method = getattr(session_control, "reset_for_next_session", None)
    if callable(reset_method):
        reset_method()


def main():
    """
    Voice entry point for CaiTI.

    STT and TTS are local I/O adapters around the existing text pipeline:
    microphone/audio -> STT text -> CaiTI LLM/RL modules -> generated text -> TTS audio.
    """
    status_monitor = build_status_monitor()
    set_active_status_monitor(status_monitor)
    stt = build_stt()
    tts = build_tts()
    music = build_music()
    status_leds = build_status_led_controller(status_monitor=status_monitor)
    intermission_runner = build_intermission_runner(
        stt=stt,
        primary_tts=tts,
        music=music,
        status_leds=status_leds,
    )
    session_control = build_session_control(status_monitor=status_monitor)
    restore_music_after_pause = threading.Event()
    shutdown_message_spoken = threading.Event()
    shutdown_voice_lock = threading.Lock()

    def speak_shutdown_once() -> None:
        if shutdown_message_spoken.is_set():
            return
        shutdown_message_spoken.set()
        with shutdown_voice_lock:
            _speak_shutdown_message(tts, music, status_leds)

    def handle_short_press() -> None:
        _handle_short_press_with_music(session_control, music, voice_idle, restore_music_after_pause)

    def handle_long_press() -> None:
        restore_music_after_pause.clear()
        event = session_control.handle_long_press()
        if event == "shutdown_confirmation":
            threading.Thread(
                target=_run_shutdown_confirmation,
                args=(session_control, stt, tts, music, status_leds, voice_idle, speak_shutdown_once),
                name="caiti-shutdown-confirmation",
                daemon=True,
            ).start()
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
        },
        daemon=True,
    )
    io_thread.start()
    logger.info("Voice I/O loop started.")
    persistent_loop = _persistent_app_loop_enabled()

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
            logger.warning("Button unavailable while persistent loop is enabled; CaiTI will stay idle.")
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
            _prepare_user_session(session_control, stt, tts, music, status_leds)
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
            if not persistent_loop:
                break
            _cleanup_after_session_cycle(
                session_control=session_control,
                music=music,
                voice_idle=voice_idle,
            )
    except SessionShutdownRequested:
        logger.info("Voice application closing after session-button shutdown request.")
        _wait_for_voice_idle(voice_idle)
        speak_shutdown_once()
    finally:
        session_control.mark_closing()
        try:
            if not session_control.is_shutdown_requested():
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


if __name__ == "__main__":
    main()
