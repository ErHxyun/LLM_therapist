import threading
import time

from src.handler_rl import HandlerRL
from src.hardware.session_button import build_session_button_controller
from src.hardware.status_leds import build_status_led_controller
from src.hardware.volume_buttons import build_volume_button_controller
from src.intermission import build_intermission_runner
from src.runtime.status_monitor import build_status_monitor
from src.session.control import SessionShutdownRequested, build_session_control
from src.utils.io_record import init_record
from src.utils.llm_client import preload_llm_runtime
from src.utils.log_util import get_logger
from src.voice.backends import VoiceInterrupted, build_stt, build_tts
from src.voice.io_loop import run_voice_io_loop, wait_for_voice_io_drain
from src.voice.music import build_music

logger = get_logger("VoiceApplication")
SHUTDOWN_SPOKEN_MESSAGE = "Okay, closing Caiti now."
BUTTON_SHUTDOWN_CONFIRMATION_PROMPT = (
    "Do you want to close Caiti now? Please say yes to close, say no to keep going, "
    "or press and hold the button again for three seconds."
)


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
    method = getattr(music, "duck", None)
    if callable(method):
        method()
        return
    _pause_music(music)


def _restore_music_volume(music) -> None:
    method = getattr(music, "restore_volume", None)
    if callable(method):
        method()


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
        _duck_music_for_tts(music)
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


def _preload_llm_runtime() -> None:
    logger.info("Preloading local CaiTI LLM runtime before first spoken turn.")
    preload_llm_runtime()
    logger.info("Local CaiTI LLM runtime ready.")


def main():
    """
    Voice entry point for CaiTI.

    STT and TTS are local I/O adapters around the existing text pipeline:
    microphone/audio -> STT text -> CaiTI LLM/RL modules -> generated text -> TTS audio.
    """
    init_record()
    status_monitor = build_status_monitor()
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

    session_button = build_session_button_controller(
        handle_short_press,
        handle_long_press,
    )
    volume_buttons = build_volume_button_controller()
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

    try:
        if status_monitor.start():
            logger.info("Open CaiTI monitor at %s", status_monitor.url)
        status_leds.start()
        volume_buttons.start()
        status_monitor.set_phase("waiting_start")
        session_button_started = session_button.start()
        if not session_button_started and session_control.settings.enabled:
            logger.warning("Button unavailable; starting CaiTI without button gating.")
            session_control.request_start("session button unavailable")
        logger.info("Waiting for button to start CaiTI.")
        if session_control.wait_for_start():
            music.start()
            _preload_llm_runtime()
            _warm_up_stt(stt)
            session_control.checkpoint("loading")
            session_control.mark_screening()
            HandlerRL(session_control=session_control).run()
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
            volume_buttons.stop()
            status_leds.stop()
            status_monitor.stop()
            time.sleep(0.3)


if __name__ == "__main__":
    main()
