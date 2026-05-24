import threading
import time

from src.handler_rl import HandlerRL
from src.hardware.session_button import build_session_button_controller
from src.hardware.status_leds import build_status_led_controller
from src.hardware.volume_buttons import build_volume_button_controller
from src.runtime.status_monitor import build_status_monitor
from src.session.control import SessionShutdownRequested, build_session_control
from src.utils.io_record import init_record
from src.utils.llm_client import preload_llm_runtime
from src.utils.log_util import get_logger
from src.voice.backends import build_stt, build_tts
from src.voice.io_loop import run_voice_io_loop, wait_for_voice_io_drain
from src.voice.music import build_music

logger = get_logger("VoiceApplication")


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
    session_control = build_session_control(status_monitor=status_monitor)
    session_button = build_session_button_controller(
        session_control.handle_short_press,
        session_control.handle_long_press,
    )
    status_leds = build_status_led_controller(status_monitor=status_monitor)
    volume_buttons = build_volume_button_controller()
    voice_idle = threading.Event()
    voice_idle.set()

    io_thread = threading.Thread(
        target=run_voice_io_loop,
        args=(stt, tts, music),
        kwargs={"activity_event": voice_idle, "status_leds": status_leds},
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
            logger.warning("Large session button unavailable; starting CaiTI without button gating.")
            session_control.request_start("session button unavailable")
        logger.info("Waiting for large session button to start CaiTI.")
        if session_control.wait_for_start():
            music.start()
            _preload_llm_runtime()
            _warm_up_stt(stt)
            session_control.mark_screening()
            HandlerRL(session_control=session_control).run()
    except SessionShutdownRequested:
        logger.info("Voice application closing after session-button shutdown request.")
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
