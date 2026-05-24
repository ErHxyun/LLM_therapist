"""Run CaiTI's CBT flow directly through the local voice shell.

This skips screening and preloads a single score=2 dimension so real STT/TTS
can exercise CBT stage 0-3 without waiting for a full 37-dimension session.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CAITI_DEVICE_MAP", "cuda:0")

from src.CBT import run_cbt  # noqa: E402
from src.hardware.session_button import build_session_button_controller  # noqa: E402
from src.hardware.status_leds import build_status_led_controller  # noqa: E402
from src.hardware.volume_buttons import build_volume_button_controller  # noqa: E402
from src.session.control import SessionShutdownRequested, build_session_control  # noqa: E402
from src.utils.io_record import init_record  # noqa: E402
from src.utils.llm_client import preload_llm_runtime  # noqa: E402
from src.utils.log_util import get_logger  # noqa: E402
from src.voice.backends import build_stt, build_tts  # noqa: E402
from src.voice.io_loop import run_voice_io_loop, wait_for_voice_io_drain  # noqa: E402
from src.voice.music import build_music  # noqa: E402

logger = get_logger("VoiceCBTSmoke")


def _preload_llm_runtime() -> None:
    logger.info("Preloading local CaiTI LLM runtime before direct CBT smoke.")
    preload_llm_runtime()
    logger.info("Local CaiTI LLM runtime ready.")


def _warm_up_stt(stt) -> None:
    warm_up = getattr(stt, "warm_up", None)
    if not callable(warm_up):
        return
    try:
        warm_up()
    except Exception as exc:
        logger.warning("STT warm-up failed; first listen will retry. error=%s", exc)


def _seed_question_lib() -> dict:
    return {
        "1": {
            "1": {
                "label": "sleep",
                "name": "Maintaining a Regular Sleep Schedule",
                "question": ["How has your sleep been recently, and do you have a regular sleep schedule?"],
                "score": [2],
                "notes": [
                    [
                        "original_resp: No, I do not have a regular sleeping schedule.",
                        "followup_resp: No, I mean, I do not have a regular sleeping schedule. My sleep time is very short.",
                        "rv_decision: 0",
                        "rv_validation: Your short and irregular sleep connects to the sleep concern you shared.",
                    ]
                ],
                "Yes": 0,
                "No": 2,
            }
        }
    }


def main() -> int:
    init_record()
    stt = build_stt()
    tts = build_tts()
    music = build_music()
    session_control = build_session_control()
    session_button = build_session_button_controller(
        session_control.handle_short_press,
        session_control.handle_long_press,
    )
    status_leds = build_status_led_controller()
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
    logger.info("Voice I/O loop started for direct CBT smoke.")

    question_lib = _seed_question_lib()

    try:
        status_leds.start()
        volume_buttons.start()
        session_button_started = session_button.start()
        if not session_button_started and session_control.settings.enabled:
            logger.warning("Large session button unavailable; starting direct CBT smoke without button gating.")
            session_control.request_start("session button unavailable")
        logger.info("Waiting for large session button to start direct CBT smoke.")
        if session_control.wait_for_start():
            music.start()
            _preload_llm_runtime()
            _warm_up_stt(stt)
            session_control.mark_cbt()
            run_cbt(question_lib, session_control=session_control)
    except SessionShutdownRequested:
        logger.info("Direct CBT smoke closing after session-button shutdown request.")
    except KeyboardInterrupt:
        logger.info("Direct CBT smoke interrupted.")
        return 130
    finally:
        session_control.mark_closing()
        try:
            if not session_control.is_shutdown_requested():
                wait_for_voice_io_drain(voice_idle, timeout_sec=90.0)
        except KeyboardInterrupt:
            logger.info("Direct CBT smoke interrupted during voice I/O drain.")
        finally:
            music.stop()
            session_button.stop()
            volume_buttons.stop()
            status_leds.stop()
            time.sleep(0.3)

    note = question_lib["1"]["1"].get("notes", [])[-1]
    logger.info("Final CBT note: %s", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
