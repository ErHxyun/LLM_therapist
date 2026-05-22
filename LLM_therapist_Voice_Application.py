import threading
import time

from src.handler_rl import HandlerRL
from src.utils.io_record import init_record
from src.utils.log_util import get_logger
from src.voice.backends import build_stt, build_tts
from src.voice.io_loop import run_voice_io_loop

logger = get_logger("VoiceApplication")


def main():
    """
    Voice entry point for CaiTI.

    STT and TTS are local I/O adapters around the existing text pipeline:
    microphone/audio -> STT text -> CaiTI LLM/RL modules -> generated text -> TTS audio.
    """
    init_record()
    stt = build_stt()
    tts = build_tts()

    io_thread = threading.Thread(
        target=run_voice_io_loop,
        args=(stt, tts),
        daemon=True,
    )
    io_thread.start()
    logger.info("Voice I/O loop started.")

    HandlerRL().run()
    time.sleep(0.3)


if __name__ == "__main__":
    main()
