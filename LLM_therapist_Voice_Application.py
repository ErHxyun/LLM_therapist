"""Minimal local voice runner for CaiTI using Gemma 4 + Faster-Whisper + Piper."""
import os
import tempfile
import threading

from src.drivers.audio_runtime import play_wav, record_until_silence
from src.handler_rl import HandlerRL
from src.models.stt import STTGenerator
from src.models.tts import TTSGenerator
from src.utils.io_record import INPUT_QUEUE, OUTPUT_QUEUE, init_record
from src.utils.log_util import get_logger

logger = get_logger("VoiceApp")


def _speaker_loop(tts: TTSGenerator):
    while True:
        text = OUTPUT_QUEUE.get()
        print(f"\nCAITI: {text}", flush=True)
        fd, wav_path = tempfile.mkstemp(prefix="caiti_tts_", suffix=".wav")
        os.close(fd)
        try:
            tts.generate(text, wav_path)
            play_wav(wav_path)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


def _listener_loop(stt: STTGenerator):
    while True:
        wav_path = None
        try:
            wav_path = record_until_silence()
            stt_payload = stt.transcribe(wav_path)
            print("USER(STT):", stt_payload, flush=True)
            INPUT_QUEUE.put(stt_payload)
        except KeyboardInterrupt:
            INPUT_QUEUE.put("stop")
            break
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)


def main():
    init_record()
    stt = STTGenerator()
    tts = TTSGenerator()

    threading.Thread(target=_speaker_loop, args=(tts,), daemon=True).start()
    threading.Thread(target=_listener_loop, args=(stt,), daemon=True).start()

    HandlerRL().run()


if __name__ == "__main__":
    main()
