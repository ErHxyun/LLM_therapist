"""Small cross-platform audio helpers for the CaiTI voice loop."""
import os
import subprocess
import sys
import tempfile
import time
import wave

from src.utils.config_loader import (
    AUDIO_CHANNELS,
    AUDIO_CHUNK_SIZE,
    AUDIO_MAX_RECORD_SEC,
    AUDIO_SAMPLE_RATE,
    AUDIO_SILENCE_TIMEOUT_SEC,
    AUDIO_VAD_AGGRESSIVENESS,
)
from src.utils.log_util import get_logger

logger = get_logger("AudioRuntime")


def play_wav(wav_path: str):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    if sys.platform.startswith("win"):
        import winsound

        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        return

    player = "afplay" if sys.platform == "darwin" else "aplay"
    subprocess.run([player, wav_path], check=True)


def record_until_silence(output_path: str | None = None) -> str:
    try:
        import pyaudio
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError(
            "Voice capture requires pyaudio and webrtcvad. Install the voice dependencies first."
        ) from exc

    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="caiti_user_", suffix=".wav")
        os.close(fd)

    vad = webrtcvad.Vad(AUDIO_VAD_AGGRESSIVENESS)
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=AUDIO_CHANNELS,
        rate=AUDIO_SAMPLE_RATE,
        input=True,
        frames_per_buffer=AUDIO_CHUNK_SIZE,
    )

    logger.info("Listening for user speech...")
    frames = []
    heard_speech = False
    silence_started_at = None
    deadline = time.time() + AUDIO_MAX_RECORD_SEC

    try:
        while time.time() < deadline:
            chunk = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
            frames.append(chunk)
            is_speech = vad.is_speech(chunk, AUDIO_SAMPLE_RATE)

            if is_speech:
                heard_speech = True
                silence_started_at = None
            elif heard_speech:
                silence_started_at = silence_started_at or time.time()
                if time.time() - silence_started_at >= AUDIO_SILENCE_TIMEOUT_SEC:
                    break
    finally:
        stream.stop_stream()
        stream.close()
        sample_width = pa.get_sample_size(pyaudio.paInt16)
        pa.terminate()

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(AUDIO_CHANNELS)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(AUDIO_SAMPLE_RATE)
        wav_file.writeframes(b"".join(frames))

    logger.info("Captured microphone audio to %s", output_path)
    return output_path
