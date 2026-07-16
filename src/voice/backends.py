from __future__ import annotations

from typing import Protocol

from scripts import faster_whisper_stt_command as whisper_stt
from src.utils.config_loader import (
    VOICE_STT_AUDIO_DEVICE,
    VOICE_STT_AUTO_STOP,
    VOICE_STT_BACKEND,
    VOICE_STT_BEAM_SIZE,
    VOICE_STT_BEST_OF,
    VOICE_STT_CHANNELS,
    VOICE_STT_COMMAND,
    VOICE_STT_COMPUTE_TYPE,
    VOICE_STT_DEBUG_AUDIO,
    VOICE_STT_DEVICE,
    VOICE_STT_INITIAL_PROMPT,
    VOICE_STT_LONG_RESPONSE_SILENCE_TIMEOUT_SEC,
    VOICE_STT_LANGUAGE,
    VOICE_STT_MIN_RECORD_SECONDS,
    VOICE_STT_MIN_SPEECH_SECONDS,
    VOICE_STT_NO_SPEECH_TIMEOUT_SEC,
    VOICE_STT_RECORD_SECONDS,
    VOICE_STT_SAMPLE_RATE,
    VOICE_STT_SAVE_WAV,
    VOICE_STT_SILENCE_THRESHOLD_DBFS,
    VOICE_STT_SILENCE_TIMEOUT_SEC,
    VOICE_STT_TIMEOUT_SEC,
    VOICE_STT_TRAILING_PAD_SEC,
    VOICE_STT_VAD_AGGRESSIVENESS,
    VOICE_STT_VAD_CHUNK_MS,
    VOICE_STT_VAD_DETECTOR,
    VOICE_STT_VAD_FILTER,
    VOICE_STT_WHISPER_MODEL,
)
from src.voice.exceptions import VoiceInterrupted
from src.voice.stt import CommandSTT, ConsoleSTT, FasterWhisperSTT
from src.voice.tts import CommandTTS, ConsoleTTS, build_primary_tts


class STTBackend(Protocol):
    def listen(self) -> str:
        """Return one user utterance as text."""


class TTSBackend(Protocol):
    def speak(self, text: str) -> None:
        """Speak one text block."""

    def speak_stream(self, text: str) -> None:
        """Speak one complete text block."""


def build_stt() -> STTBackend:
    backend = VOICE_STT_BACKEND.strip().lower()
    if backend == "console":
        return ConsoleSTT()
    if backend == "command":
        return CommandSTT(VOICE_STT_COMMAND, VOICE_STT_TIMEOUT_SEC)
    if backend in {"faster_whisper", "faster-whisper", "local_whisper", "local-whisper"}:
        return FasterWhisperSTT(
            model=VOICE_STT_WHISPER_MODEL,
            record_seconds=VOICE_STT_RECORD_SECONDS,
            sample_rate=VOICE_STT_SAMPLE_RATE,
            channels=VOICE_STT_CHANNELS,
            audio_device=VOICE_STT_AUDIO_DEVICE,
            stt_device=VOICE_STT_DEVICE,
            compute_type=VOICE_STT_COMPUTE_TYPE,
            beam_size=VOICE_STT_BEAM_SIZE,
            best_of=VOICE_STT_BEST_OF,
            language=VOICE_STT_LANGUAGE,
            initial_prompt=VOICE_STT_INITIAL_PROMPT,
            vad_filter=VOICE_STT_VAD_FILTER,
            auto_stop=VOICE_STT_AUTO_STOP,
            vad_detector=VOICE_STT_VAD_DETECTOR,
            vad_aggressiveness=VOICE_STT_VAD_AGGRESSIVENESS,
            silence_threshold_dbfs=VOICE_STT_SILENCE_THRESHOLD_DBFS,
            silence_timeout_sec=VOICE_STT_SILENCE_TIMEOUT_SEC,
            trailing_pad_sec=VOICE_STT_TRAILING_PAD_SEC,
            min_speech_seconds=VOICE_STT_MIN_SPEECH_SECONDS,
            long_response_silence_timeout_sec=VOICE_STT_LONG_RESPONSE_SILENCE_TIMEOUT_SEC,
            min_record_seconds=VOICE_STT_MIN_RECORD_SECONDS,
            no_speech_timeout_sec=VOICE_STT_NO_SPEECH_TIMEOUT_SEC,
            vad_chunk_ms=VOICE_STT_VAD_CHUNK_MS,
            stt_timeout_sec=VOICE_STT_TIMEOUT_SEC,
            debug_audio=VOICE_STT_DEBUG_AUDIO,
            save_wav=VOICE_STT_SAVE_WAV,
        )
    raise ValueError(f"Unsupported STT backend: {VOICE_STT_BACKEND}")


def build_tts() -> TTSBackend:
    return build_primary_tts()
