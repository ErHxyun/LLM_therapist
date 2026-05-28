import subprocess
import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from scripts import faster_whisper_stt_command as whisper_stt
from src.emotion import build_emotion_side_channel
from src.utils.config_loader import (
    VOICE_STT_AUDIO_DEVICE,
    VOICE_STT_BACKEND,
    VOICE_STT_AUTO_STOP,
    VOICE_STT_BEAM_SIZE,
    VOICE_STT_BEST_OF,
    VOICE_STT_CHANNELS,
    VOICE_STT_COMMAND,
    VOICE_STT_COMPUTE_TYPE,
    VOICE_STT_DEBUG_AUDIO,
    VOICE_STT_DEVICE,
    VOICE_STT_INITIAL_PROMPT,
    VOICE_STT_LANGUAGE,
    VOICE_STT_MIN_SPEECH_SECONDS,
    VOICE_STT_MIN_RECORD_SECONDS,
    VOICE_STT_NO_SPEECH_TIMEOUT_SEC,
    VOICE_STT_RECORD_SECONDS,
    VOICE_STT_SAMPLE_RATE,
    VOICE_STT_SAVE_WAV,
    VOICE_STT_SILENCE_THRESHOLD_DBFS,
    VOICE_STT_SILENCE_TIMEOUT_SEC,
    VOICE_STT_TRAILING_PAD_SEC,
    VOICE_STT_TIMEOUT_SEC,
    VOICE_STT_VAD_AGGRESSIVENESS,
    VOICE_STT_VAD_CHUNK_MS,
    VOICE_STT_VAD_DETECTOR,
    VOICE_STT_VAD_FILTER,
    VOICE_STT_WHISPER_MODEL,
)
from src.utils.log_util import get_logger
from src.voice.exceptions import VoiceInterrupted
from src.voice.tts import CommandTTS, ConsoleTTS, build_primary_tts

logger = get_logger("VoiceBackends")


class STTBackend(Protocol):
    def listen(self) -> str:
        """Return one user utterance as text."""


class TTSBackend(Protocol):
    def speak(self, text: str) -> None:
        """Speak one text block."""

    def speak_stream(self, text: str) -> None:
        """Speak one complete text block."""


@dataclass
class ConsoleSTT:
    prompt: str = "Your answer: "

    def listen(self) -> str:
        try:
            return input(self.prompt).strip()
        except KeyboardInterrupt:
            raise


@dataclass
class CommandSTT:
    command: str
    timeout_sec: int = 120
    runner: Callable = subprocess.run

    def listen(self) -> str:
        if not self.command.strip():
            raise ValueError("CommandSTT requires a non-empty command.")
        completed = self.runner(
            self.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(f"STT command failed with code {completed.returncode}: {stderr}")
        transcript = (completed.stdout or "").strip()
        logger.info("STT command produced transcript length=%s", len(transcript))
        return transcript


@dataclass
class FasterWhisperSTT:
    model: str = "base.en"
    record_seconds: float = 30.0
    sample_rate: int = 16000
    channels: int = 1
    audio_device: str = ""
    stt_device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    best_of: int = 5
    language: str = "en"
    initial_prompt: str = ""
    vad_filter: bool = True
    auto_stop: bool = True
    vad_detector: str = "auto"
    vad_aggressiveness: int = 3
    silence_threshold_dbfs: float = -45.0
    silence_timeout_sec: float = 1.2
    trailing_pad_sec: float = 0.4
    min_speech_seconds: float = 0.25
    min_record_seconds: float = 1.0
    no_speech_timeout_sec: float = 5.0
    vad_chunk_ms: int = 30
    debug_audio: bool = False
    save_wav: str = ""
    _model: Any = field(default=None, init=False, repr=False)
    _model_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _interrupt_check: Callable[[], bool] | None = field(default=None, init=False, repr=False)
    _emotion_side_channel: Any = field(default=None, init=False, repr=False)
    last_audio_duration_sec: float = field(default=0.0, init=False)

    def set_interrupt_check(self, checker: Callable[[], bool] | None) -> None:
        self._interrupt_check = checker

    def _should_interrupt(self) -> bool:
        return bool(self._interrupt_check is not None and self._interrupt_check())

    def _get_emotion_side_channel(self):
        if self._emotion_side_channel is None:
            self._emotion_side_channel = build_emotion_side_channel()
        return self._emotion_side_channel

    def warm_up(self) -> None:
        self._get_model()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                logger.info(
                    "Loading persistent Faster-Whisper STT model %s on %s/%s",
                    self.model,
                    self.stt_device,
                    self.compute_type,
                )
                self._model = whisper_stt.load_whisper_model(
                    self.model,
                    device=self.stt_device,
                    compute_type=self.compute_type,
                )
                logger.info("Persistent Faster-Whisper STT model ready.")
        return self._model

    def _record(self, wav_file: str) -> None:
        if self._should_interrupt():
            raise VoiceInterrupted("STT interrupted before recording.")
        if self.auto_stop:
            whisper_stt.record_wav_auto_stop(
                wav_file,
                max_seconds=self.record_seconds,
                sample_rate=self.sample_rate,
                channels=self.channels,
                device=self.audio_device,
                vad_detector=self.vad_detector,
                vad_aggressiveness=self.vad_aggressiveness,
                silence_threshold_dbfs=self.silence_threshold_dbfs,
                silence_timeout_sec=self.silence_timeout_sec,
                trailing_pad_sec=self.trailing_pad_sec,
                min_speech_seconds=self.min_speech_seconds,
                min_record_seconds=self.min_record_seconds,
                no_speech_timeout_sec=self.no_speech_timeout_sec,
                chunk_ms=self.vad_chunk_ms,
                should_stop=self._should_interrupt,
            )
            if self._should_interrupt():
                raise VoiceInterrupted("STT interrupted during recording.")
            return

        whisper_stt.record_wav(
            wav_file,
            seconds=self.record_seconds,
            sample_rate=self.sample_rate,
            channels=self.channels,
            device=self.audio_device,
            timeout_sec=VOICE_STT_TIMEOUT_SEC,
        )
        if self._should_interrupt():
            raise VoiceInterrupted("STT interrupted during recording.")

    def _make_wav_file(self) -> tuple[str, bool]:
        if self.save_wav:
            os.makedirs(os.path.dirname(self.save_wav) or ".", exist_ok=True)
            return self.save_wav, False
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_file = tmp.name
        tmp.close()
        return wav_file, True

    def _transcribe(self, wav_file: str) -> str:
        if self._should_interrupt():
            raise VoiceInterrupted("STT interrupted before transcription.")
        metrics = {}
        try:
            metrics = whisper_stt.analyze_wav(wav_file)
            self.last_audio_duration_sec = float(metrics.get("duration_sec", 0.0))
        except Exception:
            self.last_audio_duration_sec = 0.0
        if self.debug_audio:
            logger.info(
                "STT audio duration=%.2fs rms=%.1fdBFS peak=%.1fdBFS",
                self.last_audio_duration_sec,
                metrics.get("rms_dbfs", -120.0),
                metrics.get("peak_dbfs", -120.0),
            )
        transcript = whisper_stt.transcribe_wav_with_model(
            self._get_model(),
            wav_file,
            beam_size=self.beam_size,
            best_of=self.best_of,
            language=self.language,
            initial_prompt=self.initial_prompt,
            vad_filter=self.vad_filter,
        )
        if self._should_interrupt():
            raise VoiceInterrupted("STT interrupted after transcription.")
        transcript = transcript.strip()
        logger.info("Persistent STT produced transcript length=%s", len(transcript))
        self._send_emotion_analysis(wav_file, transcript)
        return transcript

    def _send_emotion_analysis(self, wav_file: str, transcript: str) -> None:
        try:
            self._get_emotion_side_channel().analyze_async(
                audio_file_path=wav_file,
                transcript=transcript,
                sample_rate=self.sample_rate,
                duration_seconds=self.last_audio_duration_sec,
            )
        except Exception as exc:
            logger.warning("Emotion side-channel dispatch failed: %s", exc)

    def listen(self) -> str:
        wav_file, cleanup = self._make_wav_file()
        try:
            self._record(wav_file)
            return self._transcribe(wav_file)
        finally:
            if cleanup:
                try:
                    os.remove(wav_file)
                except OSError:
                    pass

    def listen_with_waiting_music(self, music) -> str:
        wav_file, cleanup = self._make_wav_file()
        try:
            self._record(wav_file)
            if music is not None:
                music.start()
            return self._transcribe(wav_file)
        finally:
            if cleanup:
                try:
                    os.remove(wav_file)
                except OSError:
                    pass


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
            min_record_seconds=VOICE_STT_MIN_RECORD_SECONDS,
            no_speech_timeout_sec=VOICE_STT_NO_SPEECH_TIMEOUT_SEC,
            vad_chunk_ms=VOICE_STT_VAD_CHUNK_MS,
            debug_audio=VOICE_STT_DEBUG_AUDIO,
            save_wav=VOICE_STT_SAVE_WAV,
        )
    raise ValueError(f"Unsupported STT backend: {VOICE_STT_BACKEND}")


def build_tts() -> TTSBackend:
    return build_primary_tts()
