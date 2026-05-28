from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from scripts import faster_whisper_stt_command as whisper_stt
from src.emotion import build_emotion_side_channel
from src.utils.log_util import get_logger
from src.voice.exceptions import VoiceInterrupted
from src.voice.stt.recorder import (
    RecordingSettings,
    analyze_wav_file,
    cleanup_wav_file,
    make_wav_file,
    record_wav_file,
)

logger = get_logger("VoiceSTT")


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
    stt_timeout_sec: int = 120
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

    def _recording_settings(self) -> RecordingSettings:
        return RecordingSettings(
            record_seconds=self.record_seconds,
            sample_rate=self.sample_rate,
            channels=self.channels,
            audio_device=self.audio_device,
            auto_stop=self.auto_stop,
            vad_detector=self.vad_detector,
            vad_aggressiveness=self.vad_aggressiveness,
            silence_threshold_dbfs=self.silence_threshold_dbfs,
            silence_timeout_sec=self.silence_timeout_sec,
            trailing_pad_sec=self.trailing_pad_sec,
            min_speech_seconds=self.min_speech_seconds,
            min_record_seconds=self.min_record_seconds,
            no_speech_timeout_sec=self.no_speech_timeout_sec,
            vad_chunk_ms=self.vad_chunk_ms,
            timeout_sec=self.stt_timeout_sec,
        )

    def _record(self, wav_file: str) -> None:
        if self._should_interrupt():
            raise VoiceInterrupted("STT interrupted before recording.")
        record_wav_file(
            wav_file,
            self._recording_settings(),
            should_stop=self._should_interrupt,
        )
        if self._should_interrupt():
            raise VoiceInterrupted("STT interrupted during recording.")

    def _make_wav_file(self) -> tuple[str, bool]:
        return make_wav_file(self.save_wav)

    def _transcribe(self, wav_file: str) -> str:
        if self._should_interrupt():
            raise VoiceInterrupted("STT interrupted before transcription.")
        metrics = {}
        try:
            metrics = analyze_wav_file(wav_file)
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
            cleanup_wav_file(wav_file, cleanup)

    def listen_with_waiting_music(self, music) -> str:
        wav_file, cleanup = self._make_wav_file()
        try:
            self._record(wav_file)
            if music is not None:
                music.start()
            return self._transcribe(wav_file)
        finally:
            cleanup_wav_file(wav_file, cleanup)
