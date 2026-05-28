from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Callable

from scripts import faster_whisper_stt_command as whisper_stt


@dataclass(frozen=True)
class RecordingSettings:
    record_seconds: float = 30.0
    sample_rate: int = 16000
    channels: int = 1
    audio_device: str = ""
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
    timeout_sec: int = 120


def make_wav_file(save_wav: str = "") -> tuple[str, bool]:
    if save_wav:
        os.makedirs(os.path.dirname(save_wav) or ".", exist_ok=True)
        return save_wav, False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_file = tmp.name
    tmp.close()
    return wav_file, True


def cleanup_wav_file(wav_file: str, cleanup: bool) -> None:
    if not cleanup:
        return
    try:
        os.remove(wav_file)
    except OSError:
        pass


def record_wav_file(
    wav_file: str,
    settings: RecordingSettings,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    if settings.auto_stop:
        whisper_stt.record_wav_auto_stop(
            wav_file,
            max_seconds=settings.record_seconds,
            sample_rate=settings.sample_rate,
            channels=settings.channels,
            device=settings.audio_device,
            vad_detector=settings.vad_detector,
            vad_aggressiveness=settings.vad_aggressiveness,
            silence_threshold_dbfs=settings.silence_threshold_dbfs,
            silence_timeout_sec=settings.silence_timeout_sec,
            trailing_pad_sec=settings.trailing_pad_sec,
            min_speech_seconds=settings.min_speech_seconds,
            min_record_seconds=settings.min_record_seconds,
            no_speech_timeout_sec=settings.no_speech_timeout_sec,
            chunk_ms=settings.vad_chunk_ms,
            should_stop=should_stop,
        )
        return

    whisper_stt.record_wav(
        wav_file,
        seconds=settings.record_seconds,
        sample_rate=settings.sample_rate,
        channels=settings.channels,
        device=settings.audio_device,
        timeout_sec=settings.timeout_sec,
    )


def analyze_wav_file(wav_file: str) -> dict[str, float]:
    return whisper_stt.analyze_wav(wav_file)
