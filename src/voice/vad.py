"""Small voice-activity detectors for microphone auto-stop.

The voice shell keeps VAD separate from STT so recording policy stays easy to
tune without touching Faster-Whisper transcription.
"""

from __future__ import annotations

import audioop
import math
from dataclasses import dataclass
from typing import Protocol


SUPPORTED_WEBRTC_SAMPLE_RATES = {8000, 16000, 32000, 48000}
SUPPORTED_WEBRTC_FRAME_MS = {10, 20, 30}


class VoiceActivityDetector(Protocol):
    name: str

    def is_speech(self, frame: bytes) -> bool:
        """Return True when this PCM16 mono frame appears to contain speech."""


def dbfs(audio: bytes, sample_width: int = 2) -> float:
    if not audio:
        return -120.0
    max_amplitude = float((1 << (8 * sample_width - 1)) - 1)
    rms = float(audioop.rms(audio, sample_width))
    if rms <= 0 or max_amplitude <= 0:
        return -120.0
    return 20.0 * math.log10(rms / max_amplitude)


@dataclass
class EnergyVAD:
    threshold_dbfs: float = -45.0
    name: str = "energy"

    def is_speech(self, frame: bytes) -> bool:
        return dbfs(frame) >= self.threshold_dbfs


@dataclass
class WebRTCVAD:
    sample_rate: int
    aggressiveness: int = 2
    name: str = "webrtc"

    def __post_init__(self) -> None:
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "webrtcvad is not installed. Install it with: python -m pip install webrtcvad"
            ) from exc
        self._vad = webrtcvad.Vad(int(self.aggressiveness))

    def is_speech(self, frame: bytes) -> bool:
        try:
            return bool(self._vad.is_speech(frame, self.sample_rate))
        except Exception:
            return False


def can_use_webrtc(sample_rate: int, channels: int, chunk_ms: int) -> bool:
    return (
        int(sample_rate) in SUPPORTED_WEBRTC_SAMPLE_RATES
        and int(channels) == 1
        and int(chunk_ms) in SUPPORTED_WEBRTC_FRAME_MS
    )


def build_vad(
    detector: str = "auto",
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    chunk_ms: int = 30,
    aggressiveness: int = 2,
    threshold_dbfs: float = -45.0,
) -> VoiceActivityDetector:
    mode = str(detector or "auto").strip().lower()
    if mode not in {"auto", "webrtc", "energy"}:
        raise ValueError(f"Unsupported VAD detector: {detector}")

    webrtc_supported = can_use_webrtc(sample_rate, channels, chunk_ms)
    if mode in {"auto", "webrtc"} and webrtc_supported:
        try:
            return WebRTCVAD(sample_rate=sample_rate, aggressiveness=aggressiveness)
        except RuntimeError:
            if mode == "webrtc":
                raise

    if mode == "webrtc":
        raise ValueError(
            "WebRTC VAD requires mono PCM16 audio at 8/16/32/48 kHz with a 10/20/30 ms frame."
        )
    return EnergyVAD(threshold_dbfs=threshold_dbfs)
