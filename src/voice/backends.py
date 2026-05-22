import subprocess
from dataclasses import dataclass
from typing import Callable, Protocol

from src.utils.config_loader import (
    VOICE_STT_BACKEND,
    VOICE_STT_COMMAND,
    VOICE_STT_TIMEOUT_SEC,
    VOICE_TTS_BACKEND,
    VOICE_TTS_COMMAND,
    VOICE_TTS_TIMEOUT_SEC,
)
from src.utils.log_util import get_logger
from src.voice.sentence_stream import iter_tts_chunks

logger = get_logger("VoiceBackends")


class STTBackend(Protocol):
    def listen(self) -> str:
        """Return one user utterance as text."""


class TTSBackend(Protocol):
    def speak(self, text: str) -> None:
        """Speak one text chunk."""

    def speak_stream(self, text: str) -> None:
        """Speak text in sentence-sized chunks."""


@dataclass
class ConsoleSTT:
    prompt: str = "Your answer: "

    def listen(self) -> str:
        try:
            return input(self.prompt).strip()
        except KeyboardInterrupt:
            raise


@dataclass
class ConsoleTTS:
    prefix: str = "AI"

    def speak(self, text: str) -> None:
        chunk = str(text or "").strip()
        if chunk:
            print(f"[{self.prefix}] {chunk}", flush=True)

    def speak_stream(self, text: str) -> None:
        for chunk in iter_tts_chunks(text):
            self.speak(chunk)


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
class CommandTTS:
    command: str
    timeout_sec: int = 60
    runner: Callable = subprocess.run

    def speak(self, text: str) -> None:
        chunk = str(text or "").strip()
        if not chunk:
            return
        if not self.command.strip():
            raise ValueError("CommandTTS requires a non-empty command.")
        completed = self.runner(
            self.command,
            input=chunk,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise RuntimeError(f"TTS command failed with code {completed.returncode}: {stderr}")
        logger.info("TTS command spoke chunk length=%s", len(chunk))

    def speak_stream(self, text: str) -> None:
        for chunk in iter_tts_chunks(text):
            self.speak(chunk)


def build_stt() -> STTBackend:
    backend = VOICE_STT_BACKEND.strip().lower()
    if backend == "console":
        return ConsoleSTT()
    if backend == "command":
        return CommandSTT(VOICE_STT_COMMAND, VOICE_STT_TIMEOUT_SEC)
    raise ValueError(f"Unsupported STT backend: {VOICE_STT_BACKEND}")


def build_tts() -> TTSBackend:
    backend = VOICE_TTS_BACKEND.strip().lower()
    if backend == "console":
        return ConsoleTTS()
    if backend == "command":
        return CommandTTS(VOICE_TTS_COMMAND, VOICE_TTS_TIMEOUT_SEC)
    raise ValueError(f"Unsupported TTS backend: {VOICE_TTS_BACKEND}")
