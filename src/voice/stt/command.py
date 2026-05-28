from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

from src.utils.log_util import get_logger

logger = get_logger("VoiceSTT")


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
