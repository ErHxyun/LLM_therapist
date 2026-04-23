"""Piper-based text-to-speech with a clear local-model contract."""
import os
import subprocess

from src.utils.config_loader import (
    TTS_EXECUTABLE,
    TTS_LENGTH_SCALE,
    TTS_MODEL_PATH,
    TTS_SENTENCE_SILENCE,
)
from src.utils.log_util import get_logger

logger = get_logger("TTSGenerator")


class TTSGenerator:
    def __init__(self):
        self.executable = TTS_EXECUTABLE
        self.model_path = TTS_MODEL_PATH

    def generate(self, text: str, output_file: str):
        if not text:
            return None
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Piper model not found at {self.model_path}. "
                "Place the .onnx model there or override TTS_MODEL_PATH."
            )

        cmd = [
            self.executable,
            "--model",
            self.model_path,
            "--length_scale",
            str(TTS_LENGTH_SCALE),
            "--sentence_silence",
            str(TTS_SENTENCE_SILENCE),
            "--output_file",
            output_file,
        ]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed: {proc.stderr.decode('utf-8', errors='ignore')}")
        if not os.path.exists(output_file):
            raise RuntimeError(f"Piper did not create output WAV: {output_file}")
        return output_file
