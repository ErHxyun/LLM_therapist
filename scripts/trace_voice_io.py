"""Trace CaiTI voice I/O protocol without real audio hardware.

This script validates the record.csv bridge:
question text -> TTS block -> STT transcript -> record.csv response.
It does not call the LLM runtime and does not use microphone or speaker devices.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.voice.io_loop as voice_io
from src.utils.io_record import HEADER


@dataclass
class ScriptedSTT:
    transcript: str

    def listen(self) -> str:
        return self.transcript


@dataclass
class CapturingTTS:
    blocks: list[str] = field(default_factory=list)

    def speak(self, text: str) -> None:
        block = str(text or "").strip()
        if block:
            self.blocks.append(block)

    def speak_stream(self, text: str) -> None:
        self.speak(text)


def main() -> int:
    question = "Have your weight changed significantly recently? Please answer briefly."
    transcript = "My weight increased a lot recently."

    with tempfile.TemporaryDirectory() as tmpdir:
        record_path = str(Path(tmpdir) / "record.csv")
        pd.DataFrame([[question, 1, "", 1]], columns=HEADER).to_csv(record_path, index=False)

        original_record_csv = voice_io.RECORD_CSV
        voice_io.RECORD_CSV = record_path
        try:
            tts = CapturingTTS()
            processed = voice_io.process_voice_turn(ScriptedSTT(transcript), tts)
            result = pd.read_csv(record_path)
        finally:
            voice_io.RECORD_CSV = original_record_csv

    print("processed:", processed)
    print("tts_blocks:")
    for block in tts.blocks:
        print("-", block)
    print("record_resp:", result.loc[0, "Resp"])
    print("question_lock:", int(result.loc[0, "Question_Lock"]))
    print("resp_lock:", int(result.loc[0, "Resp_Lock"]))

    if not processed:
        return 1
    if result.loc[0, "Resp"] != transcript:
        return 1
    if int(result.loc[0, "Question_Lock"]) != 0 or int(result.loc[0, "Resp_Lock"]) != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
