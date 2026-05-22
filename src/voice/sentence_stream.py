import re
from typing import Iterable, List


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*")


def split_for_tts(text: str) -> List[str]:
    """Split generated CaiTI text into stable sentence-sized TTS chunks."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks = []
    for part in _SENTENCE_BOUNDARY.split(normalized):
        part = part.strip()
        if part:
            chunks.append(part)
    return chunks or [normalized]


def iter_tts_chunks(text: str) -> Iterable[str]:
    yield from split_for_tts(text)
