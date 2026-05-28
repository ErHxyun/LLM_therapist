"""Shared types for local CaiTI LLM inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LLMTask(str, Enum):
    """Fixed CaiTI LLM task names."""

    BASE = "base"
    TASK1_RESPONSE_ANALYZER = "task1_response_analyzer"
    TASK2_GENERAL_RESPONSE = "task2_general_response"
    TASK3_RV_REASONER = "task3_rv_reasoner"
    TASK4_CBT_STAGE1 = "task4_cbt_stage1"
    TASK4_CBT_STAGE2 = "task4_cbt_stage2"
    TASK4_CBT_STAGE3 = "task4_cbt_stage3"


@dataclass(frozen=True)
class GenerationConfig:
    """Generation settings that cross the runtime boundary."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = False
    use_chat_template: bool = True
    max_input_tokens: int = 2048
    stop_regex: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Raw model result plus routing metadata for logging/review."""

    text: str
    task: LLMTask
    adapter: str | None
    raw_text: str
