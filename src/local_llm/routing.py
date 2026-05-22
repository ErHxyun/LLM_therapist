"""Adapter routing for CaiTI task-specific local inference."""

from __future__ import annotations

from types import MappingProxyType

from src.local_llm.types import LLMTask


TASK_TO_ADAPTER = MappingProxyType(
    {
        LLMTask.TASK1_RESPONSE_ANALYZER: "adapters/task1_response_analyzer",
        LLMTask.TASK2_GENERAL_RESPONSE: "adapters/task2_general_response",
        LLMTask.TASK3_RV_REASONER: "adapters/task3_rv_reasoner",
        LLMTask.TASK4_CBT_STAGE1: "adapters/task4_cbt_stage1",
        LLMTask.TASK4_CBT_STAGE2: "adapters/task4_cbt_stage2",
        LLMTask.TASK4_CBT_STAGE3: "adapters/task4_cbt_stage3",
    }
)


def is_adapter_task(task: LLMTask) -> bool:
    """Return true when a task is served by a LoRA adapter."""

    return task in TASK_TO_ADAPTER


def resolve_adapter(task: LLMTask) -> str:
    """Return the adapter subdirectory for an adapter-backed task."""

    if task == LLMTask.BASE:
        raise ValueError("BASE task does not use an adapter")
    try:
        return TASK_TO_ADAPTER[task]
    except KeyError as exc:
        raise ValueError(f"Unknown adapter task: {task}") from exc

