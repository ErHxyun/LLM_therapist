"""Smoke test all CaiTI task adapters against the configured local model.

This script performs real local inference. It does not mock the model.
It is intended for quick checks after installing the Hugging Face bundle
and runtime dependencies on the target machine.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import CBT, reflection_validation, response_analyzer
from src.local_llm.routing import resolve_adapter
from src.local_llm.types import LLMTask
from src.utils.llm_client import llm_complete_task
from src.utils.llm_output_contracts import (
    normalize_decision_output,
    normalize_task1_output,
    normalize_task2_output,
)


@dataclass(frozen=True)
class SmokeCase:
    name: str
    task: LLMTask
    system_prompt: str
    payload: str
    max_new_tokens: int
    normalize: Callable[[str], str]
    validate: Callable[[str], tuple[bool, str]]


def _normalize_task1(raw: str) -> str:
    return normalize_task1_output(raw).normalized_output


def _validate_task1(raw: str) -> tuple[bool, str]:
    contract = normalize_task1_output(raw)
    if contract.is_valid:
        return True, "valid Dimension, Score"
    return False, "task1 did not produce a valid 37-dimension label with score 0/1/2"


def _normalize_task2(raw: str) -> str:
    return normalize_task2_output(raw).normalized_output


def _validate_task2(raw: str) -> tuple[bool, str]:
    contract = normalize_task2_output(raw)
    if contract.is_valid:
        return True, "valid general response category"
    return False, "task2 did not produce Yes/No/Maybe/Question/Stop"


def _normalize_decision(raw: str) -> str:
    return normalize_decision_output(raw, default="1").normalized_output


def _validate_decision(raw: str) -> tuple[bool, str]:
    contract = normalize_decision_output(raw, default="1")
    if contract.is_valid:
        return True, "valid DECISION: 0/1"
    return False, "reasoner did not produce a parseable 0/1 decision"


def _normalize_cbt_decision(raw: str) -> str:
    return normalize_decision_output(raw, default="1").normalized_output


def build_smoke_cases() -> list[SmokeCase]:
    answer = "My weight increased a lot recently."
    statement = "I often do stress eating during deadlines."
    unhelpful = "If I gain weight, I am failing."
    challenge = "Deadlines affect my eating, but gaining weight does not mean I am failing."
    reframe = "I can plan regular meals during deadlines and treat this as something to work on."

    return [
        SmokeCase(
            name="task1_response_analyzer",
            task=LLMTask.TASK1_RESPONSE_ANALYZER,
            system_prompt=response_analyzer.INIT_ASKER_SYSTEM_PROMPT_V2,
            payload=response_analyzer._format_task1_input(answer),
            max_new_tokens=16,
            normalize=_normalize_task1,
            validate=_validate_task1,
        ),
        SmokeCase(
            name="task2_general_response",
            task=LLMTask.TASK2_GENERAL_RESPONSE,
            system_prompt=response_analyzer.GENERAL_RESPONSE_SYSTEM_PROMPT,
            payload="Response: Nope.",
            max_new_tokens=8,
            normalize=_normalize_task2,
            validate=_validate_task2,
        ),
        SmokeCase(
            name="task3_rv_reasoner_invalid",
            task=LLMTask.TASK3_RV_REASONER,
            system_prompt=reflection_validation.RV_FOLLOW_UP_SYSTEM_REASONER_PROMPT,
            payload=reflection_validation._format_rv_reasoner_input(
                "weight",
                answer,
                "I like painting.",
            ),
            max_new_tokens=8,
            normalize=_normalize_decision,
            validate=_validate_decision,
        ),
        SmokeCase(
            name="task4_cbt_stage1",
            task=LLMTask.TASK4_CBT_STAGE1,
            system_prompt=CBT.ADAPTER_CBT_STAGE1_PROMPT,
            payload=CBT._format_stage1_input(statement, unhelpful),
            max_new_tokens=8,
            normalize=_normalize_cbt_decision,
            validate=_validate_decision,
        ),
        SmokeCase(
            name="task4_cbt_stage2",
            task=LLMTask.TASK4_CBT_STAGE2,
            system_prompt=CBT.ADAPTER_CBT_STAGE2_PROMPT,
            payload=CBT._format_stage2_input(statement, unhelpful, challenge),
            max_new_tokens=8,
            normalize=_normalize_cbt_decision,
            validate=_validate_decision,
        ),
        SmokeCase(
            name="task4_cbt_stage3",
            task=LLMTask.TASK4_CBT_STAGE3,
            system_prompt=CBT.ADAPTER_CBT_STAGE3_PROMPT,
            payload=CBT._format_stage3_input(statement, unhelpful, challenge, reframe),
            max_new_tokens=8,
            normalize=_normalize_cbt_decision,
            validate=_validate_decision,
        ),
    ]


def missing_runtime_dependencies() -> list[str]:
    required = ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]
    return [name for name in required if importlib.util.find_spec(name) is None]


def _shorten(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def run_case(case: SmokeCase, preview_chars: int) -> bool:
    adapter = resolve_adapter(case.task)
    print(f"\n=== {case.name} ===")
    print(f"task: {case.task.value}")
    print(f"adapter: {adapter}")
    print(f"max_new_tokens: {case.max_new_tokens}")
    print("payload:")
    print(case.payload)
    print("system_prompt_preview:")
    print(_shorten(case.system_prompt, preview_chars))

    start = time.perf_counter()
    result = llm_complete_task(
        case.task,
        case.system_prompt,
        case.payload,
        max_new_tokens=case.max_new_tokens,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    normalized = case.normalize(result.text)
    contract_ok, contract_message = case.validate(result.text)

    print("raw_output:")
    print(result.text)
    print("normalized_output:")
    print(normalized)
    print("contract:")
    print("OK: " + contract_message if contract_ok else "FAIL: " + contract_message)
    print(f"latency_ms: {latency_ms:.1f}")
    return contract_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real local smoke tests for CaiTI adapters.")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=360,
        help="Number of system-prompt characters to print per task.",
    )
    parser.add_argument(
        "--ignore-missing-deps",
        action="store_true",
        help="Try running even if Python package dependency probes fail.",
    )
    args = parser.parse_args(argv)

    missing = missing_runtime_dependencies()
    if missing and not args.ignore_missing_deps:
        print("Cannot run real adapter smoke test because runtime dependencies are missing.")
        print("Missing packages: " + ", ".join(missing))
        print("Install the local runtime dependencies, then re-run:")
        print("  pip install torch transformers peft accelerate bitsandbytes")
        return 2

    cases = build_smoke_cases()
    print(f"Running {len(cases)} real CaiTI adapter smoke cases.")
    all_ok = True
    for case in cases:
        all_ok = run_case(case, args.preview_chars) and all_ok
    if all_ok:
        print("\nAll smoke cases completed with valid output contracts.")
        return 0
    print("\nSmoke cases completed, but at least one output contract failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
