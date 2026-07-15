"""Print a deterministic CaiTI end-to-end control-flow trace.

This script verifies the application wiring without loading the local model:
screening -> task1 -> R-V invalid/valid loop -> validation prefix -> CBT stages.
Use scripts/smoke_test_adapters.py for real adapter inference checks.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import CBT, questioner, reflection_validation
from src.local_llm.types import GenerationResult
from src.utils import response_bridge
from src.utils.llm_output_contracts import normalize_task1_output


def _read_events(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            """
            SELECT task, adapter, dimension, score, segment_text, raw_llm_output, normalized_output
            FROM session_events
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()


def main() -> int:
    question_lib = {
        "1": {
            "1": {
                "label": "weight",
                "name": "Maintaining stable weight",
                "question": ["Have your weight changed significantly recently?"],
                "score": [],
                "notes": [],
                "Yes": 2,
                "No": 0,
            }
        }
    }
    question_text = question_lib["1"]["1"]["question"][0]
    user_segments = ["My weight increased a lot recently."]
    rv_responses = iter(
        [
            "I like painting.",
            "I often do stress eating during deadlines.",
        ]
    )
    rv_decisions = iter(["1", "0"])
    cbt_responses = iter(
        [
            "1",
            "If I gain weight, I am failing.",
            "Deadlines affect my eating, but gaining weight does not mean I am failing.",
            "I can treat this as something to understand, not as proof that I am failing.",
        ]
    )
    ai_outputs: list[str] = []
    prefixes: list[str] = []

    originals = {
        "response_bridge.classify_dimension_and_score_result": (
            response_bridge.classify_dimension_and_score_result
        ),
        "reflection_validation.llm_complete_task": reflection_validation.llm_complete_task,
        "questioner.get_resp_log": questioner.get_resp_log,
        "questioner.log_question": questioner.log_question,
        "questioner.rv_guide": questioner.rv_guide,
        "questioner.rv_validation": questioner.rv_validation,
        "CBT.get_resp_log": CBT.get_resp_log,
        "CBT.log_question": CBT.log_question,
        "CBT.set_question_prefix": CBT.set_question_prefix,
        "CBT.llm_complete_task": CBT.llm_complete_task,
        "CBT.recap_stage3_challenge": CBT.recap_stage3_challenge,
    }

    def fake_rv_complete_task(task, system_content, user_content, max_new_tokens=None):
        raw = next(rv_decisions)
        return GenerationResult(
            text=raw,
            task=task,
            adapter="adapters/task3_rv_reasoner",
            raw_text=raw,
        )

    def fake_cbt_complete_task(task, system_content, user_content, max_new_tokens=None):
        return GenerationResult(
            text="0",
            task=task,
            adapter=f"adapters/{task.value}",
            raw_text="0",
        )

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "trace_events.sqlite3")
        old_db_path = os.environ.get("CAITI_STRUCTURED_LOG_DB")
        old_session_id = os.environ.get("CAITI_SESSION_ID")
        os.environ["CAITI_STRUCTURED_LOG_DB"] = db_path
        os.environ["CAITI_SESSION_ID"] = "trace-mock-e2e"

        try:
            response_bridge.classify_dimension_and_score_result = (
                lambda _answer, _question: normalize_task1_output("1_weight, 2")
            )
            reflection_validation.llm_complete_task = fake_rv_complete_task
            questioner.get_resp_log = lambda: next(rv_responses)
            questioner.log_question = lambda text: ai_outputs.append(text)
            questioner.rv_guide = lambda *_args: "Guide: Thank you for sharing that. I want to return to what you mentioned earlier: \"My weight increased a lot recently.\" Could you tell me more about that?"
            questioner.rv_validation = lambda *_args: "VALIDATION: You mentioned stress eating during deadlines, and that connects to the weight change you shared earlier. I appreciate you explaining that connection."

            CBT.get_resp_log = lambda: next(cbt_responses)
            CBT.log_question = lambda text: ai_outputs.append(text)
            CBT.set_question_prefix = lambda text: prefixes.append(text)
            CBT.llm_complete_task = fake_cbt_complete_task
            CBT.recap_stage3_challenge = (
                lambda *_args: "You challenged the thought by naming another possibility."
            )

            print("=== Screening ===")
            print(f"[AI Rephraser] {question_text}")
            print(f"[User] {user_segments[0]}")
            dla_result = questioner.classify_segments(user_segments, question_text, "weight")
            print(f"[Response Analyzer] {dla_result}")

            valid, terminate, followup, updated = questioner.evaluate_result(
                question_lib,
                dla_result,
                1,
                "1",
                user_segments,
                question_text,
            )
            print(f"[Control] valid={valid}, terminate={terminate}")
            print(f"[ReflectiveSummarizer] {followup}")

            print("\n=== R-V Loop ===")
            print(f"[AI] {ai_outputs[0]}")
            print("[User] I like painting.")
            print(f"[R-V Reasoner] DECISION: 1")
            print(f"[R-V Guide] {ai_outputs[1]}")
            print("[User] I often do stress eating during deadlines.")
            print(f"[R-V Reasoner] DECISION: 0")
            print(f"[R-V Validator queued] {questioner._PENDING_VALIDATION_TEXT}")

            print("\n=== CBT ===")
            CBT.run_cbt(updated)
            for idx, text in enumerate(ai_outputs[2:], start=1):
                print(f"[CBT AI {idx}] {text}")

            print("\n=== Notes ===")
            for note in updated["1"]["1"]["notes"]:
                print(note)

            print("\n=== Structured Events ===")
            for row in _read_events(db_path):
                task, adapter, dimension, score, segment, raw, normalized = row
                print(
                    f"{task} | adapter={adapter} | dim={dimension} | score={score} | "
                    f"segment={segment!r} | raw={raw!r} | normalized={normalized!r}"
                )
        finally:
            response_bridge.classify_dimension_and_score_result = originals[
                "response_bridge.classify_dimension_and_score_result"
            ]
            reflection_validation.llm_complete_task = originals[
                "reflection_validation.llm_complete_task"
            ]
            questioner.get_resp_log = originals["questioner.get_resp_log"]
            questioner.log_question = originals["questioner.log_question"]
            questioner.rv_guide = originals["questioner.rv_guide"]
            questioner.rv_validation = originals["questioner.rv_validation"]
            CBT.get_resp_log = originals["CBT.get_resp_log"]
            CBT.log_question = originals["CBT.log_question"]
            CBT.set_question_prefix = originals["CBT.set_question_prefix"]
            CBT.llm_complete_task = originals["CBT.llm_complete_task"]
            CBT.recap_stage3_challenge = originals["CBT.recap_stage3_challenge"]
            if old_db_path is None:
                os.environ.pop("CAITI_STRUCTURED_LOG_DB", None)
            else:
                os.environ["CAITI_STRUCTURED_LOG_DB"] = old_db_path
            if old_session_id is None:
                os.environ.pop("CAITI_SESSION_ID", None)
            else:
                os.environ["CAITI_SESSION_ID"] = old_session_id

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
