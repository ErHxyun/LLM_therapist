"""Run a deterministic long-session reliability test for CaiTI.

This script exercises the real HandlerRL control flow with deterministic mocks
for LLM outputs and a fake local user that drives record.csv. It is intended to
catch long-session lock, report, SQLite logging, and memory regressions without
loading the real model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import CBT, questioner, reflection_validation  # noqa: E402
from src.handler_rl import HandlerRL  # noqa: E402
from src.local_llm.types import GenerationResult, LLMTask  # noqa: E402
from src.utils import io_question_lib, io_record, session_event_logger  # noqa: E402
from src.utils.config_loader import QUESTION_LIB_FILENAME  # noqa: E402
from src.utils.session_event_logger import log_llm_event  # noqa: E402

import src.handler_rl as handler_rl_module  # noqa: E402


@dataclass
class ScriptedUserState:
    prompt_count: int = 0
    response_count: int = 0
    screening_response_count: int = 0
    followup_response_count: int = 0
    prompts: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)


class ScriptedUser:
    """Poll record.csv, unlock prompts, and provide deterministic responses."""

    def __init__(self, record_path: Path, stop_event: threading.Event, poll_interval_sec: float = 0.05):
        self.record_path = record_path
        self.stop_event = stop_event
        self.poll_interval_sec = poll_interval_sec
        self.state = ScriptedUserState()
        self.error: str | None = None

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self._respond_once_if_ready()
                time.sleep(self.poll_interval_sec)
        except Exception as exc:
            self.error = repr(exc)
            self.stop_event.set()

    def _respond_once_if_ready(self) -> None:
        if not self.record_path.exists():
            return
        df = pd.read_csv(
            self.record_path,
            dtype={"Question": str, "Question_Lock": "int64", "Resp": str, "Resp_Lock": "int64"},
        )
        if int(df.loc[0, "Question_Lock"]) != 1:
            return

        prompt = str(df.loc[0, "Question"])
        response = self.response_for(prompt)
        df.loc[0, "Question_Lock"] = 0
        df.loc[0, "Resp"] = response
        df.loc[0, "Resp_Lock"] = 0
        tmp_path = str(self.record_path) + ".tmp"
        df.to_csv(tmp_path, columns=io_record.HEADER, index=False)
        os.replace(tmp_path, self.record_path)

        self.state.prompt_count += 1
        self.state.response_count += 1
        self.state.prompts.append(prompt)
        self.state.responses.append(response)

    def response_for(self, prompt: str) -> str:
        text = " ".join(str(prompt or "").split()).lower()
        if "which dimension would you like to work on today" in text:
            return "1"
        if "identify any unhelpful thoughts" in text or "provide your unhelpful_thoughts again" in text:
            return "If I struggle with this, I am failing."
        if "reframe" in text:
            return "I can treat this as a problem to work on, not proof that I am failing."
        if "challenge" in text:
            return "Having a difficult week does not mean I am failing."
        if "can you tell me more about that" in text:
            self.state.followup_response_count += 1
            return "It has been hard during deadlines, and I want to understand the pattern."
        if "great work today" in text or "we will conclude here" in text:
            return "Thank you."

        self.state.screening_response_count += 1
        if self.state.screening_response_count == 1:
            return "This has been hard recently."
        return "No current issue."


class RestoreStack:
    def __init__(self):
        self._items: list[tuple[object, str, Any]] = []

    def set(self, target: object, name: str, value: Any) -> None:
        self._items.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def restore(self) -> None:
        for target, name, old_value in reversed(self._items):
            setattr(target, name, old_value)


@contextmanager
def patched_environment(run_dir: Path, subject_id: str) -> Iterator[None]:
    record_path = run_dir / "record.csv"
    question_lib_path = run_dir / "question_lib.json"
    report_file = run_dir / f"Report_{subject_id}.csv"
    notes_file = run_dir / f"Notes_{subject_id}.csv"
    db_path = run_dir / "session_events.sqlite3"

    shutil.copyfile(QUESTION_LIB_FILENAME, question_lib_path)
    reset_question_lib_file(question_lib_path)

    old_env = {
        "CAITI_STRUCTURED_LOG_DB": os.environ.get("CAITI_STRUCTURED_LOG_DB"),
        "CAITI_SESSION_ID": os.environ.get("CAITI_SESSION_ID"),
    }
    os.environ["CAITI_STRUCTURED_LOG_DB"] = str(db_path)
    os.environ["CAITI_SESSION_ID"] = subject_id

    stack = RestoreStack()

    def fake_llm_complete(_system_content: str, user_content: str) -> str:
        if "opening greeting" in user_content.lower() or "hello" in user_content.lower():
            return "Hello, I am CaiTI. Let us begin with a few questions."
        return "VALIDATION: Thank you for explaining that clearly."

    def fake_llm_complete_task(task, _system_content, _user_content, max_new_tokens=None):
        raw = "0"
        return GenerationResult(text=raw, task=task, adapter=f"adapters/{task.value}", raw_text=raw)

    classifier_state = {"count": 0}

    def fake_get_openai_resp(segment: str, original_question: str, dimension_label: str):
        classifier_state["count"] += 1
        score = 2 if classifier_state["count"] == 1 else 0
        log_llm_event(
            task=LLMTask.TASK1_RESPONSE_ANALYZER,
            adapter="adapters/task1_response_analyzer",
            dimension=dimension_label,
            score=score,
            segment_text=segment,
            question_text=original_question,
            raw_llm_output=f"{dimension_label}, {score}",
            normalized_output=f"{dimension_label}, {score}",
            metadata={"mode": "long_session_mock"},
        )
        return dimension_label, score

    def deterministic_choose_action(_state, _q_table, mask, _number_states, actions, _action_labels=None):
        for idx in range(1, min(len(mask), len(actions))):
            if mask[idx] == 1:
                return actions[idx]
        return actions[-1]

    def generate_results_to_run_dir(question_lib: dict, new_response: list):
        return io_question_lib.generate_results(
            question_lib,
            new_response,
            report_file=str(report_file),
            notes_file=str(notes_file),
        )

    stack.set(io_record, "RECORD_CSV", str(record_path))
    stack.set(handler_rl_module, "RECORD_CSV", str(record_path))
    stack.set(handler_rl_module, "QUESTION_LIB_FILENAME", str(question_lib_path))
    stack.set(handler_rl_module, "DATA_DIR", str(run_dir))
    stack.set(handler_rl_module, "SUBJECT_ID", subject_id)
    stack.set(handler_rl_module, "choose_action", deterministic_choose_action)
    stack.set(handler_rl_module, "generate_results", generate_results_to_run_dir)
    stack.set(handler_rl_module, "llm_complete", fake_llm_complete)

    stack.set(questioner, "generate_synonymous_sentences", lambda text: text)
    stack.set(questioner, "get_openai_resp", fake_get_openai_resp)

    stack.set(reflection_validation, "llm_complete_task", fake_llm_complete_task)
    stack.set(reflection_validation, "llm_complete", fake_llm_complete)
    stack.set(CBT, "llm_complete_task", fake_llm_complete_task)
    stack.set(CBT, "llm_complete", fake_llm_complete)

    try:
        yield
    finally:
        stack.restore()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def snapshot_process_memory_mb() -> dict[str, float | None]:
    values = {"process_rss_mb": None, "process_peak_rss_mb": None}
    key_map = {"VmRSS": "process_rss_mb", "VmHWM": "process_peak_rss_mb"}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in key_map:
                values[key_map[key]] = float(rest.strip().split()[0]) / 1024.0
    except OSError:
        pass
    return values


def reset_question_lib_file(question_lib_path: Path) -> None:
    question_lib = json.loads(question_lib_path.read_text(encoding="utf-8"))
    for i in range(1, len(question_lib) + 1):
        for j in range(1, len(question_lib[str(i)]) + 1):
            question_lib[str(i)][str(j)]["score"] = []
            question_lib[str(i)][str(j)]["notes"] = []
    question_lib_path.write_text(json.dumps(question_lib), encoding="utf-8")


def read_record_locks(record_path: Path) -> dict[str, int | None]:
    if not record_path.exists():
        return {"question_lock": None, "resp_lock": None}
    df = pd.read_csv(record_path)
    return {
        "question_lock": int(df.loc[0, "Question_Lock"]),
        "resp_lock": int(df.loc[0, "Resp_Lock"]),
    }


def read_event_counts(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT task, COUNT(*) FROM session_events GROUP BY task ORDER BY task").fetchall()
        return {str(task): int(count) for task, count in rows}
    finally:
        conn.close()


def count_report_rows(report_file: Path) -> int:
    if not report_file.exists():
        return 0
    with report_file.open("r", newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _row in csv.reader(f)) - 1)


def analyze_question_lib(question_lib: dict) -> dict[str, Any]:
    score_count = 0
    score2_count = 0
    note_count = 0
    cbt_success = False
    dimensions_with_scores = []
    for i in range(1, len(question_lib) + 1):
        entry = question_lib[str(i)]["1"]
        scores = entry.get("score", [])
        if scores:
            score_count += 1
            dimensions_with_scores.append(entry.get("label", str(i)))
        if any(score == 2 for score in scores):
            score2_count += 1
        for note in entry.get("notes", []):
            if isinstance(note, list):
                note_count += len(note)
                if any(item == "CBT_stage: success" for item in note):
                    cbt_success = True
            else:
                note_count += 1
    return {
        "dimension_count": len(question_lib),
        "score_count": score_count,
        "score2_count": score2_count,
        "note_count": note_count,
        "cbt_success": cbt_success,
        "dimensions_with_scores": dimensions_with_scores,
    }


def latest_question_lib_snapshot(run_dir: Path, fallback: Path) -> Path:
    snapshots = sorted(run_dir.glob("question_lib_*.json"), key=lambda path: path.stat().st_mtime)
    return snapshots[-1] if snapshots else fallback


def default_run_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "data" / "results" / f"long_session_reliability_{stamp}"


def run_long_session(run_dir: Path, subject_id: str, timeout_sec: float) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    record_path = run_dir / "record.csv"
    db_path = run_dir / "session_events.sqlite3"
    report_file = run_dir / f"Report_{subject_id}.csv"
    notes_file = run_dir / f"Notes_{subject_id}.csv"
    question_lib_path = run_dir / "question_lib.json"

    stop_event = threading.Event()
    scripted_user = ScriptedUser(record_path, stop_event)
    user_thread = threading.Thread(target=scripted_user.run, name="long-session-scripted-user", daemon=True)

    memory_before = snapshot_process_memory_mb()
    start = time.perf_counter()
    completed = False
    error = None
    timed_out = False

    with patched_environment(run_dir, subject_id):
        user_thread.start()
        result_holder: dict[str, Any] = {}

        def run_handler():
            try:
                HandlerRL().run()
                result_holder["completed"] = True
            except Exception as exc:
                result_holder["error"] = repr(exc)
                result_holder["traceback"] = __import__("traceback").format_exc()

        handler_thread = threading.Thread(target=run_handler, name="long-session-handler", daemon=True)
        handler_thread.start()
        handler_thread.join(timeout_sec)
        if handler_thread.is_alive():
            timed_out = True
            error = f"HandlerRL did not finish within {timeout_sec} seconds."
        else:
            completed = bool(result_holder.get("completed"))
            error = result_holder.get("error")

        stop_event.set()
        user_thread.join(timeout=2.0)

    duration_sec = time.perf_counter() - start
    memory_after = snapshot_process_memory_mb()

    analyzed_question_lib_path = latest_question_lib_snapshot(run_dir, question_lib_path)
    question_lib = {}
    if analyzed_question_lib_path.exists():
        question_lib = json.loads(analyzed_question_lib_path.read_text(encoding="utf-8"))

    qlib_summary = analyze_question_lib(question_lib) if question_lib else {}
    locks = read_record_locks(record_path)
    event_counts = read_event_counts(db_path)
    report_rows = count_report_rows(report_file)
    expected_dimensions = int(qlib_summary.get("dimension_count", 0))
    prompt_type_counts = Counter(
        "cbt" if "CBT" in prompt or "unhelpful" in prompt.lower() or "challenge" in prompt.lower() else "screening_or_system"
        for prompt in scripted_user.state.prompts
    )

    checks = {
        "completed": completed and not timed_out and error is None,
        "scripted_user_ok": scripted_user.error is None,
        "all_dimensions_scored": qlib_summary.get("score_count") == expected_dimensions,
        "cbt_success": bool(qlib_summary.get("cbt_success")),
        "question_lock_released": locks.get("question_lock") == 0,
        "report_rows_match_dimensions": report_rows == expected_dimensions,
        "event_log_has_task1": event_counts.get(LLMTask.TASK1_RESPONSE_ANALYZER.value, 0) == expected_dimensions,
        "event_log_has_cbt": all(
            event_counts.get(task.value, 0) >= 1
            for task in (
                LLMTask.TASK4_CBT_STAGE1,
                LLMTask.TASK4_CBT_STAGE2,
                LLMTask.TASK4_CBT_STAGE3,
            )
        ),
    }

    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "subject_id": subject_id,
        "run_dir": str(run_dir),
        "duration_sec": duration_sec,
        "completed": completed,
        "timed_out": timed_out,
        "error": error or scripted_user.error,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "scripted_user": asdict(scripted_user.state),
        "prompt_type_counts": dict(prompt_type_counts),
        "question_lib": qlib_summary,
        "record_locks": locks,
        "event_counts": event_counts,
        "artifacts": {
            "record_csv": str(record_path),
            "event_db": str(db_path),
            "question_lib": str(question_lib_path),
            "analyzed_question_lib": str(analyzed_question_lib_path),
            "report_csv": str(report_file),
            "notes_csv": str(notes_file),
            "qtable_dir": str(run_dir / "q_tables"),
        },
        "report_rows": report_rows,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (run_dir / "reliability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("CaiTI long-session reliability")
    print(f"run_dir: {report['run_dir']}")
    print(f"duration_sec: {report['duration_sec']:.2f}")
    print(f"passed: {report['passed']}")
    print(f"completed: {report['completed']}")
    print(f"error: {report['error']}")
    print("\nchecks:")
    for name, value in report["checks"].items():
        print(f"- {name}: {value}")
    print("\nsummary:")
    print(f"- dimensions: {report['question_lib'].get('dimension_count')}")
    print(f"- scored dimensions: {report['question_lib'].get('score_count')}")
    print(f"- score=2 dimensions: {report['question_lib'].get('score2_count')}")
    print(f"- cbt_success: {report['question_lib'].get('cbt_success')}")
    print(f"- prompt_count: {report['scripted_user'].get('prompt_count')}")
    print(f"- report_rows: {report['report_rows']}")
    print(f"- record_locks: {report['record_locks']}")
    print(f"- event_counts: {report['event_counts']}")
    before = report["memory_before"].get("process_rss_mb")
    after = report["memory_after"].get("process_rss_mb")
    peak = report["memory_after"].get("process_peak_rss_mb")
    if before is not None and after is not None:
        print(f"- process_rss_mb: {before:.1f} -> {after:.1f} (peak {peak:.1f})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mocked full-session CaiTI reliability test.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for run artifacts.")
    parser.add_argument("--subject-id", default="long_session_mock", help="Session id used for artifacts/logs.")
    parser.add_argument("--timeout-sec", type=float, default=180.0, help="Maximum time for HandlerRL().run().")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.output_dir or default_run_dir()
    report = run_long_session(run_dir.resolve(), args.subject_id, args.timeout_sec)
    print_summary(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
