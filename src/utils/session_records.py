import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Iterable


RL_TRACE_FIELDS = [
    "RunID",
    "SubjectID",
    "Step",
    "Timestamp",
    "State",
    "Action",
    "NextState",
    "Dimension",
    "Question",
    "UserResponse",
    "Classification",
    "Score",
    "Reward",
    "QBefore",
    "QAfter",
    "Terminate",
    "AttemptCount",
]


def make_run_id(subject_id: str) -> str:
    _ = subject_id
    return str(int(time.time()))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def join_segments(segments: Iterable[Any]) -> str:
    return " | ".join(str(segment).strip() for segment in segments if str(segment).strip())


def normalize_classification(classification: Iterable[tuple[Any, Any]]) -> list[list[Any]]:
    normalized = []
    for label, score in classification or []:
        normalized.append([str(label), score])
    return normalized


def format_classification(classification: Iterable[tuple[Any, Any]]) -> str:
    return json.dumps(normalize_classification(classification), ensure_ascii=False)


def latest_added_score(before_scores: Iterable[Any], after_scores: Iterable[Any]) -> Any:
    before = list(before_scores or [])
    after = list(after_scores or [])
    if len(after) > len(before):
        return after[-1]
    return ""


def build_question_attempt_record(
    *,
    item_id: int,
    question_index: str,
    dimension: str,
    question_text: str,
    user_segments: Iterable[str],
    classification: Iterable[tuple[Any, Any]],
    score_before: Iterable[Any],
    score_after: Iterable[Any],
    valid: int,
    terminate: int,
    attempt: str,
    triggered_reflection: bool,
) -> dict[str, Any]:
    score = latest_added_score(score_before, score_after)
    return {
        "item": item_id,
        "question": question_index,
        "Original_question": question_text,
        "DLA_result": format_classification(classification),
        "User_input": join_segments(user_segments),
        "User_comment": "",
        "Dimension": dimension,
        "Score": score,
        "Valid": int(valid),
        "Terminate": int(terminate),
        "Attempt": attempt,
        "Triggered_reflection": int(bool(triggered_reflection)),
        "Timestamp": now_iso(),
    }


def build_rl_trace_path(result_dir: str, subject_id: str, run_id: str) -> str:
    return os.path.join(result_dir, f"RLTrace_{subject_id}_{run_id}.csv")


def build_session_summary_path(result_dir: str, subject_id: str, run_id: str) -> str:
    return os.path.join(result_dir, f"SessionSummary_{subject_id}_{run_id}.json")


def write_rl_trace(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RL_TRACE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _flatten_notes(notes: Iterable[Any]) -> list[str]:
    flattened = []
    for note in notes or []:
        if isinstance(note, list):
            flattened.extend(str(item) for item in note)
        else:
            flattened.append(str(note))
    return flattened


def _responses_from_notes(notes: Iterable[Any]) -> str:
    prefixes = ("original_resp:", "followup_resp:", "followup_resp_1:")
    responses = []
    for item in _flatten_notes(notes):
        if item.startswith(prefixes):
            responses.append(item.split(":", 1)[1].strip())
    return " | ".join(response for response in responses if response)


def cbt_candidates_from_question_lib(question_lib: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for i in range(1, len(question_lib) + 1):
        for j in range(1, len(question_lib[str(i)]) + 1):
            entry = question_lib[str(i)][str(j)]
            scores = entry.get("score", [])
            if any(score == 2 for score in scores):
                candidates.append(
                    {
                        "item": i,
                        "question": j,
                        "dimension": entry.get("label", ""),
                        "name": entry.get("name", entry.get("label", "")),
                        "scores": list(scores),
                        "responses": _responses_from_notes(entry.get("notes", [])),
                    }
                )
    return candidates


def write_session_summary(path: str, summary: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
