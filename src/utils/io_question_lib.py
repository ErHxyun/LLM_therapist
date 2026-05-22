import json
import csv
import os
from src.utils.config_loader import REPORT_FILE, NOTES_FILE

def load_question_lib(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_question_lib(path: str, question_lib: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(question_lib, f)

def _flatten_notes(notes):
    for note in notes or []:
        if isinstance(note, list):
            for item in note:
                yield str(item)
        else:
            yield str(note)

def _format_score(score):
    if isinstance(score, list):
        return "; ".join(str(item) for item in score)
    return "" if score is None else str(score)

def _format_responses(notes):
    response_prefixes = (
        "original_resp:",
        "followup_resp:",
        "followup_resp_1:",
        "CBT_unhelpful_thoughts:",
        "CBT_challenge:",
        "CBT_reframe:",
    )
    responses = []
    for item in _flatten_notes(notes):
        if item.startswith(response_prefixes):
            responses.append(item.split(":", 1)[1].strip())
    return " | ".join(response for response in responses if response)

def _format_analysis(label, notes):
    details = [f"Dimension: {label}"]
    details.extend(_flatten_notes(notes))
    return " | ".join(detail for detail in details if detail)

def generate_results(
    question_lib: dict,
    new_response: list,
    report_file: str = REPORT_FILE,
    notes_file: str = NOTES_FILE
):
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    os.makedirs(os.path.dirname(notes_file), exist_ok=True)

    rows = []
    for i in range(1, len(question_lib) + 1):
        for ind in range(1, len(question_lib[str(i)]) + 1):
            entry = question_lib[str(i)][str(ind)]
            rows.append([
                _format_score(entry.get("score")),
                _format_responses(entry.get("notes")),
                _format_analysis(entry.get("label", ""), entry.get("notes")),
            ])

    # atomic write for report_file
    _tmp_report = report_file + ".tmp"
    with open(_tmp_report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Score", "Responses", "Analysis"])
        w.writerows(rows)
    os.replace(_tmp_report, report_file)

    rows_new = []
    for rec in new_response:
        try:
            rows_new.append([
                rec["item"],
                rec["question"],
                rec["DLA_result"],
                rec["User_input"],
                rec["User_comment"]
            ])
        except:
            rows_new.append([
                rec["item"],
                rec["question"],
                rec["DLA_result"],
                rec["User_input"]
            ])

    # atomic write for notes_file
    _tmp_notes = notes_file + ".tmp"
    with open(_tmp_notes, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(['Item', "question", "Original_question", "DLA_result", "User_input", "User_comment"])
        w.writerows(rows_new)
    os.replace(_tmp_notes, notes_file)
