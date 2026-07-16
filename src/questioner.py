import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np

from src.emotion import (
    assess_emotion_followup,
    build_emotion_followup_settings,
    pop_late_emotion_followup,
    queue_late_emotion_followup_request,
    wait_for_emotion_result,
)
from src.runtime.status_monitor import get_active_status_monitor
from src.utils.response_bridge import get_openai_resp, is_explicit_stop_request
from src.response_analyzer import reflective_summarizer
from src.utils.text_generators import (
    generate_change,
    generate_change_positive,
    generate_change_negative,
    generate_synonymous_sentences,
    generate_therapist_chat,
)
from src.utils.llm_client import llm_complete
from src.utils.session_event_logger import log_llm_event
from src.utils.session_records import build_question_attempt_record

# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.io_record import get_answer, get_resp_log, log_question, long_response_prompt
logger = get_logger("Questioner")

from src.reflection_validation import parse_rv_decision, rv_reasoner, rv_guide, rv_validation

# System prompt for generating a retry guide when re-asking the same question.
RETRY_GUIDE_SYSTEM_PROMPT = '''You are a concise and supportive therapist-assistant.

You will be provided with:
1) The topic label of the question (Topic)
2) The original question (Original Question)
3) The user's original answer (Original Answer)

Your task is to generate a short guidance that helps the user retry answering the question.
Rules:
- If the Original Answer includes a sentence that shows the user does not understand the question (e.g., "I don't understand", "I don't get it", "what do you mean"), then CLARIFY the question directly in one sentence.
- If the Original Answer includes a sentence that shows doubt/unsure/maybe (e.g., "I'm not sure", "maybe", "unsure", "I doubt"), then ASK the SAME QUESTION from a DIFFERENT ANGLE/PERSPECTIVE in one sentence.
- Otherwise, briefly restate the essence of the Original Question and encourage a concise answer.

Output format (ONE line only):
GUIDE: <your guidance here>

Example A (not understand):
{"Topic": "DLA_1_mood", "Original Question": "How has your mood been?", "Original Answer": "I don't get it."}
GUIDE: I’m would like to know about your recent feelings and mood; could you describe how you’ve been feeling lately?

Example B (unsure/maybe):
{"Topic": "DLA_1_weight", "Original Question": "Have you experienced significant weight change recently?", "Original Answer": "I'm not sure."}
GUIDE: Let us try from a different perspective: have your clothes been fitting tighter or looser than usual lately?

Example C (neither):
{"Topic": "DLA_5_sleep", "Original Question": "Have you been sleeping enough recently?", "Original Answer": "I sleep sometimes."}
GUIDE: Let us focus on sleeping time: in the past week, have you generally slept enough hours most nights?
'''

FOLLOWUP_CONTINUE_TEXT = "Thank you for your clarification. Let's continue our questions."
RV_MAX_GUIDE_RETRIES = 2
_PENDING_NEXT_QUESTION_INTRO = ""
_PENDING_VALIDATION_TEXT = ""


@dataclass(frozen=True)
class SegmentAssessment:
    segment_index: int
    text: str
    dimension: str
    item_id: int
    score: int | None
    response_type: str
    valid: bool


@dataclass
class EvaluationOutcome:
    assessments: List[SegmentAssessment] = field(default_factory=list)
    covered_item_ids: set[int] = field(default_factory=set)
    current_answered: bool = False
    score2_queue: List[SegmentAssessment] = field(default_factory=list)
    terminate: bool = False
    unresolved_segments: List[str] = field(default_factory=list)
    current_followup: str = ""

@dataclass
class QuestionTurnOutcome:
    reward: float
    terminate: int
    previous_question: str
    covered_item_ids: set[int] = field(default_factory=set)
    current_answered: bool = False


def _chat_complete(system_content: str, user_content: str):
    """
    Unified LLM entry that delegates to llm_complete.
    """
    return llm_complete(system_content, user_content)

def retry_guide(topic: str, original_question: str, original_answer: str) -> str:
    """
    Generate a concise guide to help the user retry answering the same question.
    - Clarify if the user did not understand
    - Ask from a different angle if user is unsure/maybe/doubt
    - Otherwise, restate essence and invite concise answer
    """
    logger.info("Generating retry guide for re-ask.")
    payload = f'{{"Topic": {topic!r}, "Original Question": {original_question!r}, "Original Answer": {original_answer!r}}}'
    return _chat_complete(RETRY_GUIDE_SYSTEM_PROMPT, payload)


def _set_pending_next_question_intro(text: str) -> None:
    global _PENDING_NEXT_QUESTION_INTRO
    _PENDING_NEXT_QUESTION_INTRO = str(text or "").strip()


def _pop_pending_next_question_intro() -> str:
    global _PENDING_NEXT_QUESTION_INTRO
    text = _PENDING_NEXT_QUESTION_INTRO
    _PENDING_NEXT_QUESTION_INTRO = ""
    return text


def _set_pending_validation_text(text: str) -> None:
    global _PENDING_VALIDATION_TEXT
    _PENDING_VALIDATION_TEXT = str(text or "").strip()


def _pop_pending_validation_text() -> str:
    global _PENDING_VALIDATION_TEXT
    text = _PENDING_VALIDATION_TEXT
    _PENDING_VALIDATION_TEXT = ""
    return text


def _strip_spoken_label(text: str) -> str:
    cleaned = str(text or "").strip()
    upper = cleaned.upper()
    for prefix in ("VALIDATION:", "GUIDE:"):
        if upper.startswith(prefix):
            return cleaned[len(prefix):].strip()
    return cleaned


def pop_pending_validation_for_workflow_transition() -> str:
    """Return validation that must be spoken before leaving screening.

    The normal screening path consumes validation together with the next
    question.  When screening ends immediately after R-V, there is no next
    screening question, so the handler transfers the validation to the first
    CBT output instead.  The screening-only continuation is discarded.
    """
    validation = _strip_spoken_label(_pop_pending_validation_text())
    _pop_pending_next_question_intro()
    return validation


def _compose_question_with_intro(question_text: str) -> str:
    validation = _strip_spoken_label(_pop_pending_validation_text())
    intro = _pop_pending_next_question_intro()
    parts = [
        text
        for text in (validation, intro, str(question_text or "").strip())
        if text
    ]
    return " ".join(parts)


def _publish_question_context(
    *,
    text: str,
    item_id: int | str,
    question_index: str,
    dimension: str,
    expects_response: bool = True,
    source: str = "screening",
) -> None:
    monitor = get_active_status_monitor()
    if monitor is None:
        return
    method = getattr(monitor, "set_prompt", None)
    if not callable(method):
        return
    try:
        method(
            text=str(text or ""),
            source=source,
            expects_response=bool(expects_response),
            item_id=str(item_id or ""),
            question_index=str(question_index or ""),
            dimension=str(dimension or ""),
        )
    except Exception:
        return


def _publish_score_update(
    *,
    item_id: int | str,
    question_index: str,
    dimension: str,
    score: int,
    user_input: str,
    classification: list[list[Any]],
    followup_text: str = "",
) -> None:
    monitor = get_active_status_monitor()
    if monitor is None:
        return
    method = getattr(monitor, "set_score", None)
    if not callable(method):
        return
    try:
        method(
            item_id=str(item_id or ""),
            question_index=str(question_index or ""),
            dimension=str(dimension or ""),
            score=score,
            user_input=str(user_input or ""),
            classification=classification,
            followup_text=str(followup_text or ""),
        )
    except Exception:
        return


def reset_questioner_session_state() -> None:
    global _PENDING_NEXT_QUESTION_INTRO
    global _PENDING_VALIDATION_TEXT
    _PENDING_NEXT_QUESTION_INTRO = ""
    _PENDING_VALIDATION_TEXT = ""

def _is_stop_request(text: str) -> bool:
    return is_explicit_stop_request(text)


def _workflow_should_stop_waiting(session_control=None):
    method = getattr(session_control, "should_interrupt_workflow_wait", None)
    if not callable(method):
        return None
    return method


def _get_answer_with_control(session_control=None):
    should_stop = _workflow_should_stop_waiting(session_control)
    try:
        return get_answer(should_stop=should_stop)
    except TypeError:
        return get_answer()


def _get_resp_log_with_control(session_control=None):
    should_stop = _workflow_should_stop_waiting(session_control)
    try:
        return get_resp_log(should_stop=should_stop)
    except TypeError:
        return get_resp_log()


def _checkpoint_requested_skip_to_cbt(session_control=None) -> bool:
    method = getattr(session_control, "checkpoint", None)
    if not callable(method):
        return False
    return method("screening") == "skip_to_cbt"


REFLECTIVE_SUMMARY_MAX_WORDS = 24
REFLECTIVE_FOLLOWUP_QUESTION = "Can you tell me more about that?"
REFLECTIVE_FALLBACK_SUMMARY = "You shared something important about this."


def _fallback_reflective_followup(
    original_response: str,
    original_question: str = "",
) -> str:
    if not str(original_response or "").strip():
        return REFLECTIVE_FOLLOWUP_QUESTION
    return f"{REFLECTIVE_FALLBACK_SUMMARY} {REFLECTIVE_FOLLOWUP_QUESTION}"


def _clean_reflective_summary(text: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    cleaned = re.sub(
        r"^REFLECTIVE_(?:SUMMARIZER|SUMMERIZER)\s*:\s*",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    followup_match = re.search(r"\bcan you tell me more\b", cleaned, flags=re.IGNORECASE)
    if followup_match:
        cleaned = cleaned[:followup_match.start()].rstrip(" ,;:-")
    first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    return first_sentence.strip()


def _generate_reflective_followup(
    original_response: str,
    original_question: str = "",
) -> tuple[str, str, bool]:
    """Return (spoken follow-up, raw model output, fallback_used)."""
    fallback = _fallback_reflective_followup(original_response, original_question)
    response = " ".join(str(original_response or "").split())
    question = " ".join(str(original_question or "").split())
    if not response:
        return fallback, "", True
    try:
        raw = reflective_summarizer(question, response)
    except Exception as exc:
        logger.warning("Reflective Summarizer failed; using bounded fallback: %s", exc)
        return fallback, "", True
    summary = _clean_reflective_summary(raw)
    if not summary:
        return fallback, str(raw or ""), True
    summary_word_count = len(summary.split())
    if summary_word_count > REFLECTIVE_SUMMARY_MAX_WORDS:
        logger.warning(
            "Reflective summary exceeded %s words (%s); using bounded fallback.",
            REFLECTIVE_SUMMARY_MAX_WORDS,
            summary_word_count,
        )
        return fallback, str(raw or ""), True
    ending = "" if summary[-1] in ".!?" else "."
    return f"{summary}{ending} {REFLECTIVE_FOLLOWUP_QUESTION}", str(raw or ""), False


def _build_reflective_followup(original_response: str, original_question: str = "") -> str:
    followup, _, _ = _generate_reflective_followup(original_response, original_question)
    return followup

def _log_reflective_followup(
    *,
    dimension: str,
    score: int,
    segment_text: str,
    question_text: str,
    followup_text: str,
    raw_output: str,
    latency_ms: float,
    fallback_used: bool,
) -> None:
    log_llm_event(
        task="reflective_summarizer",
        dimension=dimension,
        score=score,
        segment_text=segment_text,
        question_text=question_text,
        raw_llm_output=raw_output,
        normalized_output=followup_text,
        metadata={
            "mode": "paper_reflective_summarizer",
            "latency_ms": round(latency_ms, 2),
            "fallback_used": fallback_used,
            "max_summary_words": REFLECTIVE_SUMMARY_MAX_WORDS,
            "spoken_word_count": len(str(followup_text or "").split()),
        },
    )


def _maybe_build_emotion_followup(
    *,
    dimension: str,
    score: int,
    segment_text: str,
    question_text: str,
):
    try:
        settings = build_emotion_followup_settings()
        if not settings.enabled:
            return None
        started_at = time.monotonic()
        record = wait_for_emotion_result(segment_text, settings.wait_timeout_sec)
        if record is None:
            queue_late_emotion_followup_request(
                dimension=dimension,
                score=int(score),
                user_text=segment_text,
                question_text=question_text,
                settings=settings,
                started_at=started_at,
            )
            return None
        decision = assess_emotion_followup(
            score=int(score),
            user_text=segment_text,
            record=record,
            settings=settings,
        )
        if decision.should_follow_up:
            metadata = dict(decision.metadata)
            metadata.update(
                {
                    "mode": "emotion_followup",
                    "reason": decision.reason,
                    "followup_text": decision.followup_text,
                }
            )
            log_llm_event(
                task="emotion_assist_followup",
                dimension=dimension,
                score=score,
                segment_text=segment_text,
                question_text=question_text,
                raw_llm_output=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                normalized_output=decision.reason,
                metadata=metadata,
            )
        return decision
    except Exception as exc:
        logger.warning("Emotion follow-up check failed: %s", exc)
        return None


def _run_late_emotion_followup_before_question(session_control=None) -> bool:
    try:
        decision = pop_late_emotion_followup()
        if not decision or not decision.should_follow_up:
            return False
        validation_text = _strip_spoken_label(_pop_pending_validation_text())
        metadata = dict(getattr(decision, "metadata", {}) or {})
        metadata.update(
            {
                "mode": "late_emotion_followup",
                "reason": decision.reason,
                "followup_text": decision.followup_text,
            }
        )
        log_llm_event(
            task="emotion_late_followup",
            dimension=str(metadata.get("dimension", "")),
            score=int(metadata.get("score", 0)),
            segment_text=str(metadata.get("user_text", "")),
            question_text=str(metadata.get("question_text", "")),
            raw_llm_output=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            normalized_output=decision.reason,
            metadata=metadata,
        )
        followup_text = str(decision.followup_text or "").strip()
        combined_prompt = f"{validation_text} {followup_text}".strip() if validation_text else followup_text
        log_question(combined_prompt)
        user_response = _get_resp_log_with_control(session_control)
        if _checkpoint_requested_skip_to_cbt(session_control):
            logger.info("Late emotion follow-up interrupted by skip-to-CBT request.")
            return True
        if _is_stop_request(user_response):
            logger.info("User requested stop during late emotion follow-up.")
            return True
        logger.info("Completed late emotion follow-up before next question: %s", decision.reason)
    except Exception as exc:
        logger.warning("Late emotion follow-up check failed: %s", exc)
    return False


def _emotion_note_lines(decision) -> List[str]:
    if not decision or not getattr(decision, "should_follow_up", False):
        return []
    metadata = getattr(decision, "metadata", {}) or {}
    return [
        "emotion_followup_reason: " + str(getattr(decision, "reason", "")),
        "emotion_confidence: " + str(metadata.get("confidence", "")),
        "emotion_credibility_risk: " + str(metadata.get("credibility_risk", "")),
    ]


def _build_followup_for_score(
    *,
    dimension: str,
    score: int,
    answer_text: str,
    original_question: str,
):
    emotion_decision = _maybe_build_emotion_followup(
        dimension=dimension,
        score=score,
        segment_text=answer_text,
        question_text=original_question,
    )
    if emotion_decision and emotion_decision.should_follow_up:
        return emotion_decision.followup_text, emotion_decision
    if score > 1:
        started_at = time.monotonic()
        followup_text, raw_output, fallback_used = _generate_reflective_followup(
            answer_text,
            original_question,
        )
        latency_ms = (time.monotonic() - started_at) * 1000.0
        _log_reflective_followup(
            dimension=dimension,
            score=score,
            segment_text=answer_text,
            question_text=original_question,
            followup_text=followup_text,
            raw_output=raw_output,
            latency_ms=latency_ms,
            fallback_used=fallback_used,
        )
        return followup_text, emotion_decision
    return "", emotion_decision

class ClassificationResults(list):
    """List-compatible classifications with per-turn performance metadata."""

    def __init__(self, values=(), metrics=None):
        super().__init__(values)
        self.metrics = dict(metrics or {})


def classify_segments(user_segments: List[str], original_question: str, dimension_label: str) -> List[Tuple[str, int]]:
    """Classify every non-empty segment and retain aggregate call metrics."""
    nonempty_segments = [str(segment).strip() for segment in user_segments if str(segment).strip()]
    logger.info("Classifying user segments. Total segments: %d", len(nonempty_segments))
    result = []
    analyzer_call_count = 0
    analyzer_latency_ms = 0.0
    segment_latencies_ms = []
    batch_fallback = False
    for seg in nonempty_segments:
        call_metrics = {}
        started_at = time.monotonic()
        label, score = get_openai_resp(
            seg,
            original_question,
            dimension_label,
            metrics=call_metrics,
        )
        elapsed_ms = (time.monotonic() - started_at) * 1000.0
        analyzer_latency_ms += elapsed_ms
        segment_latencies_ms.append(round(elapsed_ms, 2))
        analyzer_call_count += int(call_metrics.get("analyzer_call_count", 0))
        batch_fallback = batch_fallback or bool(call_metrics.get("batch_fallback", False))
        logger.debug("Segment classified: '%s' -> (dim: %s, val: %s)", seg, label, str(score))
        result.append((label, score))
    metrics = {
        "segment_count": len(nonempty_segments),
        "analyzer_call_count": analyzer_call_count,
        "analyzer_latency_ms": round(analyzer_latency_ms, 2),
        "segment_latencies_ms": segment_latencies_ms,
        "batch_fallback": batch_fallback,
    }
    logger.info("Classification complete. Results: %s metrics=%s", str(result), metrics)
    return ClassificationResults(result, metrics)



def _resolve_dimension_target(
    question_lib: Dict[str, Any],
    label: str,
    current_item_index: int,
    current_question_index: str,
) -> Tuple[str, str] | None:
    """Resolve a Task 1 dimension to one fixed question-library item."""
    label_norm = str(label or "").strip().lower()
    current_item = str(current_item_index)
    current_question = str(current_question_index)
    current_entry = question_lib[current_item][current_question]
    if str(current_entry.get("label", "")).strip().lower() == label_norm:
        return current_item, current_question

    matches: List[Tuple[str, str]] = []
    for candidate_item, item_questions in question_lib.items():
        if not isinstance(item_questions, dict):
            continue
        for candidate_question, entry in item_questions.items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("label", "")).strip().lower() == label_norm:
                matches.append((str(candidate_item), str(candidate_question)))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            "Skipping cross-dimension score for ambiguous label %s; matches=%s.",
            label,
            matches,
        )
    else:
        logger.info("Skipping score for label %s because it is not in the question library.", label)
    return None


def _record_scored_segment(
    *,
    question_lib: Dict[str, Any],
    target_item: str,
    target_question: str,
    dimension: str,
    score: int,
    answer_text: str,
    original_question: str,
    classification_label: str,
    classification_value: Any,
    is_current_dimension: bool,
    allow_followup: bool = True,
    emotion_decision_override=None,
) -> str:
    """Persist one classified segment and return a current-dimension follow-up, if any."""
    entry = question_lib[target_item][target_question]
    entry.setdefault("score", []).append(int(score))

    followup_text = ""
    emotion_decision = emotion_decision_override
    if is_current_dimension and allow_followup:
        followup_text, emotion_decision = _build_followup_for_score(
            dimension=dimension,
            score=int(score),
            answer_text=answer_text,
            original_question=original_question,
        )

    note_resp = [
        "original_question: " + original_question,
        "original_resp: " + answer_text,
        "classified_dimension: " + dimension,
        "cross_dimension: " + str(not is_current_dimension).lower(),
    ]
    note_resp.extend(_emotion_note_lines(emotion_decision))
    entry.setdefault("notes", []).append(note_resp)
    _publish_score_update(
        item_id=target_item,
        question_index=target_question,
        dimension=dimension,
        score=int(score),
        user_input=answer_text,
        classification=[[str(classification_label), classification_value]],
        followup_text=followup_text,
    )
    logger.info(
        "Recorded score %s for dimension %s at question_lib[%s][%s] (cross_dimension=%s).",
        score,
        dimension,
        target_item,
        target_question,
        not is_current_dimension,
    )
    return followup_text


def _current_dimension_response_text(
    dla_result: List[Tuple[str, Any]],
    user_segments: List[str],
    question_label: str,
) -> str:
    for index, (label, score_val) in enumerate(dla_result):
        is_general_answer = str(score_val) in {"Yes", "No"}
        is_current_score = (
            str(label).strip().lower() == str(question_label).strip().lower()
            and score_val in [0, 1, 2]
        )
        if is_general_answer or is_current_score:
            return user_segments[index] if index < len(user_segments) else ""
    return user_segments[0] if user_segments else ""




def _if_valid_response(
    dla_result: List[Tuple[str, Any]],
    item_index: int,
    question_index: str,
    user_segments: List[str],
    original_question: str,
    question_lib: Dict[str, Any],
) -> EvaluationOutcome:
    """Assess every segment, persist valid scores, and return one structured outcome."""
    outcome = EvaluationOutcome()
    current_item = str(item_index)
    current_item_id = int(item_index)
    current_question = str(question_index)
    question_entry = question_lib[current_item][current_question]
    question_label = str(question_entry["label"])
    queued_score2_by_item: Dict[int, SegmentAssessment] = {}

    if not dla_result:
        logger.info("No DLA result provided. Returning an empty evaluation outcome.")
        return outcome

    for index, (label, score_val) in enumerate(dla_result):
        label_norm = str(label).strip()
        score_norm = score_val
        answer_text = user_segments[index] if index < len(user_segments) else ""
        logger.info("Processing dla_result entry: %s, %s", label_norm, score_norm)

        if str(score_norm) == "Stop":
            is_explicit = is_explicit_stop_request(answer_text)
            assessment = SegmentAssessment(
                segment_index=index,
                text=answer_text,
                dimension=question_label,
                item_id=current_item_id,
                score=None,
                response_type="stop",
                valid=is_explicit,
            )
            outcome.assessments.append(assessment)
            if is_explicit:
                outcome.terminate = True
                logger.info("Recorded explicit Stop without returning before later assessments.")
            else:
                outcome.unresolved_segments.append(answer_text)
                logger.info("Ignored non-explicit Stop classification.")
            continue

        if str(score_norm) in {"Yes", "No"}:
            mapped_score = question_entry.get(str(score_norm), 99)
            is_valid = mapped_score in [0, 1, 2]
            assessment = SegmentAssessment(
                segment_index=index,
                text=answer_text,
                dimension=question_label,
                item_id=current_item_id,
                score=int(mapped_score) if is_valid else None,
                response_type="general",
                valid=is_valid,
            )
            outcome.assessments.append(assessment)
            if not is_valid:
                outcome.unresolved_segments.append(answer_text)
                logger.warning("Unmapped general response %s for %s.", score_norm, question_label)
                continue

            if int(mapped_score) == 2:
                if current_item_id not in queued_score2_by_item:
                    queued_score2_by_item[current_item_id] = assessment
                logger.info(
                    "Queued provisional Score 2 for current dimension %s.",
                    question_label,
                )
            else:
                candidate_followup = _record_scored_segment(
                    question_lib=question_lib,
                    target_item=current_item,
                    target_question=current_question,
                    dimension=question_label,
                    score=int(mapped_score),
                    answer_text=answer_text or str(score_norm),
                    original_question=original_question,
                    classification_label=label_norm,
                    classification_value=score_norm,
                    is_current_dimension=True,
                    allow_followup=not bool(outcome.current_followup),
                )
                if candidate_followup and not outcome.current_followup:
                    outcome.current_followup = candidate_followup
                outcome.covered_item_ids.add(current_item_id)
            continue

        if score_norm in [0, 1, 2]:
            target = _resolve_dimension_target(
                question_lib,
                label_norm,
                item_index,
                question_index,
            )
            if target is None:
                outcome.assessments.append(
                    SegmentAssessment(
                        segment_index=index,
                        text=answer_text,
                        dimension=label_norm,
                        item_id=0,
                        score=int(score_norm),
                        response_type="score",
                        valid=False,
                    )
                )
                outcome.unresolved_segments.append(answer_text)
                continue

            target_item, target_question = target
            target_item_id = int(target_item)
            target_entry = question_lib[target_item][target_question]
            target_dimension = str(target_entry["label"])
            is_current_dimension = (
                target_item == current_item and target_question == current_question
            )
            assessment = SegmentAssessment(
                segment_index=index,
                text=answer_text,
                dimension=target_dimension,
                item_id=target_item_id,
                score=int(score_norm),
                response_type="score",
                valid=True,
            )
            outcome.assessments.append(assessment)
            if int(score_norm) == 2:
                if target_item_id not in queued_score2_by_item:
                    queued_score2_by_item[target_item_id] = assessment
                logger.info(
                    "Queued provisional Score 2 for dimension %s at item %s.",
                    target_dimension,
                    target_item_id,
                )
            else:
                candidate_followup = _record_scored_segment(
                    question_lib=question_lib,
                    target_item=target_item,
                    target_question=target_question,
                    dimension=target_dimension,
                    score=int(score_norm),
                    answer_text=answer_text,
                    original_question=original_question,
                    classification_label=label_norm,
                    classification_value=score_norm,
                    is_current_dimension=is_current_dimension,
                    allow_followup=not bool(outcome.current_followup),
                )
                if is_current_dimension and candidate_followup and not outcome.current_followup:
                    outcome.current_followup = candidate_followup
                outcome.covered_item_ids.add(target_item_id)
            continue

        response_type = (
            str(score_norm).lower()
            if str(score_norm) in {"Maybe", "Question"}
            else "unresolved"
        )
        outcome.assessments.append(
            SegmentAssessment(
                segment_index=index,
                text=answer_text,
                dimension=label_norm,
                item_id=0,
                score=None,
                response_type=response_type,
                valid=False,
            )
        )
        outcome.unresolved_segments.append(answer_text)
        logger.info("No score recorded for response value %s.", score_norm)

    outcome.current_answered = current_item_id in outcome.covered_item_ids
    queued = list(queued_score2_by_item.values())
    outcome.score2_queue = (
        [assessment for assessment in queued if assessment.item_id == current_item_id]
        + [assessment for assessment in queued if assessment.item_id != current_item_id]
    )
    if outcome.terminate:
        outcome.current_followup = ""
    if not outcome.current_answered and not outcome.terminate:
        logger.info(
            "Current dimension %s remains unanswered; valid cross-dimension scores were retained.",
            question_label,
        )
    return outcome


def _run_score2_reflection_validation(
    *,
    assessment: SegmentAssessment,
    followup_text: str,
    original_question: str,
    session_control=None,
    validation_prefix: str = "",
) -> Dict[str, Any]:
    """Run one interactive R-V cycle without committing its provisional Score 2."""
    rv_latency_ms = 0.0
    spoken_validation = _strip_spoken_label(validation_prefix)
    prompt = (
        f"{spoken_validation} {followup_text}".strip()
        if spoken_validation
        else followup_text
    )
    logger.info(
        "Running provisional Score 2 R-V for item %s (%s).",
        assessment.item_id,
        assessment.dimension,
    )
    log_question(long_response_prompt(prompt))
    user_response = _get_resp_log_with_control(session_control)
    followup_responses = [user_response]
    interrupted = _checkpoint_requested_skip_to_cbt(session_control)
    interruption_reason = "skip_to_cbt" if interrupted else ""
    rv_stopped = _is_stop_request(user_response)

    rv_decision_token = "1"
    rv_decision_raws: List[str] = []
    rv_guide_texts: List[str] = []
    rv_retry_limit = max(0, int(RV_MAX_GUIDE_RETRIES))
    rv_retry_count = 0

    while not interrupted and not rv_stopped:
        logger.info(
            "Running ReflectionValidation reasoner for provisional dimension '%s'.",
            assessment.dimension,
        )
        rv_call_started = time.monotonic()
        rv_decision_raw = rv_reasoner(
            assessment.dimension,
            original_question,
            assessment.text,
            user_response,
        )
        rv_latency_ms += (time.monotonic() - rv_call_started) * 1000.0
        rv_decision_raws.append(str(rv_decision_raw))
        rv_decision_token = parse_rv_decision(rv_decision_raw, default="1")
        logger.info("ReflectionValidation decision: %s", rv_decision_token)

        if rv_decision_token == "0":
            break
        if rv_retry_count >= rv_retry_limit:
            logger.info(
                "ReflectionValidation retries exhausted for provisional item %s after %s guide attempt(s).",
                assessment.item_id,
                rv_retry_count,
            )
            break

        rv_call_started = time.monotonic()
        rv_guide_text = rv_guide(
            assessment.dimension,
            original_question,
            assessment.text,
            user_response,
        )
        rv_latency_ms += (time.monotonic() - rv_call_started) * 1000.0
        rv_guide_texts.append(rv_guide_text)
        log_question(long_response_prompt(rv_guide_text))
        user_response = _get_resp_log_with_control(session_control)
        followup_responses.append(user_response)
        rv_retry_count += 1
        if _checkpoint_requested_skip_to_cbt(session_control):
            interrupted = True
            interruption_reason = "skip_to_cbt"
            break
        rv_stopped = _is_stop_request(user_response)

    completed = (
        rv_decision_token == "0"
        and not interrupted
        and not rv_stopped
    )
    validation_text = ""
    if completed:
        logger.info(
            "Running ReflectionValidation validator for provisional dimension '%s'.",
            assessment.dimension,
        )
        rv_call_started = time.monotonic()
        validation_text = rv_validation(
            assessment.dimension,
            original_question,
            assessment.text,
            user_response,
        )
        rv_latency_ms += (time.monotonic() - rv_call_started) * 1000.0

    retry_exhausted = (
        rv_decision_token != "0"
        and not interrupted
        and not rv_stopped
        and rv_retry_count >= rv_retry_limit
    )
    notes = [
        "original_question: " + original_question,
        "original_resp: " + assessment.text,
        "followup_resp: " + (followup_responses[0] if followup_responses else ""),
        "rv_decision: " + rv_decision_token,
        "rv_decision_raw: " + " | ".join(rv_decision_raws),
        "rv_guide: " + " | ".join(rv_guide_texts),
        "rv_retry_count: " + str(rv_retry_count),
        "rv_retry_exhausted: " + str(retry_exhausted).lower(),
        "followup_resp_1: " + (
            followup_responses[-1] if len(followup_responses) > 1 else ""
        ),
        "rv_validation: " + validation_text,
        "rv_completed: " + str(completed).lower(),
        "rv_interrupted: " + interruption_reason,
        "therapist_resp: ",
    ]
    return {
        "completed": completed,
        "terminate": bool(interrupted or rv_stopped),
        "followup_text": followup_text,
        "validation_text": validation_text,
        "notes": notes,
        "rv_latency_ms": round(rv_latency_ms, 2),
    }



def evaluate_result(question_lib, DLA_result, S, question_A, user_input, original_question_asked, session_control=None, metrics=None):
    """
    Evaluate the result of a user's response to a question.
    Updates the question library and last question as needed.
    ReflectionValidation three steps（topic = the dimension label of the current question）
    """
    if isinstance(metrics, dict):
        metrics.setdefault("rv_latency_ms", 0.0)
    logger.info(f"Evaluating result for item {S}, question {question_A}.")
    # If valid user response, update the question library and last question
    outcome = _if_valid_response(
        [(lbl, sc) for lbl, sc in DLA_result], S, question_A, user_input, original_question_asked, question_lib
    )
    valid = int(outcome.current_answered)
    terminate = int(outcome.terminate)
    followup_to_RV = outcome.current_followup
    question_label = question_lib[str(S)][str(question_A)]["label"]
    current_response_text = _current_dimension_response_text(
        [(lbl, sc) for lbl, sc in DLA_result],
        user_input,
        question_label,
    )
    # Update previous_question if a new one is provided
    previous_question = followup_to_RV 
    if followup_to_RV:
        # If valid user response, log the last question and collect user response
        logger.info(f"Logging AI follow-up question and collecting user response for item {S}, question {question_A}.")
        # Log the last AI question and get a user response
        log_question(long_response_prompt(followup_to_RV))
        user_response = _get_resp_log_with_control(session_control)
        if _checkpoint_requested_skip_to_cbt(session_control):
            logger.info("Reflection follow-up interrupted by skip-to-CBT request.")
            return valid, 1, previous_question, question_lib

        # ReflectionValidation three steps（topic = the dimension label of the current question）
        topic = question_lib[str(S)][str(question_A)]["label"]
        original_resp = current_response_text

        rv_decision_token = "1"
        rv_decision_raws = []
        rv_guide_texts = []
        followup_responses = [user_response]
        rv_stopped = _is_stop_request(user_response)

        rv_retry_limit = max(0, int(RV_MAX_GUIDE_RETRIES))
        rv_retry_count = 0
        while not rv_stopped:
            logger.info(f"Running ReflectionValidation reasoner for topic '{topic}'.")
            rv_decision_raw = rv_reasoner(topic, original_question_asked, original_resp, user_response)
            rv_decision_raws.append(str(rv_decision_raw))
            # 0 means related/valid; 1 means unrelated/invalid.
            rv_decision_token = parse_rv_decision(rv_decision_raw, default="1")
            logger.info(f"ReflectionValidation decision: {rv_decision_token}")

            if rv_decision_token == "0":
                break

            if rv_retry_count >= rv_retry_limit:
                logger.info(
                    "ReflectionValidation retries exhausted after %s guide attempt(s); continuing without further recollection.",
                    rv_retry_count,
                )
                break

            logger.info("Follow-up not related, generating guidance and recollecting follow-up.")
            rv_guide_text = rv_guide(topic, original_question_asked, original_resp, user_response)
            rv_guide_texts.append(rv_guide_text)
            log_question(long_response_prompt(rv_guide_text))
            user_response = _get_resp_log_with_control(session_control)
            if _checkpoint_requested_skip_to_cbt(session_control):
                logger.info("Reflection retry interrupted by skip-to-CBT request.")
                return valid, 1, previous_question, question_lib
            followup_responses.append(user_response)
            rv_stopped = _is_stop_request(user_response)
            rv_retry_count += 1

        rv_validation_text = ""
        if rv_decision_token == "0" and not rv_stopped:
            # Empathic validation happens only after the Reasoner accepts the follow-up.
            logger.info("Running ReflectionValidation empathic validation.")
            rv_validation_text = rv_validation(topic, original_question_asked, original_resp, user_response)
            _set_pending_validation_text(rv_validation_text)
            _set_pending_next_question_intro(FOLLOWUP_CONTINUE_TEXT)
            logger.info("Queued validation for the next combined follow-up/question prompt.")
        elif rv_decision_token != "0" and not rv_stopped and rv_retry_count >= rv_retry_limit:
            _set_pending_next_question_intro(FOLLOWUP_CONTINUE_TEXT)
            logger.info("Proceeding to the next question after ReflectionValidation retry limit was reached.")
        
        # Skip generating therapist response to avoid unnecessary LLM calls
        therapist_resp = ""
        
        # Record notes (expand RV fields)
        logger.info("Recording notes for this question/response.")
        note_resp = [
            "original_question: " + original_question_asked,
            "original_resp: " + current_response_text,
            "followup_resp: " + (followup_responses[0] if followup_responses else ""),
            "rv_decision: " + rv_decision_token,
            "rv_decision_raw: " + " | ".join(rv_decision_raws),
            "rv_guide: " + " | ".join(rv_guide_texts),
            "rv_retry_count: " + str(rv_retry_count),
            "rv_retry_exhausted: " + str(
                rv_decision_token != "0" and not rv_stopped and rv_retry_count >= rv_retry_limit
            ).lower(),
            "followup_resp_1: " + (followup_responses[-1] if len(followup_responses) > 1 else ""),
            "rv_validation: " + rv_validation_text,
            "therapist_resp: " + therapist_resp
        ]
        question_lib[str(S)][str(question_A)]["notes"].append(note_resp)

        if rv_stopped:
            logger.info("User requested stop during ReflectionValidation.")
            return valid, 1, previous_question, question_lib
        
    if outcome.score2_queue and not terminate:
        validation_prefix = _pop_pending_validation_text()
        _pop_pending_next_question_intro()
        last_rv_completed = False

        for assessment in outcome.score2_queue:
            followup_text, emotion_decision = _build_followup_for_score(
                dimension=assessment.dimension,
                score=2,
                answer_text=assessment.text,
                original_question=original_question_asked,
            )
            if not previous_question:
                previous_question = followup_text

            rv_run = _run_score2_reflection_validation(
                assessment=assessment,
                followup_text=followup_text,
                original_question=original_question_asked,
                session_control=session_control,
                validation_prefix=validation_prefix,
            )
            if isinstance(metrics, dict):
                metrics["rv_latency_ms"] += float(rv_run.get("rv_latency_ms", 0.0))
                metrics["rv_latency_ms"] = round(metrics["rv_latency_ms"], 2)
            target_item = str(assessment.item_id)
            target_questions = question_lib[target_item]
            target_question = next(
                (
                    str(candidate_question)
                    for candidate_question, entry in target_questions.items()
                    if isinstance(entry, dict)
                    and str(entry.get("label", "")).strip().lower()
                    == assessment.dimension.strip().lower()
                ),
                "1",
            )
            committed = bool(rv_run["completed"])
            if committed:
                _record_scored_segment(
                    question_lib=question_lib,
                    target_item=target_item,
                    target_question=target_question,
                    dimension=assessment.dimension,
                    score=2,
                    answer_text=assessment.text,
                    original_question=original_question_asked,
                    classification_label=assessment.dimension,
                    classification_value=2,
                    is_current_dimension=assessment.item_id == int(S),
                    allow_followup=False,
                    emotion_decision_override=emotion_decision,
                )
                outcome.covered_item_ids.add(assessment.item_id)
                validation_prefix = str(rv_run["validation_text"])
                last_rv_completed = True
                logger.info(
                    "Committed Score 2 after successful R-V for item %s (%s).",
                    assessment.item_id,
                    assessment.dimension,
                )
            else:
                validation_prefix = ""
                last_rv_completed = False
                logger.info(
                    "Left Score 2 provisional and uncommitted for item %s (%s).",
                    assessment.item_id,
                    assessment.dimension,
                )

            rv_note = list(rv_run["notes"])
            rv_note.extend(
                [
                    "provisional_score: 2",
                    "score_committed: " + str(committed).lower(),
                ]
            )
            question_lib[target_item][target_question].setdefault("notes", []).append(
                rv_note
            )

            if rv_run["terminate"]:
                terminate = 1
                break

        if last_rv_completed and not terminate:
            _set_pending_validation_text(validation_prefix)
            _set_pending_next_question_intro(FOLLOWUP_CONTINUE_TEXT)
            logger.info(
                "Queued the final R-V validation before the next screening question."
            )

    outcome.current_answered = int(S) in outcome.covered_item_ids
    valid = int(outcome.current_answered)

    return valid, terminate, previous_question, question_lib

def ask_question(
    question_lib,
    S: int,
    turn_records: List[Dict[str, Any]] = None,
    session_control=None,
) -> QuestionTurnOutcome:
        """
        Handles the RL loop for asking questions within a given item (S).
        Returns reward plus the dimensions actually covered by this question turn.
        """
        logger.info(f"Starting question RL loop for item S={S}.")
        question_reward = []
        DLA_terminate = 0
        
        previous_question = ""
        starting_score_counts = {
            int(item_id): len(item_questions.get("1", {}).get("score", []))
            for item_id, item_questions in question_lib.items()
            if str(item_id).isdigit() and isinstance(item_questions, dict)
        }

        def finish_question(
            reward: float,
            terminate: int,
            last_question: str,
        ) -> QuestionTurnOutcome:
            covered_item_ids = {
                int(item_id)
                for item_id, item_questions in question_lib.items()
                if str(item_id).isdigit()
                and isinstance(item_questions, dict)
                and len(item_questions.get("1", {}).get("score", []))
                > starting_score_counts.get(int(item_id), 0)
            }
            current_answered = bool(
                question_lib.get(str(S), {}).get("1", {}).get("score", [])
            )
            return QuestionTurnOutcome(
                reward=float(reward),
                terminate=int(terminate),
                previous_question=last_question,
                covered_item_ids=covered_item_ids,
                current_answered=current_answered,
            )

        
        # If there is only one question for this item, ask it directly
        question_A = "1"
        # Check if the score list for this item is empty (i.e., not answered yet)
        if len(question_lib[str(S)][str(question_A)]["score"]) == 0:
            # if the item is not answered yet, ask it directly
            if _run_late_emotion_followup_before_question(session_control):
                logger.info("Late emotion follow-up ended the screening flow before the next question.")
                return finish_question(0.0, 1, previous_question)
            
            # Get the number of available question variants for this item
            number_of_questions = len(question_lib[str(S)][str(question_A)]["question"])
            # Randomly select one question variant to ask
            choice_of_question = np.random.randint(number_of_questions)
            question_text = question_lib[str(S)][str(question_A)]["question"][choice_of_question]
            # Ask the vetted library wording directly. Runtime paraphrases can
            # drift polarity, which breaks the fixed Yes/No score mapping.
            # Concatenate the last question (context) with the current question
            question_text_ask = _compose_question_with_intro(question_text)
            # Log the question being asked
            log_question(question_text_ask)
            _publish_question_context(
                text=question_text_ask,
                item_id=S,
                question_index=question_A,
                dimension=question_lib[str(S)][str(question_A)]["label"],
                expects_response=True,
                source="screening",
            )
            # Get user input for the question
            _ , user_input = _get_answer_with_control(session_control)
            if _checkpoint_requested_skip_to_cbt(session_control):
                logger.info("Screening question interrupted by skip-to-CBT request.")
                return finish_question(0.0, 1, previous_question)
            # Classify and evaluate while retaining processing metrics.
            attempt_started_at = time.monotonic()
            dimension_label = question_lib[str(S)][str(question_A)]["label"]
            classification_result = classify_segments(
                user_input, question_text, dimension_label
            )
            attempt_metrics = dict(getattr(classification_result, "metrics", {}))
            attempt_metrics.setdefault("segment_count", len(user_input))
            attempt_metrics.setdefault("analyzer_call_count", len(classification_result))
            attempt_metrics.setdefault("analyzer_latency_ms", 0.0)
            attempt_metrics.setdefault("batch_fallback", False)
            DLA_result = [[label, score] for label, score in classification_result]
            score_before = list(question_lib[str(S)][str(question_A)]["score"])
            valid, DLA_terminate, previous_question, question_lib = evaluate_result(
                question_lib,
                DLA_result,
                S,
                question_A,
                user_input,
                question_text,
                session_control=session_control,
                metrics=attempt_metrics,
            )
            attempt_metrics["total_turn_latency_ms"] = round(
                (time.monotonic() - attempt_started_at) * 1000.0, 2
            )
            if turn_records is not None:
                turn_records.append(
                    build_question_attempt_record(
                        item_id=S,
                        question_index=question_A,
                        dimension=dimension_label,
                        question_text=question_text,
                        user_segments=user_input,
                        classification=DLA_result,
                        score_before=score_before,
                        score_after=question_lib[str(S)][str(question_A)]["score"],
                        valid=valid,
                        terminate=DLA_terminate,
                        attempt="initial",
                        triggered_reflection=bool(previous_question),
                        metrics=attempt_metrics,
                    )
                )
            # If the answer is invalid (valid == 0) and the process has not been terminated (DLA_terminate == 0), 
            # we may want to give the user a chance to clarify their response.
            # Only retry if DLA_result is empty or every (label, score) pair suggests NA or an unclassified response (score==99 or label=="NA").
            if valid == 0 and DLA_terminate == 0:
                # Generate a concise retry guide based on topic, original question, and original answer
                topic = question_lib[str(S)][str(question_A)]["label"]
                original_answer_text = " ".join(user_input) if user_input else ""
                guide_text = retry_guide(topic, question_text, original_answer_text)
                guide_text = _compose_question_with_intro(guide_text)
                # Show the guide to the user and collect a new response
                log_question(guide_text)
                _publish_question_context(
                    text=guide_text,
                    item_id=S,
                    question_index=question_A,
                    dimension=dimension_label,
                    expects_response=True,
                    source="retry_guide",
                )
                _ , user_input = _get_answer_with_control(session_control)
                if _checkpoint_requested_skip_to_cbt(session_control):
                    logger.info("Screening retry interrupted by skip-to-CBT request.")
                    return finish_question(0.0, 1, previous_question)
                # Re-run classification and evaluation with fresh retry metrics.
                attempt_started_at = time.monotonic()
                dimension_label = question_lib[str(S)][str(question_A)]["label"]
                classification_result = classify_segments(
                    user_input, question_text, dimension_label
                )
                attempt_metrics = dict(getattr(classification_result, "metrics", {}))
                attempt_metrics.setdefault("segment_count", len(user_input))
                attempt_metrics.setdefault("analyzer_call_count", len(classification_result))
                attempt_metrics.setdefault("analyzer_latency_ms", 0.0)
                attempt_metrics.setdefault("batch_fallback", False)
                DLA_result = [[label, score] for label, score in classification_result]
                score_before = list(question_lib[str(S)][str(question_A)]["score"])
                valid, DLA_terminate, previous_question, question_lib = evaluate_result(
                    question_lib,
                    DLA_result,
                    S,
                    question_A,
                    user_input,
                    question_text,
                    session_control=session_control,
                    metrics=attempt_metrics,
                )
                attempt_metrics["total_turn_latency_ms"] = round(
                    (time.monotonic() - attempt_started_at) * 1000.0, 2
                )
                if turn_records is not None:
                    turn_records.append(
                        build_question_attempt_record(
                            item_id=S,
                            question_index=question_A,
                            dimension=dimension_label,
                            question_text=question_text,
                            user_segments=user_input,
                            classification=DLA_result,
                            score_before=score_before,
                            score_after=question_lib[str(S)][str(question_A)]["score"],
                            valid=valid,
                            terminate=DLA_terminate,
                            attempt="retry",
                            triggered_reflection=bool(previous_question),
                            metrics=attempt_metrics,
                        )
                    )
        
        # Retrieve all scores for this question after answering
        all_score = question_lib[str(S)][str(question_A)]["score"]
        # Calculate the mean score if available, otherwise set to 0.0
        question_openai_res = np.mean(all_score) if all_score else 0.0
        # Append the result to the question reward list
        question_reward.append(question_openai_res)

        # Return the total reward, termination flag, and last question
        logger.info(f"Finished question RL loop for item S={S}. Total reward: {float(sum(question_reward))}, DLA_terminate: {int(DLA_terminate)}")
        result = finish_question(
            float(sum(question_reward)),
            int(DLA_terminate),
            previous_question,
        )
        logger.info("Question turn covered item ids: %s; current_answered=%s", sorted(result.covered_item_ids), result.current_answered)
        return result
