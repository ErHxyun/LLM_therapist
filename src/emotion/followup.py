from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from src.utils import config_loader


_RECENT_MAX = 50
_RECENT_MAX_AGE_SEC = 180.0
_recent_results: list[tuple[float, str, dict[str, Any]]] = []
_pending_late_followups: list["_PendingLateFollowup"] = []
_queued_late_followups: list["EmotionFollowupDecision"] = []
_condition = threading.Condition()

_UNRELIABLE_QUALITY_FLAGS = {
    "missing_audio",
    "has_audio_input:false",
    "audio_loaded_ok:false",
    "audio_quality_ok:false",
    "audio_valid_for_scoring:false",
}


@dataclass(frozen=True)
class EmotionFollowupSettings:
    enabled: bool
    wait_timeout_sec: float
    min_confidence: int
    risk_threshold: int
    light_risk_threshold: int
    late_followup_window_sec: float = 10.0


@dataclass
class EmotionFollowupDecision:
    should_follow_up: bool
    reason: str = ""
    followup_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PendingLateFollowup:
    created_at: float
    expires_at: float
    normalized_transcript: str
    dimension: str
    score: int
    user_text: str
    question_text: str
    settings: EmotionFollowupSettings


def build_emotion_followup_settings() -> EmotionFollowupSettings:
    return EmotionFollowupSettings(
        enabled=bool(config_loader.EMOTION_ENABLED and config_loader.EMOTION_ASSIST_FOLLOWUP_ENABLED),
        wait_timeout_sec=float(config_loader.EMOTION_ASSIST_WAIT_TIMEOUT_SEC),
        min_confidence=int(config_loader.EMOTION_ASSIST_MIN_CONFIDENCE),
        risk_threshold=int(config_loader.EMOTION_ASSIST_RISK_THRESHOLD),
        light_risk_threshold=int(config_loader.EMOTION_ASSIST_LIGHT_RISK_THRESHOLD),
        late_followup_window_sec=float(config_loader.EMOTION_ASSIST_LATE_FOLLOWUP_WINDOW_SEC),
    )


def normalize_transcript(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def register_emotion_result(record: dict[str, Any]) -> None:
    transcript = _record_transcript(record)
    normalized = normalize_transcript(transcript)
    if not normalized:
        return
    now = time.monotonic()
    with _condition:
        _recent_results.append((now, normalized, record))
        _prune_locked(now)
        _queue_matching_late_followups_locked(now, normalized, record)
        _condition.notify_all()


def wait_for_emotion_result(transcript: str, timeout_sec: float) -> dict[str, Any] | None:
    normalized = normalize_transcript(transcript)
    if not normalized:
        return None

    deadline = time.monotonic() + max(0.0, float(timeout_sec or 0.0))
    with _condition:
        while True:
            now = time.monotonic()
            _prune_locked(now)
            record = _find_result_locked(normalized)
            if record is not None:
                return record
            remaining = deadline - now
            if remaining <= 0:
                return None
            _condition.wait(timeout=remaining)


def clear_emotion_results_for_tests() -> None:
    with _condition:
        _recent_results.clear()
        _pending_late_followups.clear()
        _queued_late_followups.clear()


def queue_late_emotion_followup_request(
    *,
    dimension: str,
    score: int,
    user_text: str,
    question_text: str,
    settings: EmotionFollowupSettings,
    started_at: float | None = None,
) -> None:
    if not settings.enabled or settings.late_followup_window_sec <= 0:
        return
    normalized = normalize_transcript(user_text)
    if not normalized:
        return

    now = time.monotonic()
    start = started_at if started_at is not None else now
    expires_at = start + float(settings.late_followup_window_sec)
    if expires_at <= now:
        return

    pending = _PendingLateFollowup(
        created_at=now,
        expires_at=expires_at,
        normalized_transcript=normalized,
        dimension=str(dimension or ""),
        score=int(score),
        user_text=str(user_text or ""),
        question_text=str(question_text or ""),
        settings=settings,
    )
    with _condition:
        _pending_late_followups.append(pending)
        _prune_locked(now)
        existing_record = _find_result_locked(normalized)
        if existing_record is not None:
            _queue_matching_late_followups_locked(now, normalized, existing_record)


def pop_late_emotion_followup() -> EmotionFollowupDecision | None:
    now = time.monotonic()
    with _condition:
        _prune_locked(now)
        if not _queued_late_followups:
            return None
        return _queued_late_followups.pop(0)


def assess_emotion_followup(
    *,
    score: int,
    user_text: str,
    record: dict[str, Any] | None,
    settings: EmotionFollowupSettings | None = None,
) -> EmotionFollowupDecision:
    settings = settings or build_emotion_followup_settings()
    metadata = _extract_metadata(score=score, user_text=user_text, record=record)

    if not settings.enabled:
        return EmotionFollowupDecision(False, "disabled", metadata=metadata)
    if not record:
        return EmotionFollowupDecision(False, "no_emotion_result", metadata=metadata)
    if record.get("status") != "ok":
        return EmotionFollowupDecision(False, "emotion_result_error", metadata=metadata)
    if not _is_reliable(metadata, settings):
        return EmotionFollowupDecision(False, "unreliable_emotion_result", metadata=metadata)

    mismatch = bool(metadata["comparison_mismatch"])
    risk = int(metadata["credibility_risk"])
    audio_valence = metadata["audio_valence"]
    audio_scores = metadata["audio_scores"]
    distressed_voice = _distressed_voice(audio_valence, audio_scores)
    strong_distressed_voice = _strong_distressed_voice(audio_valence, audio_scores)
    positive_voice = _positive_voice(audio_valence)

    if score == 0 and mismatch and distressed_voice and risk >= settings.risk_threshold:
        metadata["rule"] = "score_0_distressed_voice"
        return EmotionFollowupDecision(
            True,
            "emotion_distress_hidden_by_low_content_score",
            "You said things are okay, and I just want to gently check in because your tone sounded a little tense. Is there anything else you want me to understand?",
            metadata,
        )

    if score == 0 and strong_distressed_voice and risk >= settings.risk_threshold:
        metadata["rule"] = "score_0_distressed_acoustic_cue"
        return EmotionFollowupDecision(
            True,
            "emotion_distress_acoustic_cue",
            "You said things are okay, and I just want to gently check in because your voice sounded a little strained. Is there anything else you want me to understand?",
            metadata,
        )

    if score >= 2 and positive_voice:
        metadata["rule"] = "score_2_positive_voice"
        return EmotionFollowupDecision(
            True,
            "emotion_positive_tone_with_high_content_score",
            "You described something pretty difficult, and I want to gently check whether the lighter tone means it feels manageable now, or if it is still weighing on you.",
            metadata,
        )

    if mismatch and risk >= settings.risk_threshold:
        metadata["rule"] = "light_mismatch"
        return EmotionFollowupDecision(
            True,
            "emotion_light_mismatch",
            "I want to gently check one more thing: does what you said match how you are feeling right now, or is there a little more to it?",
            metadata,
        )

    return EmotionFollowupDecision(False, "emotion_consistent_or_low_risk", metadata=metadata)


def _record_transcript(record: dict[str, Any]) -> str:
    request = record.get("request") if isinstance(record, dict) else {}
    response = record.get("response") if isinstance(record, dict) else {}
    if isinstance(request, dict) and request.get("transcript"):
        return str(request.get("transcript"))
    if isinstance(response, dict) and response.get("transcript"):
        return str(response.get("transcript"))
    if isinstance(record, dict) and record.get("transcript"):
        return str(record.get("transcript"))
    return ""


def _prune_locked(now: float) -> None:
    del _recent_results[:-_RECENT_MAX]
    while _recent_results and now - _recent_results[0][0] > _RECENT_MAX_AGE_SEC:
        del _recent_results[0]
    while _pending_late_followups and _pending_late_followups[0].expires_at <= now:
        del _pending_late_followups[0]
    del _queued_late_followups[:-_RECENT_MAX]


def _queue_matching_late_followups_locked(
    now: float,
    normalized: str,
    record: dict[str, Any],
) -> None:
    if not _pending_late_followups:
        return

    remaining: list[_PendingLateFollowup] = []
    for pending in _pending_late_followups:
        if pending.expires_at <= now:
            continue
        if not _transcripts_match(normalized, pending.normalized_transcript):
            remaining.append(pending)
            continue

        decision = assess_emotion_followup(
            score=pending.score,
            user_text=pending.user_text,
            record=record,
            settings=pending.settings,
        )
        if decision.should_follow_up:
            decision.metadata.update(
                {
                    "mode": "late_emotion_followup",
                    "dimension": pending.dimension,
                    "question_text": pending.question_text,
                    "late_followup_created_at": pending.created_at,
                }
            )
            _queued_late_followups.append(decision)

    _pending_late_followups[:] = remaining


def _find_result_locked(normalized: str) -> dict[str, Any] | None:
    for _, candidate, record in reversed(_recent_results):
        if _transcripts_match(normalized, candidate):
            return record
    return None


def _transcripts_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return len(left) >= 16 and len(right) >= 16 and (left in right or right in left)


def _extract_metadata(*, score: int, user_text: str, record: dict[str, Any] | None) -> dict[str, Any]:
    response = record.get("response", {}) if isinstance(record, dict) else {}
    response = response if isinstance(response, dict) else {}
    final_assessment = response.get("final_assessment", {})
    final_assessment = final_assessment if isinstance(final_assessment, dict) else {}
    final_result = response.get("final_result", {})
    final_result = final_result if isinstance(final_result, dict) else {}
    comparison = response.get("emotion_comparison", {})
    comparison = comparison if isinstance(comparison, dict) else {}
    audio_scores = response.get("audio_scores", {})
    audio_scores = audio_scores if isinstance(audio_scores, dict) else {}

    audio_emotion = response.get("audio_emotion")
    context_emotion = response.get("context_emotion")
    quality_flags = final_assessment.get("quality_flags") or []
    if not isinstance(quality_flags, list):
        quality_flags = [str(quality_flags)]

    comparison_mismatch = any(
        bool(comparison.get(key))
        for key in (
            "valence_conflict",
            "arousal_conflict",
            "contradiction_or_sarcasm",
        )
    ) or comparison.get("audio_vs_context_consistent") is False

    return {
        "score": int(score),
        "user_text": str(user_text or ""),
        "confidence": _int_value(final_assessment.get("confidence"), 0),
        "credibility_risk": _int_value(
            final_result.get("credibility_risk", final_assessment.get("credibility_risk")),
            0,
        ),
        "risk_level": final_result.get("risk_level", final_assessment.get("risk_level", "")),
        "quality_flags": [str(flag) for flag in quality_flags],
        "comparison_mismatch": bool(comparison_mismatch),
        "audio_emotion": audio_emotion,
        "context_emotion": context_emotion,
        "audio_valence": _emotion_valence(audio_emotion),
        "context_valence": _emotion_valence(context_emotion),
        "audio_scores": audio_scores,
        "emotion_note": comparison.get("note", ""),
    }


def _is_reliable(metadata: dict[str, Any], settings: EmotionFollowupSettings) -> bool:
    if metadata["confidence"] < settings.min_confidence:
        return False
    quality_flags = {str(flag).strip().lower() for flag in metadata["quality_flags"]}
    return not bool(quality_flags & _UNRELIABLE_QUALITY_FLAGS)


def _emotion_valence(value: Any) -> int | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _int_value(value[1], None)
    return None


def _distressed_voice(audio_valence: int | None, audio_scores: dict[str, Any]) -> bool:
    if audio_valence is not None and audio_valence < 0:
        return True
    tension = _int_value(audio_scores.get("tension"), 0)
    hesitation = _int_value(audio_scores.get("hesitation"), 0)
    stability = _int_value(audio_scores.get("stability"), 100)
    return tension >= 60 or hesitation >= 70 or stability <= 40


def _strong_distressed_voice(audio_valence: int | None, audio_scores: dict[str, Any]) -> bool:
    tension = _int_value(audio_scores.get("tension"), 0)
    hesitation = _int_value(audio_scores.get("hesitation"), 0)
    stability = _int_value(audio_scores.get("stability"), 100)
    negative_voice = audio_valence is not None and audio_valence < 0
    strained_delivery = hesitation >= 85 and stability <= 30
    tense_delivery = tension >= 70 and (hesitation >= 75 or stability <= 35)
    return (negative_voice and strained_delivery) or tense_delivery


def _positive_voice(audio_valence: int | None) -> bool:
    return audio_valence is not None and audio_valence > 0


def _int_value(value: Any, default: int | None) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default
