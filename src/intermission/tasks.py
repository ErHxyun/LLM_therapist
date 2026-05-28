"""Scripted mini-task content for CaiTI intermissions.

These tasks are intentionally deterministic and LLM-free. PHQ/GAD responses are
parsed by the runner and are not written to CaiTI's main records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class MiniTaskKind(str, Enum):
    SCREENING = "screening"
    BREATHING = "breathing"
    MINDFULNESS = "mindfulness"


@dataclass(frozen=True)
class ScreeningItem:
    item_id: str
    scale: str
    prompt: str


@dataclass(frozen=True)
class ScreeningScore:
    status: str
    score: int | None
    reason: str


@dataclass(frozen=True)
class ScriptedTask:
    task_id: str
    kind: MiniTaskKind
    text: str


_ANSWER_GUIDE = (
    "You can answer: not at all, several days, more than half the days, "
    "nearly every day, or skip."
)


def _private_prompt(question: str, *, first: bool = False) -> str:
    return f"Over the last two weeks, how often have you been bothered by {question}? {_ANSWER_GUIDE}"


_PHQ2_QUESTIONS = [
    "little interest or pleasure in doing things",
    "feeling down, depressed, or hopeless",
]


_GAD4_QUESTIONS = [
    "feeling nervous, anxious, or on edge",
    "not being able to stop or control worrying",
    "worrying too much about different things",
    "trouble relaxing",
]


SCREENING_ITEMS = [
    *[
        ScreeningItem(
            f"phq2_{index}",
            "phq2",
            _private_prompt(question, first=index == 1),
        )
        for index, question in enumerate(_PHQ2_QUESTIONS, start=1)
    ],
    *[
        ScreeningItem(
            f"gad4_{index}",
            "gad4",
            _private_prompt(question),
        )
        for index, question in enumerate(_GAD4_QUESTIONS, start=1)
    ],
]


BREATHING_TASKS = [
    ScriptedTask(
        "simple_breath_awareness",
        MiniTaskKind.BREATHING,
        (
            "Simple breath awareness. Sit comfortably with your spine relaxed but upright. "
            "Gently close your eyes, or soften your gaze. Bring your attention to your "
            "breathing. Notice the air moving in through your nose, and out again. There "
            "is no need to change the breath. Simply observe it. Feel the inhale filling "
            "the body slightly, and the exhale releasing. If your mind wanders, gently "
            "guide your attention back to the next breath. Let each inhale arrive "
            "naturally. Let each exhale soften the body a little more. Continue resting "
            "your awareness on the rhythm of breathing."
        ),
    ),
    ScriptedTask(
        "counting_the_breath",
        MiniTaskKind.BREATHING,
        (
            "Counting the breath. Sit comfortably and bring attention to your breathing. "
            "As you inhale, silently count one. As you exhale, count two. Inhale three. "
            "Exhale four. Continue counting up to ten, then begin again at one. If the "
            "mind drifts or you lose track, simply return to one without judgment. Allow "
            "the counting to anchor your attention to the steady rhythm of breathing."
        ),
    ),
    ScriptedTask(
        "expanding_breath_body",
        MiniTaskKind.BREATHING,
        (
            "Expanding breath through the body. Sit or lie down comfortably. Take a slow "
            "breath in and notice the chest gently expand. Exhale and feel the body "
            "soften. Now imagine the breath spreading through the body. As you inhale, "
            "feel the breath reaching the ribs, the back, and the belly. As you exhale, "
            "allow the shoulders and jaw to release any tension. Each breath expands "
            "awareness slightly through the body. Each exhale invites a sense of ease."
        ),
    ),
    ScriptedTask(
        "short_body_scan",
        MiniTaskKind.BREATHING,
        (
            "Short body scan. Sit or lie down comfortably and bring attention to the "
            "body. Notice the sensation of your feet touching the floor or surface "
            "beneath you. Move your awareness slowly up to the legs, simply noticing any "
            "sensations. Bring attention to the belly and chest, noticing the gentle "
            "movement of breathing. Now notice the shoulders, letting them soften if "
            "they are holding tension. Finally, bring awareness to the face, jaw, "
            "cheeks, and forehead, allowing them to relax. Rest for a few breaths, "
            "feeling the body as a whole."
        ),
    ),
    ScriptedTask(
        "breath_at_nostrils",
        MiniTaskKind.BREATHING,
        (
            "Breath at the nostrils. Sit comfortably and bring attention to the tip of "
            "your nose. Notice the subtle sensation of air entering the nostrils as you "
            "inhale. It may feel slightly cool. As you exhale, notice the warmth of the "
            "air leaving the body. Allow your attention to stay with these small "
            "sensations of breathing. When the mind wanders, gently return to the "
            "feeling of the breath at the nostrils. Remain with this simple awareness "
            "for the next few breaths."
        ),
    ),
]


MINDFULNESS_TASKS = []


def is_screening_skip(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    return bool(
        normalized in {"skip", "pass", "no", "no thanks", "not now", "prefer not"}
        or re.search(r"\b(skip|pass)\b", normalized)
    )


def parse_screening_score(text: str) -> int | None:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return None
    if "nearly every" in normalized or "almost every" in normalized:
        return 3
    if "more than half" in normalized or "half the days" in normalized:
        return 2
    if "several" in normalized or "some days" in normalized:
        return 1
    if "not at all" in normalized or normalized in {"never", "none", "zero", "0"}:
        return 0
    number_words = {"one": 1, "two": 2, "three": 3}
    if normalized in {"1", "2", "3"}:
        return int(normalized)
    if normalized in number_words:
        return number_words[normalized]
    return None


def classify_screening_response(text: str) -> ScreeningScore:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return ScreeningScore(status="UNRESOLVED", score=None, reason="empty_transcript")
    if is_screening_skip(normalized):
        return ScreeningScore(status="SKIPPED", score=None, reason="user_skip")
    score = parse_screening_score(normalized)
    if score is None:
        return ScreeningScore(status="UNRESOLVED", score=None, reason="unparsed_response")
    return ScreeningScore(status="ANSWERED", score=score, reason="")
