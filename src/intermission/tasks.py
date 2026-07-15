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
    text: str = ""
    steps: tuple["ScriptedStep", ...] = ()

    def iter_steps(self) -> tuple["ScriptedStep", ...]:
        if self.steps:
            return self.steps
        chunk = str(self.text or "").strip()
        if not chunk:
            return ()
        return (ScriptedStep(chunk),)


@dataclass(frozen=True)
class ScriptedStep:
    text: str
    pause_after_sec: float = 0.0


_ANSWER_GUIDE = (
    "You can answer: not at all, several days, more than half the days, "
    "nearly every day, or skip."
)

GUIDED_STEP_PAUSE_SEC = 4.0
GUIDED_BREATH_PAUSE_SEC = 5.5


def _step(text: str, pause_after_sec: float = 0.0) -> ScriptedStep:
    return ScriptedStep(text=str(text or "").strip(), pause_after_sec=max(0.0, float(pause_after_sec)))


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
        steps=(
            _step(
                "Simple breath awareness. Sit comfortably with your spine relaxed but upright. "
                "Gently close your eyes, or soften your gaze.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Bring your attention to your breathing. Notice the air moving in through "
                "your nose, and out again.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "There is no need to change the breath. Simply observe it. Feel the inhale "
                "filling the body slightly, and the exhale releasing.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "If your mind wanders, gently guide your attention back to the next breath. "
                "Let each inhale arrive naturally. Let each exhale soften the body a little more.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "Continue resting your awareness on the rhythm of breathing."
            ),
        ),
    ),
    ScriptedTask(
        "counting_the_breath",
        MiniTaskKind.BREATHING,
        steps=(
            _step(
                "Counting the breath. Sit comfortably and bring attention to your breathing.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "As you inhale, silently count one. As you exhale, count two. Inhale three. "
                "Exhale four.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "Continue counting up to ten, then begin again at one.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "If the mind drifts or you lose track, simply return to one without judgment.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Allow the counting to anchor your attention to the steady rhythm of breathing."
            ),
        ),
    ),
    ScriptedTask(
        "expanding_breath_body",
        MiniTaskKind.BREATHING,
        steps=(
            _step(
                "Expanding breath through the body. Sit or lie down comfortably.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Take a slow breath in and notice the chest gently expand. Exhale and feel the body soften.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "Now imagine the breath spreading through the body. As you inhale, feel the breath "
                "reaching the ribs, the back, and the belly.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "As you exhale, allow the shoulders and jaw to release any tension. Each breath "
                "expands awareness slightly through the body. Each exhale invites a sense of ease."
            ),
        ),
    ),
    ScriptedTask(
        "short_body_scan",
        MiniTaskKind.BREATHING,
        steps=(
            _step(
                "Short body scan. Sit or lie down comfortably and bring attention to the body.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Notice the sensation of your feet touching the floor or surface beneath you. "
                "Move your awareness slowly up to the legs, simply noticing any sensations.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Bring attention to the belly and chest, noticing the gentle movement of breathing.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "Now notice the shoulders, letting them soften if they are holding tension. "
                "Finally, bring awareness to the face, jaw, cheeks, and forehead, allowing them to relax.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Rest for a few breaths, feeling the body as a whole."
            ),
        ),
    ),
    ScriptedTask(
        "breath_at_nostrils",
        MiniTaskKind.BREATHING,
        steps=(
            _step(
                "Breath at the nostrils. Sit comfortably and bring attention to the tip of your nose.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Notice the subtle sensation of air entering the nostrils as you inhale. "
                "It may feel slightly cool.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "As you exhale, notice the warmth of the air leaving the body. Allow your attention "
                "to stay with these small sensations of breathing.",
                GUIDED_BREATH_PAUSE_SEC,
            ),
            _step(
                "When the mind wanders, gently return to the feeling of the breath at the nostrils. "
                "Remain with this simple awareness for the next few breaths."
            ),
        ),
    ),
]


MINDFULNESS_TASKS = [
    ScriptedTask(
        "five_senses_grounding",
        MiniTaskKind.MINDFULNESS,
        steps=(
            _step(
                "Five senses grounding. Sit comfortably and let your body settle where it is.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Notice five things you can see around you. Let your eyes move slowly, without rushing.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Now notice four things you can feel, such as your feet on the floor, your hands resting, or your clothing against your skin.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Notice three things you can hear. Listen for sounds near you, and sounds farther away.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Now notice two things you can smell, and one thing you can taste. Let yourself arrive more fully in this moment."
            ),
        ),
    ),
    ScriptedTask(
        "mindful_listening",
        MiniTaskKind.MINDFULNESS,
        steps=(
            _step(
                "Mindful listening. Sit comfortably and allow your shoulders to soften.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Bring your attention to the sounds around you. You do not need to search for anything special.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "Simply notice each sound as it appears and fades. Try not to judge it or name it too quickly.",
                GUIDED_STEP_PAUSE_SEC,
            ),
            _step(
                "If your mind wanders, gently return to listening. Let the sounds come and go while you stay present with them."
            ),
        ),
    ),
]


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
