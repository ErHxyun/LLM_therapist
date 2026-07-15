"""Output contracts for CaiTI analytical LLM modules."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


GENERAL_CATEGORIES = ("Yes", "No", "Maybe", "Question", "Stop")

LOCAL_DIMENSIONS = (
    "weight",
    "mood",
    "medication",
    "care",
    "house",
    "talk",
    "emo",
    "safe",
    "risk",
    "sleep",
    "eat",
    "work",
    "work_dayoff",
    "showup",
    "finance",
    "nutrition",
    "problem",
    "family_support",
    "family",
    "alcohol",
    "ciga",
    "drug",
    "hobbies",
    "creativity",
    "community",
    "social_support",
    "social",
    "comfortable",
    "protection",
    "productivity",
    "work_motivation",
    "coping",
    "sib",
    "arrest",
    "legal",
    "hygiene",
    "sports",
)

MODEL_LABEL_TO_LOCAL_LABEL = {
    "1_weight": "weight",
    "2_mood": "mood",
    "3_medication": "medication",
    "4_care": "care",
    "5_house": "house",
    "6_talk": "talk",
    "7_emo": "emo",
    "8_safe": "safe",
    "9_risk": "risk",
    "10_sleep": "sleep",
    "11_eat": "eat",
    "12_work": "work",
    "13_work_life": "work_dayoff",
    "14_show_up": "showup",
    "15_finance": "finance",
    "16_nutrition": "nutrition",
    "17_problem": "problem",
    "18_support": "family_support",
    "19_family_relationship": "family",
    "20_alcohol": "alcohol",
    "21_tobacco": "ciga",
    "22_substance": "drug",
    "23_leisure": "hobbies",
    "24_creativity": "creativity",
    "25_community": "community",
    "26_support_social": "social_support",
    "26_support": "social_support",
    "27_relationship_social": "social",
    "28_boundary": "comfortable",
    "29_safe_sex": "protection",
    "30_productivity": "productivity",
    "31_motivation": "work_motivation",
    "32_coping": "coping",
    "33_self_harm": "sib",
    "34_lawful": "arrest",
    "35_legal": "legal",
    "36_hygiene": "hygiene",
    "37_exercise": "sports",
}

LOCAL_LABEL_ALIASES = {
    "family_relationship": "family",
    "work_life": "work_dayoff",
    "show_up": "showup",
    "tobacco": "ciga",
    "substance": "drug",
    "leisure": "hobbies",
    "support_social": "social_support",
    "relationship_social": "social",
    "boundary": "comfortable",
    "safe_sex": "protection",
    "motivation": "work_motivation",
    "self_harm": "sib",
    "lawful": "arrest",
    "exercise": "sports",
}


@dataclass(frozen=True)
class DimScoreContract:
    raw_output: str | None
    normalized_output: str
    dimension: str
    score: int
    is_valid: bool
    source: str


@dataclass(frozen=True)
class CategoryContract:
    raw_output: str | None
    normalized_output: str
    category: str | None
    is_valid: bool
    source: str


@dataclass(frozen=True)
class DecisionContract:
    raw_output: str | None
    normalized_output: str
    decision: str
    is_valid: bool
    source: str


def normalize_dim_score(dim: str, score: int, validate_dimension: bool = True) -> tuple[str, int] | None:
    dim = str(dim or "").strip()
    dim = re.sub(r"^DLA_", "", dim, flags=re.IGNORECASE)
    dim = re.sub(r"\s+", "_", dim).lower()

    if dim in MODEL_LABEL_TO_LOCAL_LABEL:
        dim = MODEL_LABEL_TO_LOCAL_LABEL[dim]
    else:
        match = re.match(r"^\d+_(.+)$", dim)
        if match:
            dim = match.group(1)
        dim = LOCAL_LABEL_ALIASES.get(dim, dim)

    if not isinstance(score, int) or score not in {0, 1, 2}:
        return None
    if validate_dimension and dim not in LOCAL_DIMENSIONS:
        return None
    return dim, score


def parse_dim_score_from_text(text: str, validate_dimension: bool = True) -> tuple[str, int] | None:
    raw = str(text or "")
    match = re.search(
        r"\b((?:DLA_)?(?:\d+_)?[A-Za-z_]+)\s*[,:\-\s]\s*([0-2])\b",
        raw,
    )
    if not match:
        return None
    return normalize_dim_score(match.group(1), int(match.group(2)), validate_dimension)


def parse_dim_score_from_json_like(raw: str, validate_dimension: bool = True) -> tuple[str, int] | None:
    text = str(raw or "").strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except Exception:
        return parse_dim_score_from_text(text, validate_dimension)

    if not isinstance(data, dict):
        return None
    keys = {str(key).lower(): value for key, value in data.items()}
    if "res" in keys:
        parsed = parse_dim_score_from_text(str(keys["res"]), validate_dimension)
        if parsed:
            return parsed
    if "dimension" in keys and "score" in keys:
        try:
            return normalize_dim_score(str(keys["dimension"]), int(keys["score"]), validate_dimension)
        except Exception:
            return None
    return parse_dim_score_from_text(str(data), validate_dimension)


def normalize_task1_output(raw_text: str | None) -> DimScoreContract:
    raw = "" if raw_text is None else str(raw_text)
    parsed = parse_dim_score_from_text(raw)
    source = "text_parse"
    if not parsed:
        parsed = parse_dim_score_from_json_like(raw)
        source = "json_like_parse"
    if parsed:
        dim, score = parsed
        return DimScoreContract(
            raw_output=raw_text,
            normalized_output=f"{dim}, {score}",
            dimension=dim,
            score=score,
            is_valid=True,
            source=source,
        )
    return DimScoreContract(
        raw_output=raw_text,
        normalized_output="NA, 99",
        dimension="NA",
        score=99,
        is_valid=False,
        source="parse_failure",
    )


def parse_task2_prediction(
    text: str | None,
    default: str | None = None,
    allow_numeric: bool = True,
) -> str | None:
    raw = str(text or "").strip()
    for category in GENERAL_CATEGORIES:
        if re.search(rf"\b{re.escape(category)}\b", raw, flags=re.IGNORECASE):
            return category

    if allow_numeric:
        match = re.search(r"\b([1-5])\b", raw)
        if match:
            return GENERAL_CATEGORIES[int(match.group(1)) - 1]

    return default if default in GENERAL_CATEGORIES else None


def normalize_task2_output(raw_text: str | None, default: str | None = None) -> CategoryContract:
    raw = "" if raw_text is None else str(raw_text)
    category = parse_task2_prediction(raw, default=default)
    if category:
        return CategoryContract(
            raw_output=raw_text,
            normalized_output=category,
            category=category,
            is_valid=True,
            source="category_parse",
        )
    return CategoryContract(
        raw_output=raw_text,
        normalized_output="UNPARSED",
        category=None,
        is_valid=False,
        source="parse_failure",
    )


def parse_binary_decision(text: str | None, default: str = "1") -> str:
    if default not in {"0", "1"}:
        raise ValueError("default must be '0' or '1'")
    raw = str(text or "").strip()
    match = re.search(r"DECISION\s*[:=]\s*([01])", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b([01])\b", raw)
    if match:
        return match.group(1)
    return default


def normalize_decision_output(raw_text: str | None, default: str = "1") -> DecisionContract:
    raw = "" if raw_text is None else str(raw_text)
    decision = parse_binary_decision(raw, default=default)
    is_valid = bool(re.search(r"DECISION\s*[:=]\s*([01])", raw, flags=re.IGNORECASE))
    is_valid = is_valid or bool(re.search(r"\b([01])\b", raw))
    return DecisionContract(
        raw_output=raw_text,
        normalized_output=f"DECISION: {decision}",
        decision=decision,
        is_valid=is_valid,
        source="decision_parse" if is_valid else "default_fallback",
    )
