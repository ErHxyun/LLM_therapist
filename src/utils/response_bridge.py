import re
from src.local_llm.types import LLMTask
from src.response_analyzer import (
    classify_dimension_and_score_result,
    classify_general_response_result,
)
from src.utils.log_util import get_logger
from src.utils.llm_output_contracts import (
    normalize_dim_score as _contract_normalize_dim_score,
    parse_dim_score_from_json_like as _contract_parse_from_json_like,
    parse_dim_score_from_text as _contract_parse_dim_score_from_text,
    parse_task2_prediction as _contract_parse_task2_prediction,
)
from src.utils.session_event_logger import log_llm_event
logger = get_logger("ResponseBridge")

_EXPLICIT_STOP_PATTERN = re.compile(
    r"("
    r"\b(let us|let's)\s+stop\b|"
    r"\bstop\s+(here|this|now|the conversation|the session)\b|"
    r"\bend\s+(the|this)\s+(conversation|session)\b|"
    r"\bno more questions\b|"
    r"\bi\s+(want|would like|need)\s+to\s+stop\b|"
    r"\bi\s+(do not|don't)\s+want\s+to\s+(continue|answer)\b|"
    r"\bi\s+(do not|don't)\s+want\s+to\s+talk\s+(anymore|about\s+(this|it)|to\s+you|with\s+you)\b|"
    r"\bplease\s+stop\b"
    r")",
    re.IGNORECASE,
)


def _normalize_general_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()


def _strip_outer_punctuation(text: str) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", text).strip()


def _contextual_general_label(user_input: str, dimension_label: str) -> str | None:
    s = _normalize_general_text(user_input)
    dim = str(dimension_label or "").strip().lower()

    if dim in {"work", "showup"} and re.search(
        r"\b(no|not|never|haven't|have not|didn't|did not)\s+"
        r"(miss|missed|missing|skip|skipped|absent)\b",
        s,
    ):
        return "Yes"
    if dim in {"work", "showup"} and re.search(
        r"\b(no|zero)\s+(missed\s+)?(days|classes|workdays|appointments)\b",
        s,
    ):
        return "Yes"

    return None


def is_explicit_stop_request(text: str) -> bool:
    s = _normalize_general_text(text)
    bare = _strip_outer_punctuation(s)
    if bare in {"stop", "quit", "exit", "cancel"}:
        return True
    return bool(_EXPLICIT_STOP_PATTERN.search(s))


def _heuristic_general_label(user_input: str) -> str | None:
    s = _normalize_general_text(user_input)
    bare = _strip_outer_punctuation(s)
    words = bare.split()

    exact_yes = {
        "yes", "yes i do", "yes i did", "yes i have", "yeah", "yep", "yup",
        "sure", "ok", "okay", "alright", "correct", "right", "absolutely",
        "definitely", "of course", "i do", "i did", "i have",
    }
    exact_no = {
        "no", "nope", "nah", "no i don't", "no i do not", "no i didn't",
        "no i did not", "not really", "not exactly", "i don't", "i do not",
        "i didn't", "i did not", "i don't think so", "i do not think so",
        "don't think so", "do not think so", "not at all", "none", "nothing",
    }
    exact_maybe = {
        "maybe", "perhaps", "possibly", "it depends", "as usual",
        "in between", "it's in between", "its in between",
    }

    if bare in exact_yes:
        return "Yes"
    if bare in exact_no:
        return "No"
    if bare in exact_maybe:
        return "Maybe"

    if re.match(r"^(yes|yeah|yep|yup)\b", bare) and len(words) <= 8:
        return "Yes"
    if re.match(r"^(no|nope|nah)\b", bare) and len(words) <= 8:
        return "No"
    if re.search(r"\b(i\s+)?do(?:n't| not)\s+think\s+so\b", bare) and len(words) <= 10:
        return "No"

    if is_explicit_stop_request(user_input):
        return "Stop"

    if any(phrase in s for phrase in (
        "i don't know", "i do not know", "i'm not sure", "im not sure",
        "i am not sure", "not sure", "unsure", "i doubt",
    )):
        return "Maybe"

    if any(phrase in s for phrase in (
        "what do you mean", "why do you ask", "can you explain",
        "could you explain", "i don't understand", "i do not understand",
        "i don't get it", "i do not get it",
    )):
        return "Question"

    if "?" in s and len(words) <= 12:
        return "Question"

    return None


def _strong_general_label(user_input: str) -> str | None:
    s = _normalize_general_text(user_input)
    bare = _strip_outer_punctuation(s)
    if not bare:
        return None

    if re.match(r"^(yes|yeah|yep|yup)\b", bare):
        conflict = re.search(
            r"\b(but|however|although|not|no|never|haven't|hasn't|don't|didn't|can't|cannot)\b",
            bare,
        )
        return None if conflict else "Yes"

    if re.match(r"^(no|nope|nah)\b", bare):
        conflict = re.search(r"\b(but|however|although|actually yes)\b", bare)
        return None if conflict else "No"
    if re.match(r"^(so\s+)?(i\s+)?do(?:n't| not)\s+think\s+so\b", bare):
        return "No"

    return None


def _looks_like_general_response(user_input: str) -> bool:
    return _heuristic_general_label(user_input) is not None or _strong_general_label(user_input) is not None


def _parse_task2_prediction(
    text: str,
    default: str | None = None,
    allow_numeric: bool = True,
) -> str | None:
    return _contract_parse_task2_prediction(text, default=default, allow_numeric=allow_numeric)


def _normalize_dim_score(dim: str, score: int):
    """
    Convert model labels such as 19_family_relationship into local CaiTI labels.
    Always validate score range.
    """
    logger.debug(f"Normalizing dimension and score: dim={dim}, score={score}")
    normalized = _contract_normalize_dim_score(dim, score)
    if not normalized:
        logger.warning(f"Invalid dimension-score contract: dim={dim}, score={score}")
    return normalized

def _parse_dim_score_from_text(text: str, validate_dimension: bool = True):
    """
    Parse '[dim][sep][score]' from a free-form text line.
    Supports formats like 'talk, 1', '3_talk, 1', 'DLA_3_talk, 1', etc.
    Accept separators: comma, colon, hyphen, or whitespace.
    """
    logger.debug(f"Parsing dimension-score from text: {text}")
    return _contract_parse_dim_score_from_text(text, validate_dimension=validate_dimension)

def _parse_from_json_like(raw: str, validate_dimension: bool = True):
    """
    If the model returns JSON-like content, try to extract:
    - {'res': '3_talk, 1'}
    - {'dimension': '3_talk', 'score': 1}
    """
    logger.debug(f"Trying to parse as JSON-like: {raw}")
    return _contract_parse_from_json_like(raw, validate_dimension=validate_dimension)


def _parse_legacy_support_output(raw: str) -> tuple[str, int] | None:
    parsed = _parse_dim_score_from_text(raw, validate_dimension=False)
    if not parsed:
        parsed = _parse_from_json_like(raw, validate_dimension=False)
    if parsed and parsed[0] == "support":
        return parsed
    return None


def _task1_fallback(
    *,
    user_input: str,
    original_question: str,
    raw: str | None,
    source: str,
    dimension_label: str,
    **metadata,
):
    log_llm_event(
        task=LLMTask.TASK1_RESPONSE_ANALYZER,
        dimension="NA",
        score=99,
        segment_text=user_input,
        question_text=original_question,
        raw_llm_output=raw,
        normalized_output="NA, 99",
        metadata={"source": source, "current_dimension": dimension_label, **metadata},
    )
    return "NA", 99


def _task1_dimension_result(
    *,
    dim: str,
    score: int,
    user_input: str,
    original_question: str,
    raw: str | None,
    source: str,
    dimension_label: str,
):
    log_llm_event(
        task=LLMTask.TASK1_RESPONSE_ANALYZER,
        dimension=dim,
        score=score,
        segment_text=user_input,
        question_text=original_question,
        raw_llm_output=raw,
        normalized_output=f"{dim}, {score}",
        metadata={
            "source": source,
            "current_dimension": dimension_label,
            "cross_dimension": str(dim).strip().lower() != str(dimension_label).strip().lower(),
        },
    )
    return dim, score


def classify_with_task2(user_input: str, dimension_label: str):
    contextual_default = _contextual_general_label(user_input, dimension_label)
    heuristic_default = contextual_default or _heuristic_general_label(user_input)
    strong_default = None if contextual_default else _strong_general_label(user_input)
    raw = None
    model_error = None
    category = None
    try:
        contract = classify_general_response_result(user_input, default=heuristic_default)
        raw = contract.raw_output
        logger.debug(f"Task2 raw: {raw}")
        category = contract.category if contract.is_valid else None
    except Exception as e:
        model_error = e
        logger.debug(f"classify_general_response exception: {e}")

    if category:
        if contextual_default and category != contextual_default:
            logger.info(
                "Overriding task2 category %s with contextual heuristic %s.",
                category,
                contextual_default,
            )
            category = contextual_default
        if category == "Stop" and not is_explicit_stop_request(user_input):
            logger.info("Stop guard rejected task2 Stop for non-explicit user text.")
            category = strong_default or heuristic_default
            if category == "Stop":
                category = None
            if not category:
                return None
        if strong_default and category != strong_default:
            logger.info(
                "Overriding task2 category %s with strong user-language heuristic %s.",
                category,
                strong_default,
            )
            category = strong_default
        logger.debug(f"Task2 category '{category}' detected; binding to dimension '{dimension_label}'")
        log_llm_event(
            task=LLMTask.TASK2_GENERAL_RESPONSE,
            dimension=dimension_label,
            score=category,
            segment_text=user_input,
            raw_llm_output=raw,
            normalized_output=f"{dimension_label}, {category}",
            metadata={
                "contextual_default": contextual_default,
                "heuristic_default": heuristic_default,
                "strong_default": strong_default,
            },
        )
        return dimension_label, category

    fallback = strong_default or heuristic_default
    if fallback:
        logger.debug(f"Task2 fallback heuristic '{fallback}' used; binding to dimension '{dimension_label}'")
        log_llm_event(
            task=LLMTask.TASK2_GENERAL_RESPONSE,
            dimension=dimension_label,
            score=fallback,
            segment_text=user_input,
            raw_llm_output=raw,
            normalized_output=f"{dimension_label}, {fallback}",
            metadata={
                "heuristic_default": heuristic_default,
                "strong_default": strong_default,
                "model_error": str(model_error) if model_error else None,
                "source": "heuristic_fallback",
            },
        )
        return dimension_label, fallback
    return None


def get_openai_resp(user_input, original_question, dimension_label: str):
    """
    Main entry point to process model response or user input and extract a unified tuple.
    For general Yes/No/Stop/Maybe/Question answers, returns (dimension_label, Keyword).
    Otherwise, attempts to return (dimension, score:int) parsed from model output.
    Fallbacks to ('NA', 99) on parse failure.
    """
    if _looks_like_general_response(user_input):
        general = classify_with_task2(user_input, dimension_label)
        if general:
            return general

    try:
        # Use the response analyzer to try to classify the input
        contract = classify_dimension_and_score_result(user_input, original_question)
        raw = contract.raw_output
        # Take just the first line (in case of multi-line output)
        first = str(raw).strip().splitlines()[0].strip()
        logger.debug(f"Response analyzer raw: {raw}")
        logger.debug(f"First line parsed: {first}")
    except Exception as e:
        # Log failure for diagnostics, fallback code
        logger.debug(f"classify_dimension_and_score exception: {e}")
        return _task1_fallback(
            user_input=user_input,
            original_question=original_question,
            raw=None,
            source="model_error",
            dimension_label=dimension_label,
            model_error=str(e),
        )

    # Some response-analyzer outputs still surface general labels directly.
    category = _parse_task2_prediction(first, allow_numeric=False)
    if category:
        if category == "Stop" and not is_explicit_stop_request(user_input):
            logger.info("Stop guard rejected task1 Stop for non-explicit user text.")
            return _task1_fallback(
                user_input=user_input,
                original_question=original_question,
                raw=raw,
                source="stop_guard",
                dimension_label=dimension_label,
                rejected_category=category,
            )
        logger.debug(f"General token '{category}' detected; binding to dimension '{dimension_label}'")
        log_llm_event(
            task=LLMTask.TASK1_RESPONSE_ANALYZER,
            dimension=dimension_label,
            score=category,
            segment_text=user_input,
            question_text=original_question,
            raw_llm_output=raw,
            normalized_output=f"{dimension_label}, {category}",
            metadata={"source": "task1_general_token"},
        )
        return dimension_label, category


    legacy_support = (
        _parse_legacy_support_output(first)
        or _parse_legacy_support_output(str(raw))
    )
    if legacy_support:
        current_dimension = str(dimension_label).strip().lower()
        if current_dimension in {"family_support", "social_support"}:
            return _task1_dimension_result(
                dim=current_dimension,
                score=legacy_support[1],
                user_input=user_input,
                original_question=original_question,
                raw=raw,
                source="legacy_support_current_dimension",
                dimension_label=dimension_label,
            )
        logger.info(
            "Rejected ambiguous legacy support score for current dimension %s.",
            dimension_label,
        )
        return _task1_fallback(
            user_input=user_input,
            original_question=original_question,
            raw=raw,
            source="ambiguous_legacy_support",
            dimension_label=dimension_label,
            rejected_dimension="support",
            rejected_score=legacy_support[1],
        )

    if contract.is_valid:
        return _task1_dimension_result(
            dim=contract.dimension,
            score=contract.score,
            user_input=user_input,
            original_question=original_question,
            raw=raw,
            source=contract.source,
            dimension_label=dimension_label,
        )

    # Maybe it's a plain-text dimension,score (e.g. 'talk, 1' or '3_talk, 1').
    got = _parse_dim_score_from_text(first)
    logger.debug(f"Parsed from text: {got}")
    if got:
        return _task1_dimension_result(
            dim=got[0],
            score=got[1],
            user_input=user_input,
            original_question=original_question,
            raw=raw,
            source="text_parse",
            dimension_label=dimension_label,
        )
    
    # Try to parse result in case it's JSON-ish (either first line or whole raw)
    got = _parse_from_json_like(first)
    logger.debug(f"Parsed first linefrom json-like: {got}")
    if not got:
        got = _parse_from_json_like(str(raw))
        logger.debug(f"Parsed whole raw answer from json-like: {got}")
    if got:
        return _task1_dimension_result(
            dim=got[0],
            score=got[1],
            user_input=user_input,
            original_question=original_question,
            raw=raw,
            source="json_like_parse",
            dimension_label=dimension_label,
        )

    # If response is 'Other, N', always fallback to NA,99
    m = re.match(r"^\s*(Other)\s*,\s*(\d+)\s*$", first, flags=re.IGNORECASE)
    if m:
        logger.debug(f"Response is 'Other, {m.group(2)}', fallback to NA,99")
        return _task1_fallback(
            user_input=user_input,
            original_question=original_question,
            raw=raw,
            source="other_fallback",
            dimension_label=dimension_label,
        )

    # If all else fails, fallback
    logger.debug("Failed to parse classification, fallback to NA,99")
    return _task1_fallback(
        user_input=user_input,
        original_question=original_question,
        raw=raw,
        source="parse_failure",
        dimension_label=dimension_label,
    )
