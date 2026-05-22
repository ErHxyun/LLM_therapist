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


def _normalize_general_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()


def _strip_outer_punctuation(text: str) -> str:
    return re.sub(r"^[\W_]+|[\W_]+$", "", text).strip()


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
        "i didn't", "i did not",
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

    if any(phrase in s for phrase in (
        "let us stop", "let's stop", "stop here", "stop this",
        "end the conversation", "end this conversation", "no more questions",
        "i want to stop", "i don't want to continue", "i do not want to continue",
        "i don't want to answer", "i do not want to answer",
    )):
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


def _looks_like_general_response(user_input: str) -> bool:
    return _heuristic_general_label(user_input) is not None


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

def _parse_dim_score_from_text(text: str):
    """
    Parse '[dim][sep][score]' from a free-form text line.
    Supports formats like 'talk, 1', '3_talk, 1', 'DLA_3_talk, 1', etc.
    Accept separators: comma, colon, hyphen, or whitespace.
    """
    logger.debug(f"Parsing dimension-score from text: {text}")
    return _contract_parse_dim_score_from_text(text)

def _parse_from_json_like(raw: str):
    """
    If the model returns JSON-like content, try to extract:
    - {'res': '3_talk, 1'}
    - {'dimension': '3_talk', 'score': 1}
    """
    logger.debug(f"Trying to parse as JSON-like: {raw}")
    return _contract_parse_from_json_like(raw)


def classify_with_task2(user_input: str, dimension_label: str):
    heuristic_default = _heuristic_general_label(user_input)
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
        logger.debug(f"Task2 category '{category}' detected; binding to dimension '{dimension_label}'")
        log_llm_event(
            task=LLMTask.TASK2_GENERAL_RESPONSE,
            dimension=dimension_label,
            score=category,
            segment_text=user_input,
            raw_llm_output=raw,
            normalized_output=f"{dimension_label}, {category}",
            metadata={"heuristic_default": heuristic_default},
        )
        return dimension_label, category

    if heuristic_default:
        logger.debug(f"Task2 fallback heuristic '{heuristic_default}' used; binding to dimension '{dimension_label}'")
        log_llm_event(
            task=LLMTask.TASK2_GENERAL_RESPONSE,
            dimension=dimension_label,
            score=heuristic_default,
            segment_text=user_input,
            raw_llm_output=raw,
            normalized_output=f"{dimension_label}, {heuristic_default}",
            metadata={
                "heuristic_default": heuristic_default,
                "model_error": str(model_error) if model_error else None,
                "source": "heuristic_fallback",
            },
        )
        return dimension_label, heuristic_default
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
        log_llm_event(
            task=LLMTask.TASK1_RESPONSE_ANALYZER,
            dimension=dimension_label,
            score=99,
            segment_text=user_input,
            question_text=original_question,
            raw_llm_output=None,
            normalized_output="NA, 99",
            metadata={"model_error": str(e)},
        )
        return "NA", 99

    # Some response-analyzer outputs still surface general labels directly.
    category = _parse_task2_prediction(first, allow_numeric=False)
    if category:
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

    if contract.is_valid:
        log_llm_event(
            task=LLMTask.TASK1_RESPONSE_ANALYZER,
            dimension=contract.dimension,
            score=contract.score,
            segment_text=user_input,
            question_text=original_question,
            raw_llm_output=raw,
            normalized_output=contract.normalized_output,
            metadata={"source": contract.source, "current_dimension": dimension_label},
        )
        return contract.dimension, contract.score

    # Maybe it's a plain-text dimension,score (e.g. 'talk, 1' or '3_talk, 1').
    got = _parse_dim_score_from_text(first)
    logger.debug(f"Parsed from text: {got}")
    if got:
        log_llm_event(
            task=LLMTask.TASK1_RESPONSE_ANALYZER,
            dimension=got[0],
            score=got[1],
            segment_text=user_input,
            question_text=original_question,
            raw_llm_output=raw,
            normalized_output=f"{got[0]}, {got[1]}",
            metadata={"source": "text_parse", "current_dimension": dimension_label},
        )
        return got
    
    # Try to parse result in case it's JSON-ish (either first line or whole raw)
    got = _parse_from_json_like(first)
    logger.debug(f"Parsed first linefrom json-like: {got}")
    if not got:
        got = _parse_from_json_like(str(raw))
        logger.debug(f"Parsed whole raw answer from json-like: {got}")
    if got:
        log_llm_event(
            task=LLMTask.TASK1_RESPONSE_ANALYZER,
            dimension=got[0],
            score=got[1],
            segment_text=user_input,
            question_text=original_question,
            raw_llm_output=raw,
            normalized_output=f"{got[0]}, {got[1]}",
            metadata={"source": "json_like_parse", "current_dimension": dimension_label},
        )
        return got

    # If response is 'Other, N', always fallback to NA,99
    m = re.match(r"^\s*(Other)\s*,\s*(\d+)\s*$", first, flags=re.IGNORECASE)
    if m:
        logger.debug(f"Response is 'Other, {m.group(2)}', fallback to NA,99")
        log_llm_event(
            task=LLMTask.TASK1_RESPONSE_ANALYZER,
            dimension="NA",
            score=99,
            segment_text=user_input,
            question_text=original_question,
            raw_llm_output=raw,
            normalized_output="NA, 99",
            metadata={"source": "other_fallback", "current_dimension": dimension_label},
        )
        return "NA", 99

    # If all else fails, fallback
    logger.debug("Failed to parse classification, fallback to NA,99")
    log_llm_event(
        task=LLMTask.TASK1_RESPONSE_ANALYZER,
        dimension="NA",
        score=99,
        segment_text=user_input,
        question_text=original_question,
        raw_llm_output=raw,
        normalized_output="NA, 99",
        metadata={"source": "parse_failure", "current_dimension": dimension_label},
    )
    return "NA", 99
