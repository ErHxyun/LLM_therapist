import os
import re
import logging
import time
import threading
import pandas as pd
from pandas.errors import EmptyDataError
from src.runtime.status_monitor import get_active_status_monitor
from src.utils.config_loader import RECORD_CSV

# Set up logger for this module
from src.utils.log_util import get_logger
logger = get_logger("IORecord")

HEADER = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
NO_RESPONSE_PREFIX = "__CAITI_NO_RESPONSE__"
LONG_RESPONSE_PREFIX = "__CAITI_LONG_RESPONSE__"


class _ResponseProfilePrompt(str):
    def __new__(cls, value: str, response_profile: str):
        instance = super().__new__(cls, str(value or ""))
        instance.response_profile = response_profile
        return instance


def long_response_prompt(text: str) -> str:
    return _ResponseProfilePrompt(text, "long")


RECORD_LOCK = threading.RLock()

# Module-level buffer to prepend content to the next question output.
# When non-empty, its content will be combined with the next question
# using two newline characters as the separator, then cleared.
_PENDING_QUESTION_PREFIX = ""


def _publish_prompt_status(text: str, expects_response: bool) -> None:
    monitor = get_active_status_monitor()
    if monitor is None:
        return
    method = getattr(monitor, "set_prompt", None)
    if not callable(method):
        return
    try:
        method(text=str(text or ""), source="record", expects_response=bool(expects_response))
    except Exception:
        return


def _publish_response_status(text: str, source: str = "record") -> None:
    monitor = get_active_status_monitor()
    if monitor is None:
        return
    method = getattr(monitor, "set_response", None)
    if not callable(method):
        return
    try:
        method(text=str(text or ""), source=source)
    except Exception:
        return

def set_question_prefix(text: str):
    """
    Set a pending prefix that will be prepended to the next question output.
    The prefix will be combined with two newlines between the prefix and the question.
    """
    global _PENDING_QUESTION_PREFIX
    _PENDING_QUESTION_PREFIX = str(text) if text is not None else ""


def append_question_prefix(text: str):
    """
    Append text to the pending prefix that will be prepended to the next question output.
    """
    global _PENDING_QUESTION_PREFIX
    text = str(text) if text is not None else ""
    if not text:
        return
    if _PENDING_QUESTION_PREFIX:
        _PENDING_QUESTION_PREFIX = f"{_PENDING_QUESTION_PREFIX}\n\n{text}"
    else:
        _PENDING_QUESTION_PREFIX = text


def _read():
    last_exc = None
    for _ in range(5):
        try:
            time.sleep(0.03)
            with RECORD_LOCK:
                return pd.read_csv(RECORD_CSV, dtype={"Question": str, "Question_Lock": "int64", "Resp": str, "Resp_Lock": "int64"})
        except (EmptyDataError, FileNotFoundError, OSError) as e:
            last_exc = e
            time.sleep(0.05)
    raise last_exc

def _write(df):
    time.sleep(0.03)
    with RECORD_LOCK:
        folder = os.path.dirname(RECORD_CSV)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        tmp_path = f"{RECORD_CSV}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
        df.to_csv(tmp_path, columns=HEADER, index=False)
        os.replace(tmp_path, RECORD_CSV)
    time.sleep(0.03)

def _log_question_record(text: str, expects_response: bool):
    response_profile = getattr(text, "response_profile", "standard")
    visible_text = str(text or "")

    while True:
        time.sleep(0.1)
        with RECORD_LOCK:
            data = _read()
            if data.loc[0, "Question_Lock"] == 0:
                # If there is a pending prefix (e.g., RV validation), combine it with the question
                global _PENDING_QUESTION_PREFIX
                combined = visible_text
                if _PENDING_QUESTION_PREFIX:
                    combined = f"{_PENDING_QUESTION_PREFIX}\n\n{visible_text}"
                    logger.info("Combining pending prefix with next question using two newlines.")
                record_text = combined
                if not expects_response:
                    record_text = f"{NO_RESPONSE_PREFIX}\n{combined}"
                elif response_profile == "long":
                    record_text = f"{LONG_RESPONSE_PREFIX}\n{combined}"
                data.loc[0, "Question"] = record_text
                data.loc[0, "Question_Lock"] = 1
                _write(data)
                # Clear the prefix once consumed
                _PENDING_QUESTION_PREFIX = ""
                logger.info(f"Prompted question: {combined}")
                _publish_prompt_status(combined, expects_response)
                break


def log_question(text: str):
    _log_question_record(text, expects_response=True)


def log_system_message(text: str):
    """Log a user-facing message that should be spoken without collecting STT."""
    _log_question_record(text, expects_response=False)

_SEGMENT_BOUNDARY_RE = re.compile(
    r"(?<!\d)\.(?!\d)|[!?;]+|,\s*(?:and|but)\b\s*|\s+\bbut\b\s+",
    flags=re.IGNORECASE,
)


def segment_user_response(user_input: str) -> list[str]:
    """Split independent clauses without matching substrings such as butter."""
    segments = []
    for part in _SEGMENT_BOUNDARY_RE.split(str(user_input or "")):
        cleaned = " ".join(part.split()).strip()
        if cleaned:
            segments.append(cleaned)
    return segments


def get_answer(should_stop=None):
    while True:
        if callable(should_stop) and should_stop():
            logger.info("Stopped waiting for segmented answer because stop condition was met.")
            return [], []
        time.sleep(0.1)
        with RECORD_LOCK:
            data = _read()
            if data.loc[0, "Resp_Lock"] == 0:
                user_input = data.loc[0, "Resp"]
                data.loc[0, "Resp_Lock"] = 1
                _write(data)
                _publish_response_status(user_input)
                break
    segments = segment_user_response(user_input)
    return [], segments

def get_resp_log(should_stop=None):
    while True:
        if callable(should_stop) and should_stop():
            logger.info("Stopped waiting for user response because stop condition was met.")
            return ""
        time.sleep(0.1)
        with RECORD_LOCK:
            data = _read()
            if data.loc[0, "Resp_Lock"] == 0:
                user_response = data.loc[0, "Resp"]
                data.loc[0, "Resp_Lock"] = 1
                _write(data)
                logger.info(f"Received user response: {user_response}")
                _publish_response_status(user_response)
                break
    return user_response

def init_record():
    with RECORD_LOCK:
        try:
            data = _read()
        except FileNotFoundError:
            data = pd.DataFrame([["", 0, "", 1]], columns=HEADER)
            _write(data)
        time.sleep(0.03)
        data.loc[0, 'Question_Lock'] = 0
        data.loc[0, 'Resp_Lock'] = 1
        _write(data)


def reset_record_state():
    global _PENDING_QUESTION_PREFIX
    with RECORD_LOCK:
        try:
            data = _read()
        except FileNotFoundError:
            data = pd.DataFrame([["", 0, "", 1]], columns=HEADER)
        data.loc[0, "Question"] = ""
        data.loc[0, "Question_Lock"] = 0
        data.loc[0, "Resp"] = ""
        data.loc[0, "Resp_Lock"] = 1
        _write(data)
        _PENDING_QUESTION_PREFIX = ""
