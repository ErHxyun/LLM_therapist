"""Queue- and DB-backed interaction bridge for both text and voice CaiTI loops."""
import json
import os
import queue

from src.drivers.db_manager import DBManager
from src.utils.config_loader import DB_PATH, RECORD_CSV, SUBJECT_ID
from src.utils.log_util import get_logger

logger = get_logger("IORecord")

OUTPUT_QUEUE = queue.Queue()
INPUT_QUEUE = queue.Queue()

DB = None
SESSION_ID = None
CURRENT_TURN_INDEX = 0
USER_CONTEXT = ""
_PENDING_QUESTION_PREFIX = ""


def get_user_context():
    return USER_CONTEXT


def set_question_prefix(text: str):
    global _PENDING_QUESTION_PREFIX
    _PENDING_QUESTION_PREFIX = str(text) if text is not None else ""


def _ensure_record_csv():
    folder = os.path.dirname(RECORD_CSV)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(RECORD_CSV):
        with open(RECORD_CSV, "w", encoding="utf-8") as handle:
            handle.write("Timestamp,Type,Speaker,Text\n")


def _append_csv(log_type: str, speaker: str, text: str):
    _ensure_record_csv()
    import datetime

    escaped = str(text).replace('"', '""')
    with open(RECORD_CSV, "a", encoding="utf-8") as handle:
        handle.write(f'"{datetime.datetime.now().isoformat()}","{log_type}","{speaker}","{escaped}"\n')


def init_record():
    global DB, SESSION_ID, CURRENT_TURN_INDEX, USER_CONTEXT
    with OUTPUT_QUEUE.mutex:
        OUTPUT_QUEUE.queue.clear()
    with INPUT_QUEUE.mutex:
        INPUT_QUEUE.queue.clear()

    DB = DBManager(DB_PATH)
    user_id = DB.get_user_id(SUBJECT_ID)
    SESSION_ID = DB.create_session(user_id)
    USER_CONTEXT = DB.get_user_context_string(user_id)
    CURRENT_TURN_INDEX = 0
    _append_csv("internal", "system", f"Session initialized for {SUBJECT_ID} (session={SESSION_ID})")
    logger.info("Initialized new session %s for subject %s", SESSION_ID, SUBJECT_ID)


def log_question(text: str, meta_data=None):
    global CURRENT_TURN_INDEX, _PENDING_QUESTION_PREFIX
    combined = text
    if _PENDING_QUESTION_PREFIX:
        combined = f"{_PENDING_QUESTION_PREFIX}\n\n{text}"
        _PENDING_QUESTION_PREFIX = ""

    OUTPUT_QUEUE.put(combined)
    if DB and SESSION_ID:
        DB.add_turn(SESSION_ID, CURRENT_TURN_INDEX, "agent", combined, meta_data=meta_data)
        CURRENT_TURN_INDEX += 1
    _append_csv("turn", "agent", combined)
    logger.info("Prompted question: %s", combined)


def _normalize_user_segments(user_input_text: str, emotion: str = "neutral"):
    user_input_text = user_input_text.replace(", and", ".").replace("but", ".")
    segments = []
    parts = [part.strip() for part in user_input_text.split(".") if part.strip()]
    for index, segment in enumerate(parts):
        if index == len(parts) - 1:
            segments.append(f"{segment} [Detected Emotion: {emotion}]")
        else:
            segments.append(segment)
    return segments


def _pull_user_message():
    global CURRENT_TURN_INDEX
    payload = INPUT_QUEUE.get()
    raw_text = str(payload)
    transcript = raw_text
    emotion = "neutral"
    try:
        parsed = json.loads(raw_text)
        transcript = str(parsed.get("transcript", "")).strip()
        emotion = str(parsed.get("detected_emotion", "neutral")).strip()
    except Exception:
        transcript = raw_text.strip()

    if DB and SESSION_ID:
        DB.add_turn(SESSION_ID, CURRENT_TURN_INDEX, "user", transcript)
        CURRENT_TURN_INDEX += 1
    _append_csv("turn", "user", transcript)
    return transcript, emotion


def get_answer():
    transcript, emotion = _pull_user_message()
    logger.info("Received user input: %s", transcript)
    return [], _normalize_user_segments(transcript, emotion)


def get_resp_log():
    transcript, emotion = _pull_user_message()
    full_text = f"{transcript} [Detected Emotion: {emotion}]"
    logger.info("Received user response: %s", full_text)
    return full_text
