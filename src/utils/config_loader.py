import os
from typing import Any, Dict

import yaml

_ROOT_DIR = os.path.abspath(".")
_CONFIG_PATH = os.path.join(_ROOT_DIR, "config.yaml")

def _load_yaml_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("config.yaml must contain a top-level mapping")
        return data

_CFG = _load_yaml_config()

APP = _CFG["app"]
PATHS = _CFG["paths"]
RL = _CFG["rl"]
LOCAL_LLM = _CFG.get("local_llm", {})
VOICE = _CFG.get("voice", {})

SUBJECT_ID = str(APP["subject_id"])

def _expand(path: str) -> str:
    return path.replace("${subject_id}", SUBJECT_ID)

DATA_DIR = _expand(PATHS["data_dir"])
LOG_DIR = _expand(PATHS["logs_dir"])
RESULT_DIR = _expand(PATHS["result_dir"])
QUESTION_LIB_FILENAME = _expand(PATHS["question_lib_filename"])
REPORT_FILE = _expand(PATHS["report_file"])
NOTES_FILE = _expand(PATHS["notes_file"])
RECORD_CSV = _expand(PATHS["record_csv"])

ITEM_N_STATES = int(RL["item_n_states"])
EPSILON = float(RL["epsilon"])
ALPHA = float(RL["alpha"])
GAMMA = float(RL["gamma"])
ITEM_IMPORTANCE = RL["item_importance"]
NUMBER_QUESTIONS = RL["number_questions"]

LOCAL_LLM_MODEL_ID = os.environ.get(
    "CAITI_MODEL_ID",
    LOCAL_LLM.get("model_id", "xxue752/caiti_best_model"),
)
LOCAL_LLM_BASE_SUBDIR = os.environ.get(
    "CAITI_BASE_SUBDIR",
    LOCAL_LLM.get("base_subdir", "compressed_model_int4"),
)
LOCAL_LLM_TOKENIZER_ID = os.environ.get(
    "CAITI_TOKENIZER_ID",
    LOCAL_LLM.get("tokenizer_id", LOCAL_LLM_MODEL_ID),
)
LOCAL_LLM_TOKENIZER_SUBDIR = os.environ.get(
    "CAITI_TOKENIZER_SUBDIR",
    LOCAL_LLM.get("tokenizer_subdir", LOCAL_LLM_BASE_SUBDIR),
)
LOCAL_LLM_DEVICE_MAP = os.environ.get(
    "CAITI_DEVICE_MAP",
    str(LOCAL_LLM.get("device_map", "auto")),
)
LOCAL_LLM_TORCH_DTYPE = os.environ.get(
    "CAITI_TORCH_DTYPE",
    str(LOCAL_LLM.get("torch_dtype", "auto")),
)
LOCAL_LLM_MAX_INPUT_TOKENS = int(
    os.environ.get(
        "CAITI_MAX_INPUT_TOKENS",
        str(LOCAL_LLM.get("max_input_tokens", 2048)),
    )
)
LOCAL_LLM_DEFAULT_MAX_NEW_TOKENS = int(
    os.environ.get(
        "CAITI_DEFAULT_MAX_NEW_TOKENS",
        str(LOCAL_LLM.get("default_max_new_tokens", 128)),
    )
)
LOCAL_LLM_TEMPERATURE = float(
    os.environ.get("CAITI_TEMPERATURE", str(LOCAL_LLM.get("temperature", 0.7)))
)
LOCAL_LLM_TOP_P = float(
    os.environ.get("CAITI_TOP_P", str(LOCAL_LLM.get("top_p", 0.95)))
)

VOICE_STT_BACKEND = os.environ.get("CAITI_STT_BACKEND", str(VOICE.get("stt_backend", "console")))
VOICE_TTS_BACKEND = os.environ.get("CAITI_TTS_BACKEND", str(VOICE.get("tts_backend", "console")))
VOICE_STT_COMMAND = os.environ.get("CAITI_STT_COMMAND", str(VOICE.get("stt_command", "")))
VOICE_TTS_COMMAND = os.environ.get("CAITI_TTS_COMMAND", str(VOICE.get("tts_command", "")))
VOICE_STT_TIMEOUT_SEC = int(
    os.environ.get("CAITI_STT_TIMEOUT_SEC", str(VOICE.get("stt_timeout_sec", 120)))
)
VOICE_TTS_TIMEOUT_SEC = int(
    os.environ.get("CAITI_TTS_TIMEOUT_SEC", str(VOICE.get("tts_timeout_sec", 60)))
)
VOICE_EMPTY_TRANSCRIPT_RETRIES = int(
    os.environ.get("CAITI_EMPTY_TRANSCRIPT_RETRIES", str(VOICE.get("empty_transcript_retries", 2)))
)


