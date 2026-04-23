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
LLM = _CFG.get("llm", {})
AUDIO = _CFG.get("audio", {})
STT = _CFG.get("stt", {})
TTS = _CFG.get("tts", {})
DATABASE = _CFG.get("database", {})

SUBJECT_ID = str(APP["subject_id"])

def _expand(path: str) -> str:
    expanded = path.replace("${subject_id}", SUBJECT_ID)
    return os.path.normpath(expanded)

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

LLM_MODEL = os.environ.get("LLM_MODEL", LLM.get("model", "gemma-4-E2B-it"))
LITERT_MODEL_PATH = os.path.normpath(
    os.environ.get("LITERT_MODEL_PATH", LLM.get("litert_model_path", "./models/litert/gemma-4-E2B-it.litertlm"))
)
LITERT_BACKEND = os.environ.get("LITERT_BACKEND", LLM.get("backend", "cpu")).strip().lower()
LITERT_CONTEXT_LENGTH = int(os.environ.get("LITERT_CONTEXT_LENGTH", str(LLM.get("context_length", 1024))))
LITERT_MAX_TOKENS = int(os.environ.get("LITERT_MAX_TOKENS", str(LLM.get("max_tokens", 256))))

AUDIO_SAMPLE_RATE = int(AUDIO.get("sample_rate", 16000))
AUDIO_CHANNELS = int(AUDIO.get("channels", 1))
AUDIO_CHUNK_SIZE = int(AUDIO.get("chunk_size", 480))
AUDIO_VAD_AGGRESSIVENESS = int(AUDIO.get("vad_aggressiveness", 2))
AUDIO_SILENCE_TIMEOUT_SEC = float(AUDIO.get("silence_timeout_sec", 1.1))
AUDIO_MAX_RECORD_SEC = float(AUDIO.get("max_record_sec", 20))

STT_MODEL_PATH = os.environ.get("STT_MODEL", STT.get("model_path", "base.en"))
STT_DEVICE = os.environ.get("STT_DEVICE", STT.get("device", "cpu"))
STT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", STT.get("compute_type", "int8"))
STT_BEAM_SIZE = int(os.environ.get("STT_BEAM_SIZE", str(STT.get("beam_size", 2))))
STT_BEST_OF = int(os.environ.get("STT_BEST_OF", str(STT.get("best_of", 1))))
STT_WITHOUT_TIMESTAMPS = str(
    os.environ.get("STT_WITHOUT_TIMESTAMPS", STT.get("without_timestamps", True))
).strip().lower() in {"1", "true", "yes", "on"}

TTS_MODEL_PATH = os.path.normpath(
    os.environ.get("TTS_MODEL_PATH", TTS.get("model_path", "./models/piper/en_US-amy-medium.onnx"))
)
TTS_EXECUTABLE = os.environ.get("TTS_EXECUTABLE", TTS.get("executable_path", "piper"))
TTS_LENGTH_SCALE = float(os.environ.get("TTS_LENGTH_SCALE", str(TTS.get("length_scale", 0.9))))
TTS_SENTENCE_SILENCE = float(os.environ.get("TTS_SENTENCE_SILENCE", str(TTS.get("sentence_silence", 0.7))))

DB_PATH = os.path.normpath(os.environ.get("DB_PATH", DATABASE.get("db_path", "data/therapist.db")))


