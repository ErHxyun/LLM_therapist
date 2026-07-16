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
HARDWARE = _CFG.get("hardware", {})
MONITOR = _CFG.get("monitor", {})
INTERMISSION = _CFG.get("intermission", {})
EMOTION = _CFG.get("emotion", {})

SUBJECT_ID = str(APP["subject_id"])

def _expand(path: str) -> str:
    return path.replace("${subject_id}", SUBJECT_ID)


def _bool_env(name: str, default: Any) -> bool:
    value = os.environ.get(name, str(default))
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

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
LOCAL_LLM_SERVER_URL = os.environ.get(
    "CAITI_LLM_SERVER_URL",
    str(LOCAL_LLM.get("server_url", "")),
).strip()
LOCAL_LLM_SERVER_HOST = os.environ.get(
    "CAITI_LLM_SERVER_HOST",
    str(LOCAL_LLM.get("server_host", "127.0.0.1")),
)
LOCAL_LLM_SERVER_PORT = int(
    os.environ.get("CAITI_LLM_SERVER_PORT", str(LOCAL_LLM.get("server_port", 8890)))
)
LOCAL_LLM_SERVER_TIMEOUT_SEC = float(
    os.environ.get("CAITI_LLM_SERVER_TIMEOUT_SEC", str(LOCAL_LLM.get("server_timeout_sec", 180)))
)

VOICE_STT_BACKEND = os.environ.get("CAITI_STT_BACKEND", str(VOICE.get("stt_backend", "console")))
VOICE_TTS_BACKEND = os.environ.get("CAITI_TTS_BACKEND", str(VOICE.get("tts_backend", "console")))
VOICE_STT_COMMAND = os.environ.get("CAITI_STT_COMMAND", str(VOICE.get("stt_command", "")))
VOICE_TTS_COMMAND = os.environ.get("CAITI_TTS_COMMAND", str(VOICE.get("tts_command", "")))
VOICE_STT_WHISPER_MODEL = os.environ.get("CAITI_WHISPER_MODEL", str(VOICE.get("stt_whisper_model", "base.en")))
VOICE_STT_RECORD_SECONDS = float(
    os.environ.get("CAITI_STT_RECORD_SECONDS", str(VOICE.get("stt_record_seconds", 30)))
)
VOICE_STT_SAMPLE_RATE = int(
    os.environ.get("CAITI_STT_SAMPLE_RATE", str(VOICE.get("stt_sample_rate", 16000)))
)
VOICE_STT_CHANNELS = int(os.environ.get("CAITI_STT_CHANNELS", str(VOICE.get("stt_channels", 1))))
VOICE_STT_AUDIO_DEVICE = os.environ.get("CAITI_STT_AUDIO_DEVICE", str(VOICE.get("stt_audio_device", "")))
VOICE_STT_DEVICE = os.environ.get("CAITI_WHISPER_DEVICE", str(VOICE.get("stt_device", "cpu")))
VOICE_STT_COMPUTE_TYPE = os.environ.get(
    "CAITI_WHISPER_COMPUTE_TYPE",
    str(VOICE.get("stt_compute_type", "int8")),
)
VOICE_STT_BEAM_SIZE = int(os.environ.get("CAITI_WHISPER_BEAM_SIZE", str(VOICE.get("stt_beam_size", 5))))
VOICE_STT_BEST_OF = int(os.environ.get("CAITI_WHISPER_BEST_OF", str(VOICE.get("stt_best_of", 5))))
VOICE_STT_LANGUAGE = os.environ.get("CAITI_WHISPER_LANGUAGE", str(VOICE.get("stt_language", "en")))
VOICE_STT_INITIAL_PROMPT = os.environ.get(
    "CAITI_WHISPER_INITIAL_PROMPT",
    str(
        VOICE.get(
            "stt_initial_prompt",
            "The speaker is answering brief daily functioning screening questions in English.",
        )
    ),
)
VOICE_STT_VAD_FILTER = _bool_env("CAITI_WHISPER_VAD_FILTER", VOICE.get("stt_vad_filter", True))
VOICE_STT_AUTO_STOP = _bool_env("CAITI_STT_AUTO_STOP", VOICE.get("stt_auto_stop", True))
VOICE_STT_VAD_DETECTOR = os.environ.get("CAITI_STT_VAD_DETECTOR", str(VOICE.get("stt_vad_detector", "auto")))
VOICE_STT_VAD_AGGRESSIVENESS = int(
    os.environ.get("CAITI_STT_VAD_AGGRESSIVENESS", str(VOICE.get("stt_vad_aggressiveness", 3)))
)
VOICE_STT_SILENCE_THRESHOLD_DBFS = float(
    os.environ.get("CAITI_STT_SILENCE_THRESHOLD_DBFS", str(VOICE.get("stt_silence_threshold_dbfs", -45)))
)
VOICE_STT_SILENCE_TIMEOUT_SEC = float(
    os.environ.get("CAITI_STT_SILENCE_TIMEOUT_SEC", str(VOICE.get("stt_silence_timeout_sec", 1.2)))
)
VOICE_STT_TRAILING_PAD_SEC = float(
    os.environ.get("CAITI_STT_TRAILING_PAD_SEC", str(VOICE.get("stt_trailing_pad_sec", 0.4)))
)
VOICE_STT_LONG_RESPONSE_SILENCE_TIMEOUT_SEC = float(
    os.environ.get(
        "CAITI_STT_LONG_RESPONSE_SILENCE_TIMEOUT_SEC",
        str(VOICE.get("stt_long_response_silence_timeout_sec", 4.5)),
    )
)
VOICE_STT_MIN_SPEECH_SECONDS = float(
    os.environ.get("CAITI_STT_MIN_SPEECH_SECONDS", str(VOICE.get("stt_min_speech_seconds", 0.25)))
)
VOICE_STT_MIN_RECORD_SECONDS = float(
    os.environ.get("CAITI_STT_MIN_RECORD_SECONDS", str(VOICE.get("stt_min_record_seconds", 1.0)))
)
VOICE_STT_NO_SPEECH_TIMEOUT_SEC = float(
    os.environ.get("CAITI_STT_NO_SPEECH_TIMEOUT_SEC", str(VOICE.get("stt_no_speech_timeout_sec", 5.0)))
)
VOICE_STT_VAD_CHUNK_MS = int(os.environ.get("CAITI_STT_VAD_CHUNK_MS", str(VOICE.get("stt_vad_chunk_ms", 30))))
VOICE_STT_DEBUG_AUDIO = _bool_env("CAITI_STT_DEBUG_AUDIO", VOICE.get("stt_debug_audio", False))
VOICE_STT_SAVE_WAV = os.environ.get("CAITI_STT_SAVE_WAV", str(VOICE.get("stt_save_wav", "")))
VOICE_STT_TIMEOUT_SEC = int(
    os.environ.get("CAITI_STT_TIMEOUT_SEC", str(VOICE.get("stt_timeout_sec", 120)))
)
VOICE_TTS_TIMEOUT_SEC = int(
    os.environ.get("CAITI_TTS_TIMEOUT_SEC", str(VOICE.get("tts_timeout_sec", 60)))
)
VOICE_EMPTY_TRANSCRIPT_RETRIES = int(
    os.environ.get("CAITI_EMPTY_TRANSCRIPT_RETRIES", str(VOICE.get("empty_transcript_retries", 2)))
)
VOICE_MUSIC_BACKEND = os.environ.get("CAITI_MUSIC_BACKEND", str(VOICE.get("music_backend", "off")))
VOICE_MUSIC_PATH = os.environ.get("CAITI_MUSIC_PATH", str(VOICE.get("music_path", "")))
VOICE_MUSIC_FIREPLACE_PATH = os.environ.get(
    "CAITI_MUSIC_FIREPLACE_PATH",
    str(VOICE.get("music_fireplace_path", "")),
)
VOICE_MUSIC_SEAWAVES_PATH = os.environ.get(
    "CAITI_MUSIC_SEAWAVES_PATH",
    str(VOICE.get("music_seawaves_path", VOICE.get("music_forest_path", ""))),
)
VOICE_MUSIC_FOREST_PATH = VOICE_MUSIC_SEAWAVES_PATH
VOICE_MUSIC_COMMAND = os.environ.get("CAITI_MUSIC_COMMAND", str(VOICE.get("music_command", "aplay -q {path}")))
VOICE_MUSIC_VOLUME_PERCENT = int(
    os.environ.get("CAITI_MUSIC_VOLUME_PERCENT", str(VOICE.get("music_volume_percent", 30)))
)
VOICE_MUSIC_DUCK_VOLUME_PERCENT = int(
    os.environ.get("CAITI_MUSIC_DUCK_VOLUME_PERCENT", str(VOICE.get("music_duck_volume_percent", 8)))
)
VOICE_MUSIC_IPC_PATH = os.environ.get(
    "CAITI_MUSIC_IPC_PATH",
    str(VOICE.get("music_ipc_path", "/tmp/caiti_mpv_music.sock")),
)

INTERMISSION_ENABLED = _bool_env("CAITI_INTERMISSION_ENABLED", INTERMISSION.get("enabled", False))
INTERMISSION_TTS_BACKEND = os.environ.get(
    "CAITI_INTERMISSION_TTS_BACKEND",
    str(INTERMISSION.get("tts_backend", "command")),
)
INTERMISSION_TTS_COMMAND = os.environ.get(
    "CAITI_INTERMISSION_TTS_COMMAND",
    str(INTERMISSION.get("tts_command", "")),
)
INTERMISSION_FALLBACK_TO_PRIMARY_TTS = _bool_env(
    "CAITI_INTERMISSION_FALLBACK_TO_PRIMARY_TTS",
    INTERMISSION.get("fallback_to_primary_tts", True),
)
INTERMISSION_SCREENING_ENABLED = _bool_env(
    "CAITI_INTERMISSION_SCREENING_ENABLED",
    INTERMISSION.get("screening_enabled", True),
)
INTERMISSION_BREATHING_ENABLED = _bool_env(
    "CAITI_INTERMISSION_BREATHING_ENABLED",
    INTERMISSION.get("breathing_enabled", True),
)
INTERMISSION_MINDFULNESS_ENABLED = _bool_env(
    "CAITI_INTERMISSION_MINDFULNESS_ENABLED",
    INTERMISSION.get("mindfulness_enabled", True),
)
INTERMISSION_MAX_SECONDS = float(
    os.environ.get("CAITI_INTERMISSION_MAX_SECONDS", str(INTERMISSION.get("max_seconds", 45.0)))
)
INTERMISSION_POLL_INTERVAL_SEC = float(
    os.environ.get(
        "CAITI_INTERMISSION_POLL_INTERVAL_SEC",
        str(INTERMISSION.get("poll_interval_sec", 0.1)),
    )
)
INTERMISSION_MAX_SCREENING_ITEMS_PER_TURN = int(
    os.environ.get(
        "CAITI_INTERMISSION_MAX_SCREENING_ITEMS_PER_TURN",
        str(INTERMISSION.get("max_screening_items_per_turn", 1)),
    )
)
INTERMISSION_TRIGGER_MIN_USER_SPEECH_SEC = float(
    os.environ.get(
        "CAITI_INTERMISSION_TRIGGER_MIN_USER_SPEECH_SEC",
        str(INTERMISSION.get("trigger_min_user_speech_sec", 10.0)),
    )
)
INTERMISSION_TRIGGER_MIN_INTERVAL_TURNS = int(
    os.environ.get(
        "CAITI_INTERMISSION_TRIGGER_MIN_INTERVAL_TURNS",
        str(INTERMISSION.get("trigger_min_interval_turns", 2)),
    )
)
INTERMISSION_TRIGGER_PROBABILITY = float(
    os.environ.get(
        "CAITI_INTERMISSION_TRIGGER_PROBABILITY",
        str(INTERMISSION.get("trigger_probability", 0.5)),
    )
)
INTERMISSION_COOLDOWN_TURNS = int(
    os.environ.get(
        "CAITI_INTERMISSION_COOLDOWN_TURNS",
        str(INTERMISSION.get("cooldown_turns", 1)),
    )
)
INTERMISSION_PERSIST_RESULTS = _bool_env(
    "CAITI_INTERMISSION_PERSIST_RESULTS",
    INTERMISSION.get("persist_results", True),
)
INTERMISSION_DB_PATH = os.environ.get(
    "CAITI_INTERMISSION_DB_PATH",
    str(INTERMISSION.get("db_path", "")),
).strip()
INTERMISSION_RESULTS_JSON_PATH = os.environ.get(
    "CAITI_INTERMISSION_RESULTS_JSON_PATH",
    str(INTERMISSION.get("results_json_path", "")),
).strip()
INTERMISSION_LEAD_IN_TEXT = os.environ.get(
    "CAITI_INTERMISSION_LEAD_IN_TEXT",
    str(
        INTERMISSION.get(
            "lead_in_text",
            "Let's do a brief check-in together.",
        )
    ),
)
INTERMISSION_BRIDGE_TEXT = os.environ.get(
    "CAITI_INTERMISSION_BRIDGE_TEXT",
    str(INTERMISSION.get("bridge_text", "Let's go back to the main session.")),
)
INTERMISSION_TRANSITION_DELAY_SEC = float(
    os.environ.get(
        "CAITI_INTERMISSION_TRANSITION_DELAY_SEC",
        str(INTERMISSION.get("transition_delay_sec", 2.0)),
    )
)

EMOTION_ENABLED = _bool_env("CAITI_EMOTION_ENABLED", EMOTION.get("enabled", False))
EMOTION_SERVICE_URL = os.environ.get(
    "CAITI_EMOTION_SERVICE_URL",
    str(EMOTION.get("service_url", "")),
).strip()
EMOTION_USER_ID = _expand(
    os.environ.get("CAITI_EMOTION_USER_ID", str(EMOTION.get("user_id", SUBJECT_ID)))
)
EMOTION_LANGUAGE = os.environ.get("CAITI_EMOTION_LANGUAGE", str(EMOTION.get("language", "en")))
EMOTION_TIMEOUT_SEC = float(
    os.environ.get("CAITI_EMOTION_TIMEOUT_SEC", str(EMOTION.get("timeout_sec", 10.0)))
)
EMOTION_RESULTS_JSONL_PATH = _expand(
    os.environ.get(
        "CAITI_EMOTION_RESULTS_JSONL_PATH",
        str(EMOTION.get("results_jsonl_path", os.path.join(DATA_DIR, "emotion", "results.jsonl"))),
    )
)
EMOTION_AUDIO_DIR = _expand(
    os.environ.get(
        "CAITI_EMOTION_AUDIO_DIR",
        str(EMOTION.get("audio_dir", os.path.join(DATA_DIR, "emotion", "audio"))),
    )
)
EMOTION_KEEP_AUDIO = _bool_env("CAITI_EMOTION_KEEP_AUDIO", EMOTION.get("keep_audio", False))
EMOTION_ASSIST_FOLLOWUP_ENABLED = _bool_env(
    "CAITI_EMOTION_ASSIST_FOLLOWUP_ENABLED",
    EMOTION.get("assist_followup_enabled", False),
)
EMOTION_ASSIST_WAIT_TIMEOUT_SEC = float(
    os.environ.get(
        "CAITI_EMOTION_ASSIST_WAIT_TIMEOUT_SEC",
        str(EMOTION.get("assist_wait_timeout_sec", 0.0)),
    )
)
EMOTION_ASSIST_LATE_FOLLOWUP_WINDOW_SEC = float(
    os.environ.get(
        "CAITI_EMOTION_ASSIST_LATE_FOLLOWUP_WINDOW_SEC",
        str(EMOTION.get("assist_late_followup_window_sec", 0.0)),
    )
)
EMOTION_ASSIST_MIN_CONFIDENCE = int(
    os.environ.get(
        "CAITI_EMOTION_ASSIST_MIN_CONFIDENCE",
        str(EMOTION.get("assist_min_confidence", 50)),
    )
)
EMOTION_ASSIST_RISK_THRESHOLD = int(
    os.environ.get(
        "CAITI_EMOTION_ASSIST_RISK_THRESHOLD",
        str(EMOTION.get("assist_risk_threshold", 60)),
    )
)
EMOTION_ASSIST_LIGHT_RISK_THRESHOLD = int(
    os.environ.get(
        "CAITI_EMOTION_ASSIST_LIGHT_RISK_THRESHOLD",
        str(EMOTION.get("assist_light_risk_threshold", 45)),
    )
)

HARDWARE_STATUS_LEDS_ENABLED = _bool_env(
    "CAITI_STATUS_LEDS_ENABLED",
    HARDWARE.get("status_leds_enabled", False),
)
HARDWARE_STATUS_LED_WHITE_BOARD_PIN = int(
    os.environ.get("CAITI_STATUS_LED_WHITE_BOARD_PIN", str(HARDWARE.get("status_led_white_board_pin", 15)))
)
HARDWARE_STATUS_LED_YELLOW_BOARD_PIN = int(
    os.environ.get("CAITI_STATUS_LED_YELLOW_BOARD_PIN", str(HARDWARE.get("status_led_yellow_board_pin", 16)))
)
_HARDWARE_STATUS_LED_BLUE_DEFAULT = str(
    HARDWARE.get("status_led_blue_board_pin", HARDWARE.get("status_led_red_board_pin", 18))
)
HARDWARE_STATUS_LED_BLUE_BOARD_PIN = int(
    os.environ.get(
        "CAITI_STATUS_LED_BLUE_BOARD_PIN",
        os.environ.get("CAITI_STATUS_LED_RED_BOARD_PIN", _HARDWARE_STATUS_LED_BLUE_DEFAULT),
    )
)
HARDWARE_STATUS_LED_RED_BOARD_PIN = HARDWARE_STATUS_LED_BLUE_BOARD_PIN
HARDWARE_STATUS_LED_GREEN_BOARD_PIN = int(
    os.environ.get("CAITI_STATUS_LED_GREEN_BOARD_PIN", str(HARDWARE.get("status_led_green_board_pin", 22)))
)
HARDWARE_STATUS_LED_ACTIVE_LOW = _bool_env(
    "CAITI_STATUS_LED_ACTIVE_LOW",
    HARDWARE.get("status_led_active_low", False),
)

HARDWARE_VOLUME_BUTTONS_ENABLED = _bool_env(
    "CAITI_VOLUME_BUTTONS_ENABLED",
    HARDWARE.get("volume_buttons_enabled", False),
)
HARDWARE_VOLUME_UP_BOARD_PIN = int(
    os.environ.get("CAITI_VOLUME_UP_BOARD_PIN", str(HARDWARE.get("volume_up_board_pin", 32)))
)
HARDWARE_VOLUME_DOWN_BOARD_PIN = int(
    os.environ.get("CAITI_VOLUME_DOWN_BOARD_PIN", str(HARDWARE.get("volume_down_board_pin", 33)))
)
HARDWARE_VOLUME_STEP_PERCENT = int(
    os.environ.get("CAITI_VOLUME_STEP_PERCENT", str(HARDWARE.get("volume_step_percent", 5)))
)
HARDWARE_VOLUME_MIN_PERCENT = int(
    os.environ.get("CAITI_VOLUME_MIN_PERCENT", str(HARDWARE.get("volume_min_percent", 0)))
)
HARDWARE_VOLUME_MAX_PERCENT = int(
    os.environ.get("CAITI_VOLUME_MAX_PERCENT", str(HARDWARE.get("volume_max_percent", 100)))
)
HARDWARE_VOLUME_DEBOUNCE_SEC = float(
    os.environ.get("CAITI_VOLUME_DEBOUNCE_SEC", str(HARDWARE.get("volume_debounce_sec", 0.5)))
)
HARDWARE_VOLUME_RELEASE_SEC = float(
    os.environ.get("CAITI_VOLUME_RELEASE_SEC", str(HARDWARE.get("volume_release_sec", 0.2)))
)
HARDWARE_VOLUME_POLL_INTERVAL_SEC = float(
    os.environ.get("CAITI_VOLUME_POLL_INTERVAL_SEC", str(HARDWARE.get("volume_poll_interval_sec", 0.01)))
)
HARDWARE_VOLUME_ACTIVE_LOW = _bool_env(
    "CAITI_VOLUME_ACTIVE_LOW",
    HARDWARE.get("volume_active_low", True),
)

HARDWARE_MUSIC_MODE_BUTTON_ENABLED = _bool_env(
    "CAITI_MUSIC_MODE_BUTTON_ENABLED",
    HARDWARE.get("music_mode_button_enabled", False),
)
HARDWARE_MUSIC_MODE_BUTTON_BOARD_PIN = int(
    os.environ.get("CAITI_MUSIC_MODE_BUTTON_BOARD_PIN", str(HARDWARE.get("music_mode_button_board_pin", 35)))
)
HARDWARE_MUSIC_MODE_BUTTON_DEBOUNCE_SEC = float(
    os.environ.get("CAITI_MUSIC_MODE_BUTTON_DEBOUNCE_SEC", str(HARDWARE.get("music_mode_button_debounce_sec", 0.5)))
)
HARDWARE_MUSIC_MODE_BUTTON_RELEASE_SEC = float(
    os.environ.get("CAITI_MUSIC_MODE_BUTTON_RELEASE_SEC", str(HARDWARE.get("music_mode_button_release_sec", 0.2)))
)
HARDWARE_MUSIC_MODE_BUTTON_POLL_INTERVAL_SEC = float(
    os.environ.get(
        "CAITI_MUSIC_MODE_BUTTON_POLL_INTERVAL_SEC",
        str(HARDWARE.get("music_mode_button_poll_interval_sec", 0.01)),
    )
)
HARDWARE_MUSIC_MODE_BUTTON_ACTIVE_LOW = _bool_env(
    "CAITI_MUSIC_MODE_BUTTON_ACTIVE_LOW",
    HARDWARE.get("music_mode_button_active_low", True),
)

HARDWARE_SESSION_BUTTON_ENABLED = _bool_env(
    "CAITI_SESSION_BUTTON_ENABLED",
    HARDWARE.get("session_button_enabled", False),
)
HARDWARE_SESSION_BUTTON_BOARD_PIN = int(
    os.environ.get("CAITI_SESSION_BUTTON_BOARD_PIN", str(HARDWARE.get("session_button_board_pin", 37)))
)
HARDWARE_SESSION_BUTTON_LONG_PRESS_SEC = float(
    os.environ.get("CAITI_SESSION_BUTTON_LONG_PRESS_SEC", str(HARDWARE.get("session_button_long_press_sec", 3.0)))
)
HARDWARE_SESSION_BUTTON_DEBOUNCE_SEC = float(
    os.environ.get("CAITI_SESSION_BUTTON_DEBOUNCE_SEC", str(HARDWARE.get("session_button_debounce_sec", 0.05)))
)
HARDWARE_SESSION_BUTTON_POLL_INTERVAL_SEC = float(
    os.environ.get("CAITI_SESSION_BUTTON_POLL_INTERVAL_SEC", str(HARDWARE.get("session_button_poll_interval_sec", 0.01)))
)
HARDWARE_SESSION_BUTTON_ACTIVE_LOW = _bool_env(
    "CAITI_SESSION_BUTTON_ACTIVE_LOW",
    HARDWARE.get("session_button_active_low", True),
)

MONITOR_ENABLED = _bool_env("CAITI_MONITOR_ENABLED", MONITOR.get("enabled", False))
MONITOR_HOST = os.environ.get("CAITI_MONITOR_HOST", str(MONITOR.get("host", "127.0.0.1")))
MONITOR_PORT = int(os.environ.get("CAITI_MONITOR_PORT", str(MONITOR.get("port", 8765))))
