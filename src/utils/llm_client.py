import os
import threading

from src.utils.config_loader import (
    LITERT_BACKEND,
    LITERT_CONTEXT_LENGTH,
    LITERT_MAX_TOKENS,
    LITERT_MODEL_PATH,
    LLM_MODEL,
)
from src.utils.log_util import get_logger

logger = get_logger("LLMClient")

_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def _init_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE

        if not os.path.exists(LITERT_MODEL_PATH):
            raise FileNotFoundError(
                f"LiteRT model not found at {LITERT_MODEL_PATH}. "
                "Point LITERT_MODEL_PATH to the local Gemma 4 file from conversational_ai_therapist_smart_speaker."
            )

        try:
            import litert_lm
        except ImportError as exc:
            raise RuntimeError(
                "litert_lm is not installed. Install the LiteRT runtime before starting CaiTI."
            ) from exc

        backend = getattr(litert_lm.Backend, "CPU")
        if LITERT_BACKEND == "gpu" and hasattr(litert_lm.Backend, "GPU"):
            backend = litert_lm.Backend.GPU

        logger.info("Loading LiteRT model from %s", LITERT_MODEL_PATH)
        _ENGINE = litert_lm.Engine(
            LITERT_MODEL_PATH,
            backend=backend,
            max_context_length=LITERT_CONTEXT_LENGTH,
        )
        return _ENGINE


def llm_complete(system_content: str, user_content: str) -> str:
    logger.info("Sending request to local LLM model %s", LLM_MODEL)
    prompt = f"{system_content}\n\n{user_content}"
    engine = _init_engine()
    with engine.create_conversation() as conversation:
        response = conversation.send_message(prompt, max_output_tokens=LITERT_MAX_TOKENS)

    if isinstance(response, dict):
        content_parts = response.get("content", [])
        if content_parts and isinstance(content_parts[0], dict):
            return str(content_parts[0].get("text", "")).strip()
        return str(response).strip()
    return str(response).strip()


__all__ = ["llm_complete"]
