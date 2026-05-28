import threading
import json
import time
from urllib import error as urlerror
from urllib import request as urlrequest

from src.local_llm.runtime import LocalCaiTIRuntime, RuntimeSettings
from src.local_llm.types import GenerationConfig, GenerationResult, LLMTask
from src.utils.config_loader import (
    LOCAL_LLM_BASE_SUBDIR,
    LOCAL_LLM_DEFAULT_MAX_NEW_TOKENS,
    LOCAL_LLM_DEVICE_MAP,
    LOCAL_LLM_MAX_INPUT_TOKENS,
    LOCAL_LLM_MODEL_ID,
    LOCAL_LLM_SERVER_TIMEOUT_SEC,
    LOCAL_LLM_SERVER_URL,
    LOCAL_LLM_TEMPERATURE,
    LOCAL_LLM_TOKENIZER_ID,
    LOCAL_LLM_TOKENIZER_SUBDIR,
    LOCAL_LLM_TOP_P,
    LOCAL_LLM_TORCH_DTYPE,
)
from src.utils.log_util import get_logger
from src.utils.session_event_logger import log_llm_event

logger = get_logger("LLMClient")

_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()
_DECISION_STOP_REGEX = r"(?:DECISION\s*[:=]\s*)?[01]\b"
_DECISION_TASKS = {
    LLMTask.TASK3_RV_REASONER,
    LLMTask.TASK4_CBT_STAGE1,
    LLMTask.TASK4_CBT_STAGE2,
    LLMTask.TASK4_CBT_STAGE3,
}


def _runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        model_id=LOCAL_LLM_MODEL_ID,
        base_subdir=LOCAL_LLM_BASE_SUBDIR,
        tokenizer_id=LOCAL_LLM_TOKENIZER_ID,
        tokenizer_subdir=LOCAL_LLM_TOKENIZER_SUBDIR,
        device_map=LOCAL_LLM_DEVICE_MAP,
        torch_dtype=LOCAL_LLM_TORCH_DTYPE,
    )


def _get_runtime() -> LocalCaiTIRuntime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            logger.info("Loading local CaiTI runtime from %s", LOCAL_LLM_MODEL_ID)
            _RUNTIME = LocalCaiTIRuntime(_runtime_settings())
    return _RUNTIME


def _server_url() -> str:
    return LOCAL_LLM_SERVER_URL.rstrip("/")


def _using_remote_server() -> bool:
    return bool(_server_url())


def _post_remote(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{_server_url()}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=LOCAL_LLM_SERVER_TIMEOUT_SEC) as response:
            return json.loads(response.read().decode("utf-8"))
    except urlerror.URLError as exc:
        raise RuntimeError(f"CaiTI LLM server request failed: {exc}") from exc


def _get_remote(path: str) -> dict:
    try:
        with urlrequest.urlopen(f"{_server_url()}{path}", timeout=LOCAL_LLM_SERVER_TIMEOUT_SEC) as response:
            return json.loads(response.read().decode("utf-8"))
    except urlerror.URLError as exc:
        raise RuntimeError(f"CaiTI LLM server is not available at {_server_url()}: {exc}") from exc


def _remote_generate(
    task: LLMTask,
    system_content: str,
    user_content: str,
    max_new_tokens: int | None = None,
    stop_regex: str | None = None,
) -> GenerationResult:
    payload = {
        "task": task.value,
        "system_content": system_content,
        "user_content": user_content,
    }
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    if stop_regex:
        payload["stop_regex"] = stop_regex
    data = _post_remote("/generate", payload)
    result_task = LLMTask(data.get("task", task.value))
    return GenerationResult(
        text=str(data.get("text", "")),
        task=result_task,
        adapter=data.get("adapter"),
        raw_text=str(data.get("raw_text", data.get("text", ""))),
    )


def preload_llm_runtime():
    """Load the local CaiTI runtime before the first user-facing turn."""

    if _using_remote_server():
        logger.info("Using persistent CaiTI LLM server at %s", _server_url())
        return _get_remote("/health")
    return _get_runtime()


def llm_complete(system_content: str, user_content: str) -> str:
    """
    Unified LLM caller used across the app.
    Inputs:
      - system_content: system prompt/instructions
      - user_content: user input/payload
    Output:
      - plain text content returned by the model
    """
    logger.info("Sending base request to local CaiTI LLM")
    if _using_remote_server():
        result = _remote_generate(LLMTask.BASE, system_content, user_content)
    else:
        result = _get_runtime().generate_base(
            system_content,
            user_content,
            GenerationConfig(
                max_new_tokens=LOCAL_LLM_DEFAULT_MAX_NEW_TOKENS,
                temperature=LOCAL_LLM_TEMPERATURE,
                top_p=LOCAL_LLM_TOP_P,
                do_sample=True,
                use_chat_template=True,
                max_input_tokens=LOCAL_LLM_MAX_INPUT_TOKENS,
            ),
        )
    log_llm_event(
        task=LLMTask.BASE,
        adapter="base_model",
        segment_text=user_content,
        raw_llm_output=result.text,
        normalized_output=result.text,
        metadata={"system_prompt_prefix": system_content[:160]},
    )
    logger.debug({"task": result.task.value, "adapter": result.adapter, "raw": result.raw_text})
    return result.text


def llm_complete_task(
    task: LLMTask,
    system_content: str,
    user_content: str,
    max_new_tokens: int | None = None,
) -> GenerationResult:
    """Run an explicit CaiTI task through its local adapter."""

    if task == LLMTask.BASE:
        text = llm_complete(system_content, user_content)
        return GenerationResult(text=text, task=task, adapter=None, raw_text=text)

    logger.info("Sending adapter request to local CaiTI LLM: %s", task.value)
    start_time = time.perf_counter()
    stop_regex = _DECISION_STOP_REGEX if task in _DECISION_TASKS else None
    if _using_remote_server():
        result = _remote_generate(
            task,
            system_content,
            user_content,
            max_new_tokens=max_new_tokens,
            stop_regex=stop_regex,
        )
        logger.info("Completed adapter request %s in %.2fs.", task.value, time.perf_counter() - start_time)
        logger.debug({"task": task.value, "adapter": result.adapter, "raw": result.raw_text})
        return result

    prompt = f"{system_content.rstrip()}\n{user_content.lstrip()}"
    result = _get_runtime().generate_adapter(
        task,
        prompt,
        GenerationConfig(
            max_new_tokens=max_new_tokens or 32,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            use_chat_template=False,
            max_input_tokens=LOCAL_LLM_MAX_INPUT_TOKENS,
            stop_regex=stop_regex,
        ),
    )
    logger.info("Completed adapter request %s in %.2fs.", task.value, time.perf_counter() - start_time)
    logger.debug({"task": task.value, "adapter": result.adapter, "raw": result.raw_text})
    return result


__all__ = ["llm_complete", "llm_complete_task", "preload_llm_runtime"]
