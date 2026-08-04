"""Small HTTP server that keeps the CaiTI local LLM runtime warm."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from src.local_llm.runtime import LocalCaiTIRuntime, RuntimeSettings
from src.local_llm.types import GenerationConfig, LLMTask
from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("LocalLLMServer")


@dataclass(frozen=True)
class LLMServerSettings:
    host: str = "127.0.0.1"
    port: int = 8890


def runtime_settings_from_config() -> RuntimeSettings:
    return RuntimeSettings(
        model_id=config_loader.LOCAL_LLM_MODEL_ID,
        base_subdir=config_loader.LOCAL_LLM_BASE_SUBDIR,
        tokenizer_id=config_loader.LOCAL_LLM_TOKENIZER_ID,
        tokenizer_subdir=config_loader.LOCAL_LLM_TOKENIZER_SUBDIR,
        device_map=config_loader.LOCAL_LLM_DEVICE_MAP,
        torch_dtype=config_loader.LOCAL_LLM_TORCH_DTYPE,
    )


class LocalLLMService:
    def __init__(self, runtime: LocalCaiTIRuntime):
        self.runtime = runtime
        self._lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        adapter_status = getattr(self.runtime, "adapter_status", None)
        startup_timings = getattr(self.runtime, "startup_timings", None)
        return {
            "ok": True,
            "readiness": "task1_ready",
            "model_id": self.runtime.settings.model_id,
            "device_map": self.runtime.settings.device_map,
            "adapters": adapter_status() if callable(adapter_status) else {},
            "startup_timings_seconds": (
                startup_timings() if callable(startup_timings) else {}
            ),
        }

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = LLMTask(str(payload.get("task", LLMTask.BASE.value)))
        system_content = str(payload.get("system_content", ""))
        user_content = str(payload.get("user_content", ""))
        max_new_tokens = payload.get("max_new_tokens")
        stop_regex = payload.get("stop_regex")

        with self._lock:
            if task == LLMTask.BASE:
                result = self.runtime.generate_base(
                    system_content,
                    user_content,
                    GenerationConfig(
                        max_new_tokens=config_loader.LOCAL_LLM_DEFAULT_MAX_NEW_TOKENS,
                        temperature=config_loader.LOCAL_LLM_TEMPERATURE,
                        top_p=config_loader.LOCAL_LLM_TOP_P,
                        do_sample=True,
                        use_chat_template=True,
                        max_input_tokens=config_loader.LOCAL_LLM_MAX_INPUT_TOKENS,
                    ),
                )
            else:
                prompt = f"{system_content.rstrip()}\n{user_content.lstrip()}"
                result = self.runtime.generate_adapter(
                    task,
                    prompt,
                    GenerationConfig(
                        max_new_tokens=int(max_new_tokens or 32),
                        temperature=0.0,
                        top_p=1.0,
                        do_sample=False,
                        use_chat_template=False,
                        max_input_tokens=config_loader.LOCAL_LLM_MAX_INPUT_TOKENS,
                        stop_regex=str(stop_regex) if stop_regex else None,
                    ),
                )

        return {
            "text": result.text,
            "task": result.task.value,
            "adapter": result.adapter,
            "raw_text": result.raw_text,
        }


class _LLMRequestHandler(BaseHTTPRequestHandler):
    server_version = "CaiTILocalLLMServer/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(200, self._service().health())
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/generate":
            self.send_error(404, "Not found")
            return
        try:
            payload = self._read_json()
            self._send_json(200, self._service().generate(payload))
        except Exception as exc:
            logger.exception("LLM generation request failed.")
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:
        logger.debug("HTTP %s", format % args)

    def _service(self) -> LocalLLMService:
        return self.server.llm_service

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object.")
        return data

    def _send_json(self, code: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(settings: LLMServerSettings, runtime: LocalCaiTIRuntime | None = None) -> ThreadingHTTPServer:
    service = LocalLLMService(runtime or LocalCaiTIRuntime(runtime_settings_from_config()))
    server = ThreadingHTTPServer((settings.host, settings.port), _LLMRequestHandler)
    server.llm_service = service
    return server


def serve_forever(settings: LLMServerSettings | None = None) -> None:
    settings = settings or LLMServerSettings(
        host=config_loader.LOCAL_LLM_SERVER_HOST,
        port=config_loader.LOCAL_LLM_SERVER_PORT,
    )
    logger.info("Loading CaiTI local LLM runtime for persistent server.")
    server = build_server(settings)
    logger.info("CaiTI local LLM server ready at http://%s:%s", settings.host, settings.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
