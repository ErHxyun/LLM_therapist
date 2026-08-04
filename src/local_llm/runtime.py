"""Local Llama + LoRA runtime for CaiTI task modules."""

from __future__ import annotations

import re
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from src.local_llm.routing import TASK_TO_ADAPTER, resolve_adapter
from src.local_llm.types import GenerationConfig, GenerationResult, LLMTask
from src.utils.log_util import get_logger


logger = get_logger("LocalLLMRuntime")


@dataclass(frozen=True)
class RuntimeSettings:
    """Model loading settings for the local runtime."""

    model_id: str
    base_subdir: str
    tokenizer_id: str
    tokenizer_subdir: str
    device_map: str
    torch_dtype: str
    trust_remote_code: bool = False


class LocalCaiTIRuntime:
    """Load the base model, then make noncritical adapters ready in the background."""

    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self._model_lock = threading.RLock()
        self._adapter_state_lock = threading.Lock()
        self._adapter_loader_thread: threading.Thread | None = None
        self._startup_timings: dict[str, float] = {}
        self._adapter_states = {
            task: {"state": "pending", "seconds": None, "error": None}
            for task in TASK_TO_ADAPTER
        }

        startup_started = time.perf_counter()
        self._load_dependencies()
        self.tokenizer = self._measure("tokenizer", self._load_tokenizer)
        self.base_model = self._measure("base_model", self._load_base_model)

        first_task = LLMTask.TASK1_RESPONSE_ANALYZER
        self.model = self._load_first_adapter(first_task)
        self.model.eval()
        self._startup_timings["task1_ready_total"] = round(
            time.perf_counter() - startup_started,
            3,
        )
        logger.info(
            "Local LLM Task 1 ready after %.3fs; loading remaining adapters in background.",
            self._startup_timings["task1_ready_total"],
        )
        self._start_background_adapter_loading()

    def generate_base(
        self,
        system_content: str,
        user_content: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        with self._model_lock:
            disable = getattr(self.model, "disable_adapter", None)
            context = disable() if callable(disable) else nullcontext()
            with context:
                text = self._generate_text(self.model, prompt, config)
        return GenerationResult(text=text, task=LLMTask.BASE, adapter=None, raw_text=text)

    def generate_adapter(
        self,
        task: LLMTask,
        prompt: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        adapter = resolve_adapter(task)
        with self._model_lock:
            self._ensure_adapter_loaded_locked(task)
            self.model.set_adapter(task.value)
            text = self._generate_text(self.model, prompt, config)
        return GenerationResult(text=text, task=task, adapter=adapter, raw_text=text)

    def adapter_status(self) -> dict[str, dict[str, object]]:
        """Return a stable snapshot suitable for the server health response."""

        with self._adapter_state_lock:
            return {
                task.value: dict(details)
                for task, details in self._adapter_states.items()
            }

    def startup_timings(self) -> dict[str, float]:
        with self._adapter_state_lock:
            return dict(self._startup_timings)

    def wait_for_background_adapters(self, timeout: float | None = None) -> bool:
        """Wait for background loading; primarily useful for diagnostics and tests."""

        thread = self._adapter_loader_thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _measure(self, name: str, operation):
        started = time.perf_counter()
        result = operation()
        elapsed = round(time.perf_counter() - started, 3)
        self._startup_timings[name] = elapsed
        logger.info("Local LLM %s loaded in %.3fs.", name, elapsed)
        return result

    def _load_first_adapter(self, task: LLMTask):
        self._set_adapter_state(task, "loading")
        started = time.perf_counter()
        try:
            source = self._resolve_adapter_source(resolve_adapter(task))
            model = self._peft_model_cls.from_pretrained(
                self.base_model,
                source,
                adapter_name=task.value,
            )
        except Exception as exc:
            self._finish_adapter_load(task, started, error=exc)
            raise
        self._finish_adapter_load(task, started)
        return model

    def _start_background_adapter_loading(self) -> None:
        self._adapter_loader_thread = threading.Thread(
            target=self._load_remaining_adapters,
            name="caiti-adapter-loader",
            daemon=True,
        )
        self._adapter_loader_thread.start()

    def _load_remaining_adapters(self) -> None:
        for task in TASK_TO_ADAPTER:
            if task == LLMTask.TASK1_RESPONSE_ANALYZER:
                continue
            try:
                with self._model_lock:
                    if self._adapter_state(task)["state"] == "ready":
                        continue
                    self._load_additional_adapter_locked(task)
            except Exception:
                logger.exception("Background adapter load failed for %s.", task.value)
            time.sleep(0)
        logger.info("Local LLM background adapter loading finished.")

    def _ensure_adapter_loaded_locked(self, task: LLMTask) -> None:
        state = self._adapter_state(task)
        if state["state"] == "ready":
            return
        if state["state"] == "failed":
            raise RuntimeError(
                f"Adapter {task.value} failed to load: {state['error']}"
            )
        self._load_additional_adapter_locked(task)

    def _load_additional_adapter_locked(self, task: LLMTask) -> None:
        self._set_adapter_state(task, "loading")
        started = time.perf_counter()
        try:
            source = self._resolve_adapter_source(resolve_adapter(task))
            self.model.load_adapter(source, adapter_name=task.value)
        except Exception as exc:
            self._finish_adapter_load(task, started, error=exc)
            raise
        self._finish_adapter_load(task, started)

    def _adapter_state(self, task: LLMTask) -> dict[str, object]:
        with self._adapter_state_lock:
            return dict(self._adapter_states[task])

    def _set_adapter_state(self, task: LLMTask, state: str) -> None:
        with self._adapter_state_lock:
            self._adapter_states[task] = {
                "state": state,
                "seconds": None,
                "error": None,
            }

    def _finish_adapter_load(
        self,
        task: LLMTask,
        started: float,
        error: Exception | None = None,
    ) -> None:
        elapsed = round(time.perf_counter() - started, 3)
        with self._adapter_state_lock:
            self._adapter_states[task] = {
                "state": "failed" if error else "ready",
                "seconds": elapsed,
                "error": str(error) if error else None,
            }
            self._startup_timings[f"adapter.{task.value}"] = elapsed
        if error is None:
            logger.info("Local LLM adapter %s loaded in %.3fs.", task.value, elapsed)

    def _load_dependencies(self) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
        except ImportError as exc:
            raise RuntimeError(
                "Local CaiTI runtime requires torch, transformers, peft, "
                "accelerate, and bitsandbytes."
            ) from exc

        self._torch = torch
        self._snapshot_download = snapshot_download
        self._peft_model_cls = PeftModel
        self._model_cls = AutoModelForCausalLM
        self._tokenizer_cls = AutoTokenizer
        self._stopping_criteria_cls = StoppingCriteria
        self._stopping_criteria_list_cls = StoppingCriteriaList

    def _load_tokenizer(self):
        tokenizer_ref = self.settings.tokenizer_id or self.settings.model_id
        kwargs = {
            "trust_remote_code": self.settings.trust_remote_code,
            "local_files_only": True,
        }
        if self.settings.tokenizer_subdir:
            kwargs["subfolder"] = self.settings.tokenizer_subdir
        try:
            tokenizer = self._tokenizer_cls.from_pretrained(
                tokenizer_ref,
                **kwargs,
            )
        except ValueError as exc:
            if "TokenizersBackend" not in str(exc):
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["tokenizer_type"] = "llama"
            tokenizer = self._tokenizer_cls.from_pretrained(
                tokenizer_ref,
                **fallback_kwargs,
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_base_model(self):
        kwargs = {
            "trust_remote_code": self.settings.trust_remote_code,
            "torch_dtype": self._resolve_dtype(),
            "local_files_only": True,
        }
        if self.settings.base_subdir:
            kwargs["subfolder"] = self.settings.base_subdir
        device_map = self._resolve_device_map(self.settings.device_map)
        if device_map is not None:
            kwargs["device_map"] = device_map
        return self._model_cls.from_pretrained(self.settings.model_id, **kwargs)

    def _resolve_adapter_source(self, adapter_subdir: str) -> str:
        local_candidate = Path(self.settings.model_id) / adapter_subdir
        if local_candidate.exists():
            return str(local_candidate)

        snapshot_dir = self._snapshot_download(
            repo_id=self.settings.model_id,
            allow_patterns=[f"{adapter_subdir}/*"],
            local_files_only=True,
        )
        return str(Path(snapshot_dir) / adapter_subdir)

    def _resolve_dtype(self):
        dtype_name = (self.settings.torch_dtype or "auto").strip().lower()
        if dtype_name == "auto":
            if self._torch.cuda.is_available() and self._torch.cuda.is_bf16_supported():
                return self._torch.bfloat16
            if self._torch.cuda.is_available():
                return self._torch.float16
            return self._torch.float32
        dtype = getattr(self._torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype: {self.settings.torch_dtype}")
        return dtype

    @staticmethod
    def _resolve_device_map(device_map: str):
        value = str(device_map or "").strip().lower()
        if not value:
            return None
        if value == "cuda":
            return {"": 0}
        if value.startswith("cuda:"):
            return {"": int(value.split(":", 1)[1])}
        return device_map

    def _generate_text(self, model, prompt: str, config: GenerationConfig) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_input_tokens,
            add_special_tokens=False,
        )
        device = self._input_device(model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if config.do_sample:
            kwargs["temperature"] = config.temperature
            kwargs["top_p"] = config.top_p
        if config.stop_regex:
            kwargs["stopping_criteria"] = self._build_stopping_criteria(
                config.stop_regex,
                prompt_length=inputs["input_ids"].shape[1],
            )

        with self._torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _build_stopping_criteria(self, stop_regex: str, prompt_length: int):
        pattern = re.compile(stop_regex, flags=re.IGNORECASE)
        tokenizer = self.tokenizer
        stopping_criteria_cls = self._stopping_criteria_cls
        stopping_criteria_list_cls = self._stopping_criteria_list_cls

        class RegexStoppingCriteria(stopping_criteria_cls):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                generated = input_ids[0, prompt_length:]
                if generated.numel() == 0:
                    return False
                text = tokenizer.decode(generated, skip_special_tokens=True)
                return bool(pattern.search(text))

        return stopping_criteria_list_cls([RegexStoppingCriteria()])

    @staticmethod
    def _input_device(model):
        if hasattr(model, "device"):
            return model.device
        return next(model.parameters()).device
