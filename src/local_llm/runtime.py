"""Local Llama + LoRA runtime for CaiTI task modules."""

from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass

from src.local_llm.routing import TASK_TO_ADAPTER, resolve_adapter
from src.local_llm.types import GenerationConfig, GenerationResult, LLMTask


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
    """Load one base model and switch between preloaded CaiTI adapters."""

    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self._load_dependencies()
        self.tokenizer = self._load_tokenizer()
        self.base_model = self._load_base_model()
        self.model = self._load_adapters()
        self.model.eval()

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
        self.model.set_adapter(task.value)
        text = self._generate_text(self.model, prompt, config)
        return GenerationResult(text=text, task=task, adapter=adapter, raw_text=text)

    def _load_dependencies(self) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
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

    def _load_tokenizer(self):
        tokenizer_ref = self.settings.tokenizer_id or self.settings.model_id
        kwargs = {
            "trust_remote_code": self.settings.trust_remote_code,
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
        }
        if self.settings.base_subdir:
            kwargs["subfolder"] = self.settings.base_subdir
        device_map = self._resolve_device_map(self.settings.device_map)
        if device_map is not None:
            kwargs["device_map"] = device_map
        return self._model_cls.from_pretrained(self.settings.model_id, **kwargs)

    def _load_adapters(self):
        adapter_items = list(TASK_TO_ADAPTER.items())
        first_task, first_adapter = adapter_items[0]
        first_adapter_source = self._resolve_adapter_source(first_adapter)
        model = self._peft_model_cls.from_pretrained(
            self.base_model,
            first_adapter_source,
            adapter_name=first_task.value,
        )
        for task, adapter in adapter_items[1:]:
            adapter_source = self._resolve_adapter_source(adapter)
            model.load_adapter(
                adapter_source,
                adapter_name=task.value,
            )
        return model

    def _resolve_adapter_source(self, adapter_subdir: str) -> str:
        local_candidate = Path(self.settings.model_id) / adapter_subdir
        if local_candidate.exists():
            return str(local_candidate)

        snapshot_dir = self._snapshot_download(
            repo_id=self.settings.model_id,
            allow_patterns=[f"{adapter_subdir}/*"],
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
        }
        if config.do_sample:
            kwargs["temperature"] = config.temperature
            kwargs["top_p"] = config.top_p

        with self._torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _input_device(model):
        if hasattr(model, "device"):
            return model.device
        return next(model.parameters()).device
