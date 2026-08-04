import threading
import unittest

from src.local_llm.runtime import LocalCaiTIRuntime, RuntimeSettings
from src.local_llm.types import GenerationConfig, LLMTask


class LocalLLMRuntimeTest(unittest.TestCase):
    def test_tokenizer_load_falls_back_to_llama_for_tokenizers_backend(self):
        calls = []

        class FakeTokenizer:
            pad_token_id = None
            eos_token_id = 128009
            eos_token = "<|eot_id|>"
            pad_token = None

        class FakeTokenizerClass:
            @staticmethod
            def from_pretrained(ref, **kwargs):
                calls.append((ref, kwargs))
                if "tokenizer_type" not in kwargs:
                    raise ValueError(
                        "Tokenizer class TokenizersBackend does not exist or is not currently imported."
                    )
                return FakeTokenizer()

        runtime = object.__new__(LocalCaiTIRuntime)
        runtime.settings = RuntimeSettings(
            model_id="xxue752/caiti_best_model",
            base_subdir="compressed_model_int4",
            tokenizer_id="xxue752/caiti_best_model",
            tokenizer_subdir="compressed_model_int4",
            device_map="auto",
            torch_dtype="auto",
        )
        runtime._tokenizer_cls = FakeTokenizerClass

        tokenizer = runtime._load_tokenizer()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["subfolder"], "compressed_model_int4")
        self.assertTrue(calls[0][1]["local_files_only"])
        self.assertNotIn("tokenizer_type", calls[0][1])
        self.assertEqual(calls[1][1]["tokenizer_type"], "llama")
        self.assertEqual(tokenizer.pad_token, "<|eot_id|>")

    def test_cuda_device_map_string_targets_single_cuda_device(self):
        runtime = object.__new__(LocalCaiTIRuntime)

        self.assertEqual(runtime._resolve_device_map("cuda:0"), {"": 0})
        self.assertEqual(runtime._resolve_device_map("cuda"), {"": 0})
        self.assertEqual(runtime._resolve_device_map("cuda:1"), {"": 1})
        self.assertEqual(runtime._resolve_device_map("auto"), "auto")
        self.assertIsNone(runtime._resolve_device_map(""))

    def test_constructor_returns_after_task1_while_other_adapters_load(self):
        additional_started = threading.Event()
        allow_additional = threading.Event()

        class FakeModel:
            def eval(self):
                return self

            def load_adapter(self, source, adapter_name):
                additional_started.set()
                allow_additional.wait(timeout=2)

        class FakePeft:
            @staticmethod
            def from_pretrained(base_model, source, adapter_name):
                return FakeModel()

        class FakeRuntime(LocalCaiTIRuntime):
            def _load_dependencies(self):
                self._peft_model_cls = FakePeft

            def _load_tokenizer(self):
                return object()

            def _load_base_model(self):
                return object()

            def _resolve_adapter_source(self, adapter_subdir):
                return adapter_subdir

        runtime = FakeRuntime(self._settings())
        self.assertTrue(additional_started.wait(timeout=1))
        status = runtime.adapter_status()
        self.assertEqual(
            status[LLMTask.TASK1_RESPONSE_ANALYZER.value]["state"],
            "ready",
        )
        self.assertEqual(
            status[LLMTask.TASK2_GENERAL_RESPONSE.value]["state"],
            "loading",
        )

        allow_additional.set()
        self.assertTrue(runtime.wait_for_background_adapters(timeout=2))
        self.assertTrue(
            all(item["state"] == "ready" for item in runtime.adapter_status().values())
        )

    def test_requested_pending_adapter_is_loaded_before_generation(self):
        loaded = []

        class FakeModel:
            def eval(self):
                return self

            def load_adapter(self, source, adapter_name):
                loaded.append(adapter_name)

            def set_adapter(self, adapter_name):
                self.active_adapter = adapter_name

        class FakePeft:
            @staticmethod
            def from_pretrained(base_model, source, adapter_name):
                model = FakeModel()
                model.active_adapter = adapter_name
                return model

        class FakeRuntime(LocalCaiTIRuntime):
            def _load_dependencies(self):
                self._peft_model_cls = FakePeft

            def _load_tokenizer(self):
                return object()

            def _load_base_model(self):
                return object()

            def _resolve_adapter_source(self, adapter_subdir):
                return adapter_subdir

            def _start_background_adapter_loading(self):
                self._adapter_loader_thread = None

            def _generate_text(self, model, prompt, config):
                return f"generated:{model.active_adapter}"

        runtime = FakeRuntime(self._settings())
        result = runtime.generate_adapter(
            LLMTask.TASK4_CBT_STAGE2,
            "prompt",
            GenerationConfig(max_new_tokens=4),
        )

        self.assertEqual(loaded, [LLMTask.TASK4_CBT_STAGE2.value])
        self.assertEqual(result.text, "generated:task4_cbt_stage2")
        self.assertEqual(
            runtime.adapter_status()[LLMTask.TASK4_CBT_STAGE2.value]["state"],
            "ready",
        )

    def test_failed_background_adapter_is_reported_and_never_used(self):
        class FakeModel:
            def eval(self):
                return self

            def load_adapter(self, source, adapter_name):
                if adapter_name == LLMTask.TASK2_GENERAL_RESPONSE.value:
                    raise OSError("bad adapter")

        class FakePeft:
            @staticmethod
            def from_pretrained(base_model, source, adapter_name):
                return FakeModel()

        class FakeRuntime(LocalCaiTIRuntime):
            def _load_dependencies(self):
                self._peft_model_cls = FakePeft

            def _load_tokenizer(self):
                return object()

            def _load_base_model(self):
                return object()

            def _resolve_adapter_source(self, adapter_subdir):
                return adapter_subdir

        runtime = FakeRuntime(self._settings())
        self.assertTrue(runtime.wait_for_background_adapters(timeout=2))
        status = runtime.adapter_status()[LLMTask.TASK2_GENERAL_RESPONSE.value]
        self.assertEqual(status["state"], "failed")
        self.assertIn("bad adapter", status["error"])

        with self.assertRaisesRegex(RuntimeError, "bad adapter"):
            runtime.generate_adapter(
                LLMTask.TASK2_GENERAL_RESPONSE,
                "prompt",
                GenerationConfig(max_new_tokens=4),
            )

    @staticmethod
    def _settings():
        return RuntimeSettings(
            model_id="fake-model",
            base_subdir="base",
            tokenizer_id="fake-model",
            tokenizer_subdir="base",
            device_map="cpu",
            torch_dtype="float32",
        )


if __name__ == "__main__":
    unittest.main()
