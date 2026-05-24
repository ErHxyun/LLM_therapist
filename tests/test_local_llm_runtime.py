import unittest

from src.local_llm.runtime import LocalCaiTIRuntime, RuntimeSettings


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


if __name__ == "__main__":
    unittest.main()
