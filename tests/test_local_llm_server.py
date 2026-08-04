import json
import threading
import unittest
from dataclasses import dataclass
from urllib import request

from src.local_llm.server import LLMServerSettings, build_server
from src.local_llm.types import GenerationResult, LLMTask


@dataclass(frozen=True)
class _FakeSettings:
    model_id: str = "fake-model"
    device_map: str = "cuda:0"


class _FakeRuntime:
    settings = _FakeSettings()

    def adapter_status(self):
        return {
            LLMTask.TASK1_RESPONSE_ANALYZER.value: {
                "state": "ready",
                "seconds": 1.25,
                "error": None,
            },
            LLMTask.TASK2_GENERAL_RESPONSE.value: {
                "state": "loading",
                "seconds": None,
                "error": None,
            },
        }

    def startup_timings(self):
        return {"tokenizer": 0.5, "task1_ready_total": 3.0}

    def generate_base(self, system_content, user_content, config):
        return GenerationResult(
            text=f"base:{system_content}:{user_content}:{config.max_new_tokens}",
            task=LLMTask.BASE,
            adapter=None,
            raw_text="raw-base",
        )

    def generate_adapter(self, task, prompt, config):
        return GenerationResult(
            text=f"{task.value}:{prompt}:{config.max_new_tokens}",
            task=task,
            adapter=f"adapter/{task.value}",
            raw_text="raw-adapter",
        )


class LocalLLMServerTests(unittest.TestCase):
    def test_server_health_and_generate(self):
        server = build_server(LLMServerSettings(host="127.0.0.1", port=0), runtime=_FakeRuntime())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with request.urlopen(f"{base_url}/health", timeout=2) as response:
                health = json.loads(response.read().decode("utf-8"))
            self.assertTrue(health["ok"])
            self.assertEqual(health["readiness"], "task1_ready")
            self.assertEqual(health["model_id"], "fake-model")
            self.assertEqual(
                health["adapters"][LLMTask.TASK2_GENERAL_RESPONSE.value]["state"],
                "loading",
            )
            self.assertEqual(health["startup_timings_seconds"]["task1_ready_total"], 3.0)

            payload = json.dumps(
                {
                    "task": LLMTask.TASK1_RESPONSE_ANALYZER.value,
                    "system_content": "sys",
                    "user_content": "user",
                    "max_new_tokens": 7,
                }
            ).encode("utf-8")
            req = request.Request(
                f"{base_url}/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=2) as response:
                result = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

        self.assertEqual(result["task"], LLMTask.TASK1_RESPONSE_ANALYZER.value)
        self.assertEqual(result["adapter"], "adapter/task1_response_analyzer")
        self.assertIn("sys\nuser", result["text"])


if __name__ == "__main__":
    unittest.main()
