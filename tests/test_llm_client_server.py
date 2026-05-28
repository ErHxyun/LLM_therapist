import json
import unittest

from src.local_llm.types import GenerationResult, LLMTask
from src.utils import llm_client


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class LLMClientServerTests(unittest.TestCase):
    def test_preload_uses_remote_health_when_server_url_is_set(self):
        calls = []
        originals = {
            "server_url": llm_client.LOCAL_LLM_SERVER_URL,
            "urlopen": llm_client.urlrequest.urlopen,
            "get_runtime": llm_client._get_runtime,
        }
        try:
            llm_client.LOCAL_LLM_SERVER_URL = "http://127.0.0.1:8890"
            llm_client.urlrequest.urlopen = lambda url, timeout=None: calls.append((url, timeout)) or _FakeHTTPResponse(
                {"ok": True}
            )
            llm_client._get_runtime = lambda: (_ for _ in ()).throw(AssertionError("local runtime should not load"))

            result = llm_client.preload_llm_runtime()
        finally:
            llm_client.LOCAL_LLM_SERVER_URL = originals["server_url"]
            llm_client.urlrequest.urlopen = originals["urlopen"]
            llm_client._get_runtime = originals["get_runtime"]

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls[0][0], "http://127.0.0.1:8890/health")

    def test_adapter_completion_uses_remote_generate(self):
        requests = []
        originals = {
            "server_url": llm_client.LOCAL_LLM_SERVER_URL,
            "urlopen": llm_client.urlrequest.urlopen,
            "get_runtime": llm_client._get_runtime,
        }
        try:
            llm_client.LOCAL_LLM_SERVER_URL = "http://127.0.0.1:8890/"

            def fake_urlopen(req, timeout=None):
                requests.append((req, timeout))
                return _FakeHTTPResponse(
                    {
                        "text": "2",
                        "task": LLMTask.TASK1_RESPONSE_ANALYZER.value,
                        "adapter": "adapters/task1_response_analyzer",
                        "raw_text": "2",
                    }
                )

            llm_client.urlrequest.urlopen = fake_urlopen
            llm_client._get_runtime = lambda: (_ for _ in ()).throw(AssertionError("local runtime should not load"))

            result = llm_client.llm_complete_task(
                LLMTask.TASK1_RESPONSE_ANALYZER,
                "system",
                "user",
                max_new_tokens=3,
            )
        finally:
            llm_client.LOCAL_LLM_SERVER_URL = originals["server_url"]
            llm_client.urlrequest.urlopen = originals["urlopen"]
            llm_client._get_runtime = originals["get_runtime"]

        request_body = json.loads(requests[0][0].data.decode("utf-8"))
        self.assertEqual(requests[0][0].full_url, "http://127.0.0.1:8890/generate")
        self.assertEqual(request_body["task"], LLMTask.TASK1_RESPONSE_ANALYZER.value)
        self.assertEqual(request_body["max_new_tokens"], 3)
        self.assertNotIn("stop_regex", request_body)
        self.assertEqual(result.text, "2")
        self.assertEqual(result.adapter, "adapters/task1_response_analyzer")

    def test_decision_adapter_completion_sends_remote_stop_regex(self):
        requests = []
        originals = {
            "server_url": llm_client.LOCAL_LLM_SERVER_URL,
            "urlopen": llm_client.urlrequest.urlopen,
            "get_runtime": llm_client._get_runtime,
        }
        try:
            llm_client.LOCAL_LLM_SERVER_URL = "http://127.0.0.1:8890"

            def fake_urlopen(req, timeout=None):
                requests.append((req, timeout))
                return _FakeHTTPResponse(
                    {
                        "text": "0",
                        "task": LLMTask.TASK4_CBT_STAGE1.value,
                        "adapter": "adapters/task4_cbt_stage1",
                        "raw_text": "0",
                    }
                )

            llm_client.urlrequest.urlopen = fake_urlopen
            llm_client._get_runtime = lambda: (_ for _ in ()).throw(AssertionError("local runtime should not load"))

            llm_client.llm_complete_task(
                LLMTask.TASK4_CBT_STAGE1,
                "system",
                "user",
                max_new_tokens=8,
            )
        finally:
            llm_client.LOCAL_LLM_SERVER_URL = originals["server_url"]
            llm_client.urlrequest.urlopen = originals["urlopen"]
            llm_client._get_runtime = originals["get_runtime"]

        request_body = json.loads(requests[0][0].data.decode("utf-8"))
        self.assertEqual(request_body["task"], LLMTask.TASK4_CBT_STAGE1.value)
        self.assertIn("stop_regex", request_body)

    def test_decision_adapter_completion_uses_local_stop_regex(self):
        captured = []

        class FakeRuntime:
            def generate_adapter(self, task, prompt, config):
                captured.append((task, prompt, config))
                return GenerationResult(
                    text="0",
                    task=task,
                    adapter="adapters/task4_cbt_stage2",
                    raw_text="0",
                )

        originals = {
            "server_url": llm_client.LOCAL_LLM_SERVER_URL,
            "get_runtime": llm_client._get_runtime,
        }
        try:
            llm_client.LOCAL_LLM_SERVER_URL = ""
            llm_client._get_runtime = lambda: FakeRuntime()

            llm_client.llm_complete_task(
                LLMTask.TASK4_CBT_STAGE2,
                "system",
                "user",
                max_new_tokens=8,
            )
        finally:
            llm_client.LOCAL_LLM_SERVER_URL = originals["server_url"]
            llm_client._get_runtime = originals["get_runtime"]

        self.assertEqual(captured[0][0], LLMTask.TASK4_CBT_STAGE2)
        self.assertIsNotNone(captured[0][2].stop_regex)


if __name__ == "__main__":
    unittest.main()
