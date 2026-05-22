import importlib.util
import sys
import unittest
from pathlib import Path

from src.local_llm.types import LLMTask


def _load_smoke_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test_adapters.py"
    spec = importlib.util.spec_from_file_location("smoke_test_adapters", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SmokeTestAdaptersScriptTest(unittest.TestCase):
    def test_build_smoke_cases_covers_all_adapter_tasks(self):
        smoke = _load_smoke_module()
        cases = smoke.build_smoke_cases()

        self.assertEqual(
            [case.task for case in cases],
            [
                LLMTask.TASK1_RESPONSE_ANALYZER,
                LLMTask.TASK2_GENERAL_RESPONSE,
                LLMTask.TASK3_RV_REASONER,
                LLMTask.TASK4_CBT_STAGE1,
                LLMTask.TASK4_CBT_STAGE2,
                LLMTask.TASK4_CBT_STAGE3,
            ],
        )

    def test_smoke_case_payloads_match_current_inference_formats(self):
        smoke = _load_smoke_module()
        cases = {case.name: case for case in smoke.build_smoke_cases()}

        self.assertEqual(
            cases["task1_response_analyzer"].payload,
            '{"in":"My weight increased a lot recently.", "res":',
        )
        self.assertEqual(cases["task2_general_response"].payload, "Response: Nope.")
        self.assertIn('"Topic":', cases["task3_rv_reasoner_invalid"].payload)
        self.assertIn('"Follow Up Response":', cases["task3_rv_reasoner_invalid"].payload)
        self.assertTrue(cases["task4_cbt_stage1"].payload.startswith('"STATEMENT:'))
        self.assertIn("UNHELPFUL_THOUGHTS:", cases["task4_cbt_stage1"].payload)
        self.assertIn("CHALLENGE:", cases["task4_cbt_stage2"].payload)
        self.assertIn("REFRAME:", cases["task4_cbt_stage3"].payload)

    def test_normalizers_match_parser_outputs(self):
        smoke = _load_smoke_module()

        self.assertEqual(smoke._normalize_task1("1_weight, 2"), "weight, 2")
        self.assertEqual(smoke._normalize_task2("2"), "No")
        self.assertEqual(smoke._normalize_decision("DECISION: 0"), "DECISION: 0")
        self.assertEqual(smoke._normalize_cbt_decision("1"), "DECISION: 1")

    def test_contract_validators_detect_invalid_outputs(self):
        smoke = _load_smoke_module()

        self.assertTrue(smoke._validate_task1('"weight, 2"}')[0])
        self.assertFalse(smoke._validate_task1("unparseable")[0])
        self.assertTrue(smoke._validate_task2("Output: No")[0])
        self.assertFalse(smoke._validate_task2("unparseable")[0])
        self.assertTrue(smoke._validate_decision("- Correct output: 1 (")[0])
        self.assertFalse(smoke._validate_decision("unparseable")[0])


if __name__ == "__main__":
    unittest.main()
