import unittest

from src.local_llm.routing import is_adapter_task, resolve_adapter
from src.local_llm.types import LLMTask


class LocalLLMRoutingTest(unittest.TestCase):
    def test_resolve_adapter_known_tasks(self):
        self.assertEqual(
            resolve_adapter(LLMTask.TASK1_RESPONSE_ANALYZER),
            "adapters/task1_response_analyzer",
        )
        self.assertEqual(
            resolve_adapter(LLMTask.TASK2_GENERAL_RESPONSE),
            "adapters/task2_general_response",
        )
        self.assertEqual(
            resolve_adapter(LLMTask.TASK3_RV_REASONER),
            "adapters/task3_rv_reasoner",
        )
        self.assertEqual(
            resolve_adapter(LLMTask.TASK4_CBT_STAGE1),
            "adapters/task4_cbt_stage1",
        )
        self.assertEqual(
            resolve_adapter(LLMTask.TASK4_CBT_STAGE2),
            "adapters/task4_cbt_stage2",
        )
        self.assertEqual(
            resolve_adapter(LLMTask.TASK4_CBT_STAGE3),
            "adapters/task4_cbt_stage3",
        )

    def test_base_task_has_no_adapter(self):
        self.assertFalse(is_adapter_task(LLMTask.BASE))
        with self.assertRaises(ValueError):
            resolve_adapter(LLMTask.BASE)

    def test_all_non_base_tasks_are_adapter_tasks(self):
        for task in LLMTask:
            if task == LLMTask.BASE:
                continue
            self.assertTrue(is_adapter_task(task))
            self.assertTrue(resolve_adapter(task).startswith("adapters/"))


if __name__ == "__main__":
    unittest.main()
