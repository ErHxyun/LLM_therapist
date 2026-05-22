import unittest

from src.utils.llm_output_contracts import (
    normalize_decision_output,
    normalize_task1_output,
    normalize_task2_output,
    parse_binary_decision,
)


class LLMOutputContractsTest(unittest.TestCase):
    def test_task1_contract_accepts_adapter_continuation_output(self):
        contract = normalize_task1_output('"weight, 2"}\n{"in":"extra"')

        self.assertTrue(contract.is_valid)
        self.assertEqual(contract.dimension, "weight")
        self.assertEqual(contract.score, 2)
        self.assertEqual(contract.normalized_output, "weight, 2")

    def test_task1_contract_rejects_unknown_dimension(self):
        contract = normalize_task1_output("unknown_dimension, 2")

        self.assertFalse(contract.is_valid)
        self.assertEqual(contract.normalized_output, "NA, 99")

    def test_task2_contract_accepts_category_and_numeric_fallback(self):
        self.assertEqual(normalize_task2_output("Output: No").normalized_output, "No")
        self.assertEqual(normalize_task2_output("4").normalized_output, "Question")

    def test_binary_decision_contract_accepts_reasoner_variants(self):
        self.assertEqual(parse_binary_decision("DECISION: 0"), "0")
        self.assertEqual(normalize_decision_output("- Correct output: 1 (").normalized_output, "DECISION: 1")
        self.assertFalse(normalize_decision_output("unparseable").is_valid)


if __name__ == "__main__":
    unittest.main()
