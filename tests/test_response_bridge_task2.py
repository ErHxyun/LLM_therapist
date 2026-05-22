import unittest

from src.utils import response_bridge
from src.utils.llm_output_contracts import normalize_task1_output, normalize_task2_output


class ResponseBridgeTask2Test(unittest.TestCase):
    def test_general_parser_accepts_category_and_numeric_fallback(self):
        self.assertEqual(response_bridge._parse_task2_prediction("No"), "No")
        self.assertEqual(response_bridge._parse_task2_prediction("4"), "Question")

    def test_general_gate_is_conservative_for_domain_content(self):
        self.assertTrue(response_bridge._looks_like_general_response("Nope."))
        self.assertFalse(
            response_bridge._looks_like_general_response("I don't want to talk to my family.")
        )

    def test_task2_result_binds_to_current_dimension(self):
        original = response_bridge.classify_general_response_result
        try:
            response_bridge.classify_general_response_result = (
                lambda _answer, default=None: normalize_task2_output("No", default=default)
            )
            self.assertEqual(
                response_bridge.classify_with_task2("Nope.", "alcohol"),
                ("alcohol", "No"),
            )
        finally:
            response_bridge.classify_general_response_result = original

    def test_model_dimension_labels_normalize_to_question_lib_labels(self):
        self.assertEqual(
            response_bridge._normalize_dim_score("19_family_relationship", 2),
            ("family", 2),
        )
        self.assertEqual(
            response_bridge._normalize_dim_score("31_motivation", 1),
            ("work_motivation", 1),
        )

    def test_task1_result_is_parsed_for_non_general_response(self):
        original = response_bridge.classify_dimension_and_score_result
        try:
            response_bridge.classify_dimension_and_score_result = (
                lambda _answer, _question: normalize_task1_output("1_weight, 2")
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "My weight increased a lot recently.",
                    "Have your weight changed significantly recently?",
                    "weight",
                ),
                ("weight", 2),
            )
        finally:
            response_bridge.classify_dimension_and_score_result = original


if __name__ == "__main__":
    unittest.main()
