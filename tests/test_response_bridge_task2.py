import unittest

from src.utils import response_bridge
from src.utils.llm_output_contracts import normalize_task1_output, normalize_task2_output


class ResponseBridgeTask2Test(unittest.TestCase):
    def test_general_parser_accepts_category_and_numeric_fallback(self):
        self.assertEqual(response_bridge._parse_task2_prediction("No"), "No")
        self.assertEqual(response_bridge._parse_task2_prediction("4"), "Question")

    def test_explicit_stop_detector_is_conservative(self):
        self.assertTrue(response_bridge.is_explicit_stop_request("I want to stop."))
        self.assertTrue(response_bridge.is_explicit_stop_request("Stop here please."))
        self.assertTrue(response_bridge.is_explicit_stop_request("I don't want to talk to you."))
        self.assertFalse(response_bridge.is_explicit_stop_request("I stopped smoking recently."))
        self.assertFalse(response_bridge.is_explicit_stop_request("I am not sure."))
        self.assertFalse(response_bridge.is_explicit_stop_request("I don't want to talk to my family."))

    def test_general_gate_is_conservative_for_domain_content(self):
        self.assertTrue(response_bridge._looks_like_general_response("Nope."))
        self.assertTrue(response_bridge._looks_like_general_response("So I don't think so."))
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

    def test_contextual_missed_days_maps_to_showing_up(self):
        original = response_bridge.classify_general_response_result
        try:
            response_bridge.classify_general_response_result = (
                lambda _answer, default=None: normalize_task2_output("No", default=default)
            )
            self.assertEqual(
                response_bridge.classify_with_task2("No missed days.", "showup"),
                ("showup", "Yes"),
            )
        finally:
            response_bridge.classify_general_response_result = original

    def test_strong_yes_start_overrides_task2_conflict(self):
        original = response_bridge.classify_general_response_result
        try:
            response_bridge.classify_general_response_result = (
                lambda _answer, default=None: normalize_task2_output("No", default=default)
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "Yes, I see my doctor and therapist regularly.",
                    "Have you been consistently visiting your doctor, therapist, or case manager?",
                    "care",
                ),
                ("care", "Yes"),
            )
        finally:
            response_bridge.classify_general_response_result = original

    def test_task2_stop_requires_explicit_user_stop(self):
        original = response_bridge.classify_general_response_result
        try:
            response_bridge.classify_general_response_result = (
                lambda _answer, default=None: normalize_task2_output("Stop", default=default)
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "I am not sure.",
                    "Have you been sleeping enough recently?",
                    "sleep",
                ),
                ("sleep", "Maybe"),
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "I want to stop.",
                    "Have you been sleeping enough recently?",
                    "sleep",
                ),
                ("sleep", "Stop"),
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
        self.assertEqual(
            response_bridge._normalize_dim_score("18_support", 2),
            ("family_support", 2),
        )
        self.assertEqual(
            response_bridge._normalize_dim_score("26_support_social", 1),
            ("social_support", 1),
        )

        self.assertEqual(
            response_bridge._normalize_dim_score("26_support", 0),
            ("social_support", 0),
        )

    def test_legacy_bare_support_only_binds_to_current_support_dimension(self):
        original = response_bridge.classify_dimension_and_score_result
        try:
            response_bridge.classify_dimension_and_score_result = (
                lambda _answer, _question: normalize_task1_output("support, 2")
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "My family has not been there for me.",
                    "Have you felt supported by your family?",
                    "family_support",
                ),
                ("family_support", 2),
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "I do not have friends I can rely on.",
                    "Do you have someone else in your support network?",
                    "social_support",
                ),
                ("social_support", 2),
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "My family has not been there for me.",
                    "Have you been sleeping enough recently?",
                    "sleep",
                ),
                ("NA", 99),
            )
        finally:
            response_bridge.classify_dimension_and_score_result = original

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

    def test_task1_cross_dimension_result_is_preserved_for_multi_segment_scoring(self):
        original = response_bridge.classify_dimension_and_score_result
        try:
            response_bridge.classify_dimension_and_score_result = (
                lambda _answer, _question: normalize_task1_output("10_sleep, 2")
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "I am seeing my doctor regularly.",
                    "Have you been consistently visiting your doctor, therapist, or case manager?",
                    "care",
                ),
                ("sleep", 2),
            )
        finally:
            response_bridge.classify_dimension_and_score_result = original

    def test_task1_adjacent_dimension_result_is_preserved(self):
        original = response_bridge.classify_dimension_and_score_result
        try:
            response_bridge.classify_dimension_and_score_result = (
                lambda _answer, _question: normalize_task1_output("emo, 1")
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "Every time I feel stressful or anxious, I just keep it to myself.",
                    "Do you have effective strategies to manage stress and difficult emotions?",
                    "coping",
                ),
                ("emo", 1),
            )
        finally:
            response_bridge.classify_dimension_and_score_result = original

    def test_task1_stop_requires_explicit_user_stop(self):
        original = response_bridge.classify_dimension_and_score_result
        try:
            response_bridge.classify_dimension_and_score_result = (
                lambda _answer, _question: normalize_task1_output("Stop, 0")
            )
            self.assertEqual(
                response_bridge.get_openai_resp(
                    "I stopped smoking recently.",
                    "Have you been smoking cigarettes recently?",
                    "ciga",
                ),
                ("NA", 99),
            )
        finally:
            response_bridge.classify_dimension_and_score_result = original


    def test_exact_yes_no_stop_skip_all_analyzer_calls(self):
        original_general = response_bridge.classify_general_response_result
        original_dimension = response_bridge.classify_dimension_and_score_result
        metrics = {}

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("exact fast path should not call an analyzer")

        try:
            response_bridge.classify_general_response_result = fail_if_called
            response_bridge.classify_dimension_and_score_result = fail_if_called
            self.assertEqual(
                response_bridge.get_openai_resp("Yes.", "Question?", "sleep", metrics),
                ("sleep", "Yes"),
            )
            self.assertEqual(
                response_bridge.get_openai_resp("No", "Question?", "sleep", metrics),
                ("sleep", "No"),
            )
            self.assertEqual(
                response_bridge.get_openai_resp("Stop", "Question?", "sleep", metrics),
                ("sleep", "Stop"),
            )
        finally:
            response_bridge.classify_general_response_result = original_general
            response_bridge.classify_dimension_and_score_result = original_dimension

        self.assertEqual(metrics["analyzer_call_count"], 0)
        self.assertEqual(metrics["source"], "exact_general_fast_path")

if __name__ == "__main__":
    unittest.main()
