import unittest

from src import handler_rl
from src.handler_rl import HandlerRL
from src.questioner import QuestionTurnOutcome


class HandlerRLRestartMaskTest(unittest.TestCase):
    def test_build_item_mask_only_keeps_unanswered_dimensions(self):
        handler = HandlerRL()
        handler.question_lib = {
            "1": {"1": {"label": "sleep", "score": [2], "notes": []}},
            "2": {"1": {"label": "eat", "score": [], "notes": []}},
            "3": {"1": {"label": "mood", "score": [0], "notes": []}},
            "4": {"1": {"label": "hygiene", "score": [], "notes": []}},
        }

        mask = handler._build_item_mask(4)

        self.assertEqual(mask, [0, 0, 1, 0, 1, 0])

    def test_sync_answered_item_mask_removes_cross_scored_dimension(self):
        handler = HandlerRL()
        handler.question_lib = {
            "1": {"1": {"label": "sleep", "score": [2], "notes": []}},
            "2": {"1": {"label": "eat", "score": [0], "notes": []}},
            "3": {"1": {"label": "mood", "score": [], "notes": []}},
        }
        mask = [0, 0, 1, 1, 0]

        cleared = handler._sync_answered_item_mask(mask)

        self.assertEqual(cleared, 1)
        self.assertEqual(mask, [0, 0, 0, 1, 0])

    def test_question_outcome_masks_current_and_cross_scored_dimensions(self):
        handler = HandlerRL()
        mask = [0, 1, 1, 1, 0]
        outcome = QuestionTurnOutcome(
            reward=2.0,
            terminate=0,
            previous_question="",
            covered_item_ids={1, 2},
            current_answered=True,
        )

        masked = handler._apply_question_outcome_to_mask(
            mask,
            current_item_index=1,
            outcome=outcome,
        )

        self.assertEqual(masked, {1, 2})
        self.assertEqual(mask, [0, 0, 0, 1, 0])

    def test_question_outcome_restores_unanswered_current_dimension(self):
        handler = HandlerRL()
        # Simulate the old pre-mask state to prove an unanswered current item is restored.
        mask = [0, 0, 1, 1, 0]
        outcome = QuestionTurnOutcome(
            reward=0.0,
            terminate=0,
            previous_question="",
            covered_item_ids={2},
            current_answered=False,
        )

        masked = handler._apply_question_outcome_to_mask(
            mask,
            current_item_index=1,
            outcome=outcome,
        )

        self.assertEqual(masked, {2})
        self.assertEqual(mask, [0, 1, 0, 1, 0])

    def test_uncommitted_score_two_keeps_current_dimension_available(self):
        handler = HandlerRL()
        mask = [0, 1, 1, 0]
        outcome = QuestionTurnOutcome(
            reward=0.0,
            terminate=0,
            previous_question="Could you tell me more?",
            covered_item_ids=set(),
            current_answered=False,
        )

        handler._apply_question_outcome_to_mask(mask, 1, outcome)

        self.assertEqual(mask, [0, 1, 1, 0])


    def test_persist_runtime_question_lib_writes_canonical_user_file(self):
        handler = HandlerRL()
        handler.question_lib = {
            "1": {"1": {"label": "sleep", "score": [2], "notes": []}},
        }
        calls = []
        original_save = handler_rl.save_question_lib
        original_path = handler_rl.QUESTION_LIB_FILENAME

        try:
            handler_rl.QUESTION_LIB_FILENAME = "data/users/006/libs/question_lib_v4.json"
            handler_rl.save_question_lib = lambda path, payload: calls.append((path, payload))
            handler._persist_runtime_question_lib()
        finally:
            handler_rl.save_question_lib = original_save
            handler_rl.QUESTION_LIB_FILENAME = original_path

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "data/users/006/libs/question_lib_v4.json")
        self.assertEqual(calls[0][1], handler.question_lib)


if __name__ == "__main__":
    unittest.main()
