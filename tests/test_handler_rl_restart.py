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

    def test_active_stage_mask_limits_rl_to_earliest_unfinished_stage(self):
        handler = HandlerRL()
        handler.screening_stages = [
            {"name": "daily_life", "item_ids": [1, 2]},
            {"name": "social", "item_ids": [3, 4]},
        ]
        item_mask = [0, 1, 1, 1, 1, 0]
        original_mask = list(item_mask)

        stage_mask, stage_index, stage_name = handler._active_stage_mask(item_mask)

        self.assertEqual(stage_mask, [0, 1, 1, 0, 0, 0])
        self.assertEqual(stage_index, 0)
        self.assertEqual(stage_name, "daily_life")
        self.assertEqual(item_mask, original_mask)

    def test_active_stage_mask_advances_and_skips_cross_scored_later_item(self):
        handler = HandlerRL()
        handler.screening_stages = [
            {"name": "daily_life", "item_ids": [1, 2]},
            {"name": "social", "item_ids": [3, 4]},
        ]
        # The daily stage is complete. Item 3 was already cross-scored by an
        # earlier multi-segment response, so only item 4 remains in stage 2.
        item_mask = [0, 0, 0, 0, 1, 0]

        stage_mask, stage_index, stage_name = handler._active_stage_mask(item_mask)

        self.assertEqual(stage_mask, [0, 0, 0, 0, 1, 0])
        self.assertEqual(stage_index, 1)
        self.assertEqual(stage_name, "social")

    def test_resolve_screening_stages_rejects_missing_dimensions(self):
        handler = HandlerRL()
        handler.question_lib = {
            "1": {"1": {"label": "sleep", "score": [], "notes": []}},
            "2": {"1": {"label": "eat", "score": [], "notes": []}},
        }
        original_enabled = handler_rl.STAGED_SCREENING_ENABLED
        original_stages = handler_rl.STAGED_SCREENING_STAGES

        try:
            handler_rl.STAGED_SCREENING_ENABLED = True
            handler_rl.STAGED_SCREENING_STAGES = [
                {"name": "daily_life", "intro": "Daily routine.", "dimensions": ["sleep"]}
            ]
            with self.assertRaisesRegex(ValueError, "does not cover"):
                handler._resolve_screening_stages()
        finally:
            handler_rl.STAGED_SCREENING_ENABLED = original_enabled
            handler_rl.STAGED_SCREENING_STAGES = original_stages

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


    def test_active_q_table_update_is_visible_to_next_step(self):
        handler = HandlerRL()
        active_q_table = handler_rl.pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            columns=["0", "1"],
        )

        before, after, target = handler._update_active_item_q_table(
            active_q_table, 0, "1", "terminal", 2.0
        )

        self.assertEqual(before, 0.0)
        self.assertEqual(target, 2.0)
        self.assertAlmostEqual(after, handler_rl.ALPHA * 2.0)
        self.assertEqual(active_q_table.loc[0, "1"], after)
        before_next, _, _ = handler._update_active_item_q_table(
            active_q_table, 0, "1", "terminal", 2.0
        )
        self.assertEqual(before_next, after)

    def test_terminal_screening_validation_is_queued_for_first_cbt_output(self):
        handler = HandlerRL()
        queued_prefixes = []
        original_pop = handler_rl.pop_pending_validation_for_workflow_transition
        original_set_prefix = handler_rl.set_question_prefix

        try:
            handler_rl.pop_pending_validation_for_workflow_transition = (
                lambda: "It makes sense that this has been difficult."
            )
            handler_rl.set_question_prefix = lambda text: queued_prefixes.append(text)

            transferred = handler._queue_pending_screening_validation_for_cbt()
        finally:
            handler_rl.pop_pending_validation_for_workflow_transition = original_pop
            handler_rl.set_question_prefix = original_set_prefix

        self.assertEqual(
            transferred,
            "It makes sense that this has been difficult.",
        )
        self.assertEqual(queued_prefixes, [transferred])

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
