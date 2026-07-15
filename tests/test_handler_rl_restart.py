import unittest

from src import handler_rl
from src.handler_rl import HandlerRL


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
