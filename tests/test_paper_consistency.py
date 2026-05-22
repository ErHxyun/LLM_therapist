import unittest

from src.utils import config_loader
from src.utils.io_question_lib import load_question_lib
from src.utils.rl_qtables import initialize_q_table, choose_action


EXPECTED_DIMENSION_LABELS = [
    "weight",
    "mood",
    "medication",
    "care",
    "house",
    "talk",
    "emo",
    "safe",
    "risk",
    "sleep",
    "eat",
    "work",
    "work_dayoff",
    "showup",
    "finance",
    "nutrition",
    "problem",
    "support",
    "family",
    "alcohol",
    "ciga",
    "drug",
    "hobbies",
    "creativity",
    "community",
    "support",
    "social",
    "comfortable",
    "protection",
    "productivity",
    "work_motivation",
    "coping",
    "sib",
    "arrest",
    "legal",
    "hygiene",
    "sports",
]


class PaperConsistencyTest(unittest.TestCase):
    def test_rl_config_matches_paper_state_and_hyperparameter_contract(self):
        self.assertEqual(config_loader.ITEM_N_STATES, 39)
        self.assertEqual(config_loader.EPSILON, 0.9)
        self.assertEqual(config_loader.ALPHA, 0.1)
        self.assertEqual(config_loader.GAMMA, 0.9)
        self.assertEqual(len(config_loader.ITEM_IMPORTANCE), 39)
        self.assertEqual(len(config_loader.NUMBER_QUESTIONS), 39)
        self.assertEqual(config_loader.ITEM_IMPORTANCE[0], 0)
        self.assertEqual(config_loader.ITEM_IMPORTANCE[-1], 0)
        self.assertEqual(config_loader.NUMBER_QUESTIONS[0], 0)
        self.assertEqual(config_loader.NUMBER_QUESTIONS[-1], 0)

    def test_question_library_preserves_37_dimension_order(self):
        question_lib = load_question_lib(config_loader.QUESTION_LIB_FILENAME)
        labels = [question_lib[str(i)]["1"]["label"] for i in range(1, len(question_lib) + 1)]
        self.assertEqual(len(labels), 37)
        self.assertEqual(labels, EXPECTED_DIMENSION_LABELS)

    def test_q_table_uses_start_dimension_and_end_state_columns(self):
        actions = [str(i) for i in range(config_loader.ITEM_N_STATES)]
        q_table = initialize_q_table(config_loader.ITEM_N_STATES, actions)
        self.assertEqual(q_table.shape, (39, 39))
        self.assertEqual(list(q_table.columns), actions)
        self.assertTrue((q_table["0"] == 0).all())
        self.assertTrue((q_table["38"] == 0).all())

    def test_choose_action_respects_start_and_end_masks_without_mutating_q_table(self):
        actions = [str(i) for i in range(config_loader.ITEM_N_STATES)]
        q_table = initialize_q_table(config_loader.ITEM_N_STATES, actions)
        before = q_table.copy()
        mask = [0] + [0] * 37 + [0]
        action = choose_action(0, q_table, mask, config_loader.ITEM_N_STATES, actions)
        self.assertEqual(action, "38")
        self.assertTrue(q_table.equals(before))


if __name__ == "__main__":
    unittest.main()
