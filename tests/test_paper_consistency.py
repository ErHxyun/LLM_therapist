import unittest
import re
from pathlib import Path

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


def _documented_question_variants():
    doc_path = Path("docs/question_variants.md")
    sections = {}
    current = None
    for raw in doc_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = re.match(r"^(\d+)\.\s+(.+?)\s+\(`([^`]+)`\)\s*$", line)
        if heading:
            current = {
                "index": int(heading.group(1)),
                "title": heading.group(2),
                "label": heading.group(3),
                "questions": [],
            }
            sections[str(current["index"])] = current
            continue
        question = re.match(r"^(\d+)\.\s*(.+)$", line)
        if question and current:
            current["questions"].append(question.group(2).strip())
    return sections


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

    def test_question_library_matches_documented_variants_exactly(self):
        question_lib = load_question_lib(config_loader.QUESTION_LIB_FILENAME)
        documented = _documented_question_variants()
        self.assertEqual(len(documented), 37)
        for i in range(1, len(question_lib) + 1):
            key = str(i)
            self.assertIn(key, documented)
            self.assertEqual(question_lib[key]["1"]["label"], documented[key]["label"])
            questions = question_lib[key]["1"]["question"]
            self.assertGreaterEqual(len(questions), 1, f"dimension {i} should have at least one variant")
            self.assertEqual(questions, documented[key]["questions"])

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
