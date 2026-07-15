import json
import tempfile
import unittest
from pathlib import Path

from src.runtime import user_context as uc
from src.utils import config_loader


class UserContextTests(unittest.TestCase):
    def test_normalize_spoken_user_id_converts_digit_words(self):
        self.assertEqual(uc.normalize_spoken_user_id("eight zero eight zero"), "8080")
        self.assertEqual(uc.normalize_spoken_user_id("AB 01"), "AB 01")

    def test_activate_user_context_prepares_user_paths_and_copies_question_lib(self):
        original_users_root = uc._USERS_ROOT_DIR
        original_question_lib = uc._DEFAULT_QUESTION_LIB_PATH
        original_subject_id = config_loader.SUBJECT_ID
        original_data_dir = config_loader.DATA_DIR
        original_log_dir = config_loader.LOG_DIR
        original_result_dir = config_loader.RESULT_DIR
        original_report_file = config_loader.REPORT_FILE
        original_notes_file = config_loader.NOTES_FILE
        original_record_csv = config_loader.RECORD_CSV
        original_intermission_db_path = config_loader.INTERMISSION_DB_PATH
        original_intermission_results_json_path = config_loader.INTERMISSION_RESULTS_JSON_PATH
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                template = root / "question_lib_v4.json"
                template.write_text('{"1":{"1":{"label":"sleep","score":[],"notes":[]}}}', encoding="utf-8")
                uc._USERS_ROOT_DIR = str(root / "users")
                uc._DEFAULT_QUESTION_LIB_PATH = str(template)

                context = uc.activate_user_context("eight zero eight zero", "Alice Example")

                self.assertEqual(context.subject_id, "8080")
                self.assertEqual(context.raw_subject_id, "8080")
                self.assertEqual(context.display_name, "Alice Example")
                self.assertTrue(Path(context.question_lib_path).exists())
                self.assertTrue(Path(context.profile_path).exists())
                self.assertTrue(Path(context.session_log_file).parent.exists())
                self.assertTrue(str(context.intermission_results_json_path).endswith("phq_gad_results.json"))
                self.assertEqual(config_loader.SUBJECT_ID, "8080")
                self.assertEqual(config_loader.RECORD_CSV, context.record_csv)
                self.assertEqual(config_loader.INTERMISSION_DB_PATH, context.structured_log_db_path)
                self.assertEqual(
                    config_loader.INTERMISSION_RESULTS_JSON_PATH,
                    context.intermission_results_json_path,
                )
                profile = json.loads(Path(context.profile_path).read_text(encoding="utf-8"))
                self.assertEqual(profile["display_name"], "Alice Example")
                self.assertEqual(profile["raw_subject_id"], "8080")
                self.assertEqual(profile["subject_id"], "8080")
        finally:
            uc._USERS_ROOT_DIR = original_users_root
            uc._DEFAULT_QUESTION_LIB_PATH = original_question_lib
            config_loader.SUBJECT_ID = original_subject_id
            config_loader.DATA_DIR = original_data_dir
            config_loader.LOG_DIR = original_log_dir
            config_loader.RESULT_DIR = original_result_dir
            config_loader.REPORT_FILE = original_report_file
            config_loader.NOTES_FILE = original_notes_file
            config_loader.RECORD_CSV = original_record_csv
            config_loader.INTERMISSION_DB_PATH = original_intermission_db_path
            config_loader.INTERMISSION_RESULTS_JSON_PATH = original_intermission_results_json_path


if __name__ == "__main__":
    unittest.main()
