import json
import tempfile
import unittest
from pathlib import Path

from src.intermission.runner import IntermissionRunner, IntermissionSettings
from src.runtime import session_context as sc
from src.runtime import user_context as uc
from src.runtime.status_monitor import StatusMonitor, StatusMonitorSettings
from src.utils import config_loader


class _SilentTTS:
    def speak(self, _text):
        return None


class SessionStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.template = self.root / "question_lib_v4.json"
        self.template.write_text('{"1": {"1": {"label": "sleep", "score": null}}}\n', encoding="utf-8")
        self.saved = {
            "users_root": uc._USERS_ROOT_DIR,
            "template": uc._DEFAULT_QUESTION_LIB_PATH,
            "enabled": config_loader.SESSION_ENABLED,
            "cutover": config_loader.SESSION_CUTOVER_PARTICIPANT_NUMBER,
            "protect": config_loader.SESSION_PROTECT_PRE_CUTOVER_PARTICIPANTS,
            "resume": config_loader.SESSION_RESUME_INCOMPLETE,
            "current": sc._CURRENT_SESSION,
        }
        uc._USERS_ROOT_DIR = str(self.root / "users")
        uc._DEFAULT_QUESTION_LIB_PATH = str(self.template)
        config_loader.SESSION_ENABLED = True
        config_loader.SESSION_CUTOVER_PARTICIPANT_NUMBER = 26
        config_loader.SESSION_PROTECT_PRE_CUTOVER_PARTICIPANTS = True
        config_loader.SESSION_RESUME_INCOMPLETE = True
        sc._CURRENT_SESSION = None

    def tearDown(self):
        uc._USERS_ROOT_DIR = self.saved["users_root"]
        uc._DEFAULT_QUESTION_LIB_PATH = self.saved["template"]
        config_loader.SESSION_ENABLED = self.saved["enabled"]
        config_loader.SESSION_CUTOVER_PARTICIPANT_NUMBER = self.saved["cutover"]
        config_loader.SESSION_PROTECT_PRE_CUTOVER_PARTICIPANTS = self.saved["protect"]
        config_loader.SESSION_RESUME_INCOMPLETE = self.saved["resume"]
        sc._CURRENT_SESSION = self.saved["current"]
        self.tmp.cleanup()

    def test_participants_one_through_twenty_five_are_protected(self):
        for participant_id in ("1", "001", "15", "025"):
            self.assertTrue(sc.is_protected_participant(participant_id))
            with self.assertRaises(PermissionError):
                sc.create_new_session(participant_id)
        self.assertFalse((self.root / "users").exists())

    def test_new_sessions_are_isolated_for_participant_twenty_six(self):
        first = sc.create_new_session("26", "Alice")
        Path(first.question_lib_path).write_text('{"changed": true}\n', encoding="utf-8")
        Path(first.q_tables_dir, "item_qtable_026.csv").write_text("first\n", encoding="utf-8")

        second = sc.create_new_session("026", "Alice")

        self.assertEqual(first.participant_id, "026")
        self.assertEqual(second.participant_id, "026")
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertNotEqual(first.session_dir, second.session_dir)
        self.assertEqual(Path(second.question_lib_path).read_text(encoding="utf-8"), self.template.read_text(encoding="utf-8"))
        self.assertEqual(list(Path(second.q_tables_dir).iterdir()), [])
        self.assertTrue(Path(first.q_tables_dir, "item_qtable_026.csv").exists())

    def test_only_incomplete_session_is_resumable(self):
        context = sc.create_new_session("026", "Alice")
        created = sc.find_resumable_session("26")
        self.assertIsNotNone(created)
        self.assertEqual(created.session_id, context.session_id)

        sc.update_session_status(context, "ACTIVE")
        resumed = sc.find_resumable_session("026")
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.session_id, context.session_id)
        self.assertTrue(resumed.resumed)

        sc.update_session_status(context, "COMPLETED")
        self.assertIsNone(sc.find_resumable_session("026"))
        payload = json.loads(Path(context.manifest_path).read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "COMPLETED")

    def test_abandoned_session_is_not_offered_again(self):
        context = sc.create_new_session("026", "Alice")
        sc.update_session_status(context, "INTERRUPTED")
        self.assertIsNotNone(sc.find_resumable_session("026"))

        sc.update_session_status(context, "ABANDONED", abandonment_reason="participant chose new")

        self.assertIsNone(sc.find_resumable_session("026"))


class IntermissionSessionIsolationTests(unittest.TestCase):
    def test_begin_session_resets_state_and_resume_reads_only_that_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = IntermissionRunner(
                settings=IntermissionSettings(enabled=True, persist_results=True),
                tts=_SilentTTS(),
            )
            first_db = root / "first" / "events.sqlite3"
            first_json = root / "first" / "results.json"
            runner.begin_session(
                db_path=str(first_db),
                results_json_path=str(first_json),
                session_id="session-a",
            )
            runner.store.upsert_item(
                item_id="phq_1",
                scale="PHQ-2",
                status="ANSWERED",
                score=2,
                response_text="More than half the days",
                reason="",
            )

            runner.begin_session(
                db_path=str(root / "second" / "events.sqlite3"),
                results_json_path=str(root / "second" / "results.json"),
                session_id="session-b",
            )
            self.assertEqual(runner.screening_answers, {})
            self.assertEqual(runner.store.fetch_items(), [])

            runner.begin_session(
                db_path=str(first_db),
                results_json_path=str(first_json),
                session_id="session-a",
                resume=True,
            )
            self.assertEqual(runner.screening_answers["phq_1"], 2)

            runner.end_session()
            self.assertIsNone(runner.store)
            self.assertEqual(runner.screening_answers, {})


class IdleControlTests(unittest.TestCase):
    def test_monitor_fallback_start_and_idle_reset(self):
        monitor = StatusMonitor(StatusMonitorSettings(enabled=False))
        calls = []
        monitor.set_start_session_callback(lambda: calls.append("start") or True)
        monitor.set_phase("ready_idle")

        accepted, _message = monitor.request_session_start()
        self.assertTrue(accepted)
        self.assertEqual(calls, ["start"])

        monitor.set_user(subject_id="026", display_name="Alice")
        monitor.set_session("session-a")
        monitor.set_prompt(text="Question", source="screening", expects_response=True)
        monitor.set_response(text="Answer")
        monitor.set_intermission_state(summary={"PHQ-2": {"total": 2}}, items=[{"item_id": "phq_1"}])
        monitor.reset_for_idle()
        snapshot = monitor.snapshot()

        self.assertEqual(snapshot["phase"], "ready_idle")
        self.assertEqual(snapshot["user"]["subject_id"], "")
        self.assertEqual(snapshot["session"]["id"], "")
        self.assertEqual(snapshot["current_prompt"]["text"], "")
        self.assertEqual(snapshot["latest_response"]["text"], "")
        self.assertEqual(snapshot["intermission"]["items"], [])
        self.assertEqual(snapshot["recent_events"], [])


if __name__ == "__main__":
    unittest.main()
