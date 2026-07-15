import json
import tempfile
import unittest
from pathlib import Path

from src.runtime import status_monitor as sm
from src.runtime.status_monitor import StatusMonitor, StatusMonitorSettings


class StatusMonitorTests(unittest.TestCase):
    def test_snapshot_tracks_phase_lights_and_button(self):
        monitor = StatusMonitor(StatusMonitorSettings(enabled=False))

        monitor.set_phase("screening")
        monitor.set_light("blue", True)
        monitor.set_button_event("skip_to_cbt")
        monitor.set_user(subject_id="001", display_name="Alice")
        monitor.set_session("session-1")
        monitor.set_prompt(text="How have you been sleeping?", source="screening", expects_response=True)
        monitor.set_response(text="Not great", source="record")
        monitor.set_score(
            item_id="10",
            question_index="1",
            dimension="sleep",
            score=2,
            user_input="Not great",
            classification=[["sleep", 2]],
            followup_text="Can you tell me more about that?",
        )
        snapshot = monitor.snapshot()

        self.assertEqual(snapshot["phase"], "screening")
        self.assertTrue(snapshot["lights"]["blue"])
        self.assertEqual(snapshot["button"]["last_event"], "skip_to_cbt")
        self.assertEqual(snapshot["user"]["subject_id"], "001")
        self.assertEqual(snapshot["session"]["id"], "session-1")
        self.assertEqual(snapshot["current_prompt"]["source"], "screening")
        self.assertEqual(snapshot["latest_response"]["text"], "Not great")
        self.assertEqual(snapshot["latest_score"]["dimension"], "sleep")
        self.assertTrue(snapshot["recent_events"])
        self.assertEqual(snapshot["recent_events"][-1]["kind"], "score")
        self.assertGreater(snapshot["version"], 0)

    def test_snapshot_is_json_serializable(self):
        monitor = StatusMonitor(StatusMonitorSettings(enabled=True, host="127.0.0.1", port=0))
        monitor.set_phase("cbt")
        monitor.set_light("green", True)
        data = json.loads(json.dumps(monitor.snapshot()))

        self.assertEqual(data["phase"], "cbt")
        self.assertTrue(data["lights"]["green"])

    def test_history_snapshot_groups_user_profiles_and_sessions(self):
        original_root = sm._data_root_dir
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                user_dir = root / "users" / "001"
                (user_dir / "results").mkdir(parents=True)
                (user_dir / "emotion").mkdir(parents=True)
                (user_dir / "intermission").mkdir(parents=True)
                (user_dir / "profile.json").write_text(
                    json.dumps(
                        {
                            "subject_id": "001",
                            "raw_subject_id": "001",
                            "display_name": "Alice",
                            "updated_at": "2026-06-20T00:00:00",
                            "created_at": "2026-06-19T00:00:00",
                        }
                    ),
                    encoding="utf-8",
                )
                (user_dir / "results" / "SessionSummary_001_1.json").write_text(
                    json.dumps(
                        {
                            "run_id": "1",
                            "subject_id": "001",
                            "timestamp": "2026-06-20T00:01:00",
                            "screening_turn_count": 3,
                            "cbt_used": True,
                            "cbt_candidates": [{"dimension": "sleep"}],
                        }
                    ),
                    encoding="utf-8",
                )
                sm._data_root_dir = lambda: root
                history = sm.build_history_snapshot()
        finally:
            sm._data_root_dir = original_root

        self.assertEqual(len(history["users"]), 1)
        self.assertEqual(history["users"][0]["subject_id"], "001")
        self.assertEqual(history["users"][0]["display_name"], "Alice")
        self.assertEqual(history["users"][0]["sessions"][0]["run_id"], "1")

    def test_history_snapshot_is_json_serializable(self):
        original_history = sm.build_history_snapshot
        try:
            sm.build_history_snapshot = lambda: {"generated_at": "now", "users": [], "legacy_users": []}
            data = json.loads(json.dumps(sm.build_history_snapshot()))
        finally:
            sm.build_history_snapshot = original_history

        self.assertEqual(data["generated_at"], "now")


if __name__ == "__main__":
    unittest.main()
