import unittest
import json
import os
import tempfile
from unittest.mock import patch

from src.intermission.runner import IntermissionRunner, IntermissionSettings
from src.intermission.storage import IntermissionScreeningStore
from src.intermission.tasks import SCREENING_ITEMS, classify_screening_response, parse_screening_score


class IntermissionRunnerTests(unittest.TestCase):
    def test_private_screening_answer_stays_in_memory_and_bridges_when_ready(self):
        ready = {"value": False}
        spoken = []
        music_events = []

        class FakeTTS:
            def speak_stream(self, text):
                spoken.append(text)

        class FakeSTT:
            def listen(self):
                ready["value"] = True
                return "nearly every day"

        class FakeMusic:
            def duck(self):
                music_events.append("duck")

            def restore_volume(self):
                music_events.append("restore")

        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "phq_gad_results.json")
            store = IntermissionScreeningStore(
                db_path=os.path.join(tmp, "events.sqlite3"),
                json_path=json_path,
                session_id="intermission-test",
            )
            runner = IntermissionRunner(
                IntermissionSettings(
                    enabled=True,
                    screening_enabled=True,
                    breathing_enabled=False,
                    mindfulness_enabled=False,
                    max_seconds=5,
                    trigger_min_user_speech_sec=0,
                    trigger_min_interval_turns=0,
                    trigger_probability=1.0,
                    bridge_text="Let's go back to the main session.",
                    transition_delay_sec=0,
                ),
                tts=FakeTTS(),
                stt=FakeSTT(),
                music=FakeMusic(),
                store=store,
            )

            runner.run_until_ready(lambda: ready["value"])

            stored_items = store.fetch_items()
            stored_summary = store.fetch_summary()
            with open(json_path, "r", encoding="utf-8") as f:
                stored_json = json.load(f)

        self.assertIn("phq2_1", runner.screening_answers)
        self.assertEqual(runner.screening_answers["phq2_1"], 3)
        self.assertEqual(runner.screening_totals["phq2"]["total"], 3)
        self.assertEqual(runner.screening_totals["phq2"]["answered"], 1)
        self.assertFalse(runner.screening_totals["phq2"]["complete"])
        self.assertEqual(stored_items[0]["item_id"], "phq2_1")
        self.assertEqual(stored_items[0]["scale"], "phq2")
        self.assertEqual(stored_items[0]["status"], "ANSWERED")
        self.assertEqual(stored_items[0]["score"], 3)
        self.assertEqual(stored_items[0]["response_text"], "nearly every day")
        self.assertEqual(stored_summary["phq2"]["total"], 3)
        self.assertEqual(stored_summary["phq2"]["answered"], 1)
        self.assertEqual(
            stored_json["sessions"]["intermission-test"]["items"][0]["item_id"],
            "phq2_1",
        )
        self.assertEqual(
            stored_json["sessions"]["intermission-test"]["summary"]["phq2"]["total"],
            3,
        )
        self.assertEqual(spoken[-1], "Let's go back to the main session.")
        self.assertGreaterEqual(music_events.count("duck"), 2)
        self.assertEqual(music_events.count("duck"), music_events.count("restore"))

    def test_scripted_task_runs_without_stt_or_main_record_dependencies(self):
        ready = {"value": False}
        spoken = []
        led_events = []
        now = {"value": 0.0}

        class FakeTTS:
            def __init__(self):
                self.playback_status_hook = None

            def set_playback_status_hook(self, hook):
                self.playback_status_hook = hook

            def speak_stream(self, text):
                if self.playback_status_hook is not None:
                    self.playback_status_hook(True)
                spoken.append(text)
                if "Continue resting your awareness on the rhythm of breathing." in text:
                    ready["value"] = True
                if self.playback_status_hook is not None:
                    self.playback_status_hook(False)

        class FakeStatusLEDs:
            def set_tts_active(self, active):
                led_events.append(active)

        def fake_monotonic():
            return now["value"]

        def fake_sleep(seconds):
            now["value"] += seconds

        runner = IntermissionRunner(
            IntermissionSettings(
                enabled=True,
                screening_enabled=False,
                breathing_enabled=True,
                mindfulness_enabled=False,
                max_seconds=60,
                poll_interval_sec=0.5,
                trigger_min_user_speech_sec=0,
                trigger_min_interval_turns=0,
                trigger_probability=1.0,
                bridge_text="Back to Caiti.",
                transition_delay_sec=0,
            ),
            tts=FakeTTS(),
            stt=None,
            music=None,
            status_leds=FakeStatusLEDs(),
        )

        with patch("src.intermission.runner.time.monotonic", fake_monotonic), patch(
            "src.intermission.runner.time.sleep",
            fake_sleep,
        ):
            runner.run_until_ready(lambda: ready["value"])

        self.assertGreater(len(spoken), 4)
        self.assertIn("check-in", spoken[0])
        self.assertTrue(any("Simple breath awareness" in text for text in spoken))
        self.assertTrue(any("rhythm of breathing" in text for text in spoken))
        self.assertEqual(spoken[-1], "Back to Caiti.")
        self.assertEqual(led_events, [value for _ in spoken for value in (True, False)])

    def test_background_music_is_stopped_for_intermission_speech_then_resumed(self):
        ready = {"value": False}
        spoken = []
        music_events = []

        class FakeTTS:
            def speak_stream(self, text):
                spoken.append(text)
                if "Continue resting your awareness on the rhythm of breathing." in text:
                    ready["value"] = True

        class FakeMusic:
            def is_background(self):
                return True

            def stop(self):
                music_events.append("stop")

            def start(self):
                music_events.append("start")

            def restore_volume(self):
                music_events.append("restore")

        runner = IntermissionRunner(
            IntermissionSettings(
                enabled=True,
                screening_enabled=False,
                breathing_enabled=True,
                mindfulness_enabled=False,
                max_seconds=5,
                trigger_min_user_speech_sec=0,
                trigger_min_interval_turns=0,
                trigger_probability=1.0,
                bridge_text="Back to Caiti.",
                transition_delay_sec=0,
            ),
            tts=FakeTTS(),
            stt=None,
            music=FakeMusic(),
        )

        with patch("src.intermission.runner.time.sleep", lambda _seconds: None):
            runner.run_until_ready(lambda: ready["value"])

        self.assertGreater(len(spoken), 4)
        expected_events = []
        for _ in spoken:
            expected_events.extend(["stop", "start", "restore"])
        self.assertEqual(music_events, expected_events)

    def test_scripted_breathing_task_pauses_between_guided_steps(self):
        ready = {"value": False}
        spoken = []
        sleeps = []
        now = {"value": 0.0}

        class FakeTTS:
            def speak_stream(self, text):
                spoken.append(text)
                if "Continue resting your awareness on the rhythm of breathing." in text:
                    ready["value"] = True

        def fake_monotonic():
            return now["value"]

        def fake_sleep(seconds):
            sleeps.append(seconds)
            now["value"] += seconds

        runner = IntermissionRunner(
            IntermissionSettings(
                enabled=True,
                screening_enabled=False,
                breathing_enabled=True,
                mindfulness_enabled=False,
                max_seconds=60,
                poll_interval_sec=0.5,
                trigger_min_user_speech_sec=0,
                trigger_min_interval_turns=0,
                trigger_probability=1.0,
                bridge_text="Back to Caiti.",
                transition_delay_sec=0,
            ),
            tts=FakeTTS(),
            stt=None,
            music=None,
        )

        with patch("src.intermission.runner.time.monotonic", fake_monotonic), patch(
            "src.intermission.runner.time.sleep",
            fake_sleep,
        ):
            runner.run_until_ready(lambda: ready["value"])

        self.assertGreater(len(spoken), 4)
        self.assertGreater(sum(sleeps), 10.0)
        self.assertEqual(spoken[-1], "Back to Caiti.")

    def test_screening_items_are_phq2_and_gad4(self):
        phq2 = [item for item in SCREENING_ITEMS if item.scale == "phq2"]
        gad4 = [item for item in SCREENING_ITEMS if item.scale == "gad4"]

        self.assertEqual(len(phq2), 2)
        self.assertEqual(len(gad4), 4)
        self.assertEqual(phq2[0].item_id, "phq2_1")
        self.assertEqual(phq2[-1].item_id, "phq2_2")
        self.assertEqual(gad4[0].item_id, "gad4_1")
        self.assertEqual(gad4[-1].item_id, "gad4_4")

    def test_parse_screening_score_handles_zero_without_treating_it_as_skip(self):
        self.assertEqual(parse_screening_score("not at all"), 0)
        self.assertEqual(parse_screening_score("0"), 0)
        self.assertEqual(parse_screening_score("several days"), 1)
        self.assertEqual(parse_screening_score("more than half the days"), 2)
        self.assertEqual(parse_screening_score("nearly every day"), 3)
        self.assertIsNone(parse_screening_score("skip this one"))

    def test_classify_screening_response_distinguishes_skip_and_unresolved(self):
        self.assertEqual(classify_screening_response("not at all").status, "ANSWERED")
        self.assertEqual(classify_screening_response("not at all").score, 0)
        self.assertEqual(classify_screening_response("skip this one").status, "SKIPPED")
        self.assertEqual(classify_screening_response("").status, "UNRESOLVED")
        self.assertEqual(classify_screening_response("I am not sure").status, "UNRESOLVED")

    def test_intermission_skips_short_user_speech(self):
        spoken = []

        class FakeTTS:
            def speak_stream(self, text):
                spoken.append(text)

        class FakeSTT:
            def listen(self):
                return "nearly every day"

        runner = IntermissionRunner(
            IntermissionSettings(
                enabled=True,
                screening_enabled=True,
                breathing_enabled=False,
                mindfulness_enabled=False,
                trigger_min_user_speech_sec=10,
            ),
            tts=FakeTTS(),
            stt=FakeSTT(),
        )

        runner.run_until_ready(lambda: False, user_speech_duration_sec=4)

        self.assertEqual(spoken, [])
        self.assertEqual(runner.screening_answers, {})

    def test_intermission_cools_down_after_long_user_speech_trigger(self):
        spoken = []
        responses = iter(["nearly every day", "several days"])

        class FakeTTS:
            def speak_stream(self, text):
                spoken.append(text)

        class FakeSTT:
            def listen(self):
                ready["value"] = True
                return next(responses)

        ready = {"value": False}
        runner = IntermissionRunner(
            IntermissionSettings(
                enabled=True,
                screening_enabled=True,
                breathing_enabled=False,
                mindfulness_enabled=False,
                max_seconds=5,
                trigger_min_user_speech_sec=10,
                trigger_min_interval_turns=0,
                trigger_probability=1.0,
                cooldown_turns=1,
                bridge_text="Let's go back to the main session.",
                transition_delay_sec=0,
            ),
            tts=FakeTTS(),
            stt=FakeSTT(),
        )

        runner.run_until_ready(lambda: ready["value"], user_speech_duration_sec=12)
        ready["value"] = False
        runner.run_until_ready(lambda: ready["value"], user_speech_duration_sec=12)
        ready["value"] = False
        runner.run_until_ready(lambda: ready["value"], user_speech_duration_sec=12)

        self.assertEqual(runner.screening_answers["phq2_1"], 3)
        self.assertEqual(runner.screening_answers["phq2_2"], 1)
        self.assertEqual(len([text for text in spoken if "check-in" in text.lower()]), 2)

    def test_intermission_requires_interval_counter_before_random_gate(self):
        spoken = []

        class FakeTTS:
            def speak_stream(self, text):
                spoken.append(text)

        class FakeSTT:
            def listen(self):
                ready["value"] = True
                return "nearly every day"

        ready = {"value": False}
        runner = IntermissionRunner(
            IntermissionSettings(
                enabled=True,
                screening_enabled=True,
                breathing_enabled=False,
                mindfulness_enabled=False,
                max_seconds=5,
                trigger_min_user_speech_sec=10,
                trigger_min_interval_turns=2,
                trigger_probability=1.0,
                cooldown_turns=0,
                bridge_text="Let's go back to the main session.",
                transition_delay_sec=0,
            ),
            tts=FakeTTS(),
            stt=FakeSTT(),
        )

        runner.run_until_ready(lambda: ready["value"], user_speech_duration_sec=12)
        ready["value"] = False
        runner.run_until_ready(lambda: ready["value"], user_speech_duration_sec=12)
        ready["value"] = False
        runner.run_until_ready(lambda: ready["value"], user_speech_duration_sec=12)

        self.assertEqual(runner.screening_answers["phq2_1"], 3)
        self.assertEqual(len([text for text in spoken if "check-in" in text.lower()]), 1)

    def test_intermission_random_gate_can_skip_after_interval(self):
        spoken = []

        class FakeTTS:
            def speak_stream(self, text):
                spoken.append(text)

        class FakeSTT:
            def listen(self):
                ready["value"] = True
                return "several days"

        ready = {"value": False}
        runner = IntermissionRunner(
            IntermissionSettings(
                enabled=True,
                screening_enabled=True,
                breathing_enabled=False,
                mindfulness_enabled=False,
                max_seconds=5,
                trigger_min_user_speech_sec=10,
                trigger_min_interval_turns=0,
                trigger_probability=0.5,
                cooldown_turns=0,
                bridge_text="Let's go back to the main session.",
                transition_delay_sec=0,
            ),
            tts=FakeTTS(),
            stt=FakeSTT(),
        )

        with patch("src.intermission.runner.random.random", return_value=0.75):
            runner.run_until_ready(lambda: ready["value"], user_speech_duration_sec=12)
        self.assertEqual(spoken, [])

        with patch("src.intermission.runner.random.random", return_value=0.25):
            runner.run_until_ready(lambda: ready["value"], user_speech_duration_sec=12)

        self.assertEqual(runner.screening_answers["phq2_1"], 1)
        self.assertEqual(len([text for text in spoken if "check-in" in text.lower()]), 1)

    def test_intermission_waits_before_returning_to_main_questions(self):
        ready = {"value": False}
        spoken = []
        sleeps = []
        now = {"value": 0.0}

        class FakeTTS:
            def speak_stream(self, text):
                spoken.append(text)
                if "Simple breath awareness" in text:
                    ready["value"] = True

        def fake_monotonic():
            return now["value"]

        def fake_sleep(seconds):
            sleeps.append(seconds)
            now["value"] += seconds

        runner = IntermissionRunner(
            IntermissionSettings(
                enabled=True,
                screening_enabled=False,
                breathing_enabled=True,
                mindfulness_enabled=False,
                max_seconds=5,
                poll_interval_sec=0.5,
                trigger_min_user_speech_sec=0,
                trigger_min_interval_turns=0,
                trigger_probability=1.0,
                bridge_text="Back to Caiti.",
                transition_delay_sec=2.0,
            ),
            tts=FakeTTS(),
            stt=None,
            music=None,
        )

        with patch("src.intermission.runner.time.monotonic", fake_monotonic), patch(
            "src.intermission.runner.time.sleep",
            fake_sleep,
        ):
            runner.run_until_ready(lambda: ready["value"])

        self.assertEqual(spoken[-1], "Back to Caiti.")
        self.assertAlmostEqual(sum(sleeps), 2.0)


if __name__ == "__main__":
    unittest.main()
