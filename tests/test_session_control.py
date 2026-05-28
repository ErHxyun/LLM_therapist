import threading
import time
import unittest

import src.session.control as control
from src.session.control import SessionControl, SessionControlSettings, SessionShutdownRequested


class SessionControlTests(unittest.TestCase):
    def test_short_press_starts_enabled_session(self):
        session = SessionControl(SessionControlSettings(enabled=True))

        session.handle_short_press()

        self.assertTrue(session.wait_for_start(poll_interval_sec=0.01))

    def test_long_press_from_screening_requests_skip_to_cbt(self):
        session = SessionControl(SessionControlSettings(enabled=True))
        session.request_start("test")
        session.mark_screening()

        self.assertEqual(session.handle_long_press(), "skip_to_cbt")

        self.assertTrue(session.should_interrupt_voice())
        self.assertTrue(session.should_interrupt_workflow_wait())
        self.assertTrue(session.should_discard_interrupted_voice_turn())
        self.assertTrue(session.should_keep_music_on_interrupted_voice_turn())
        self.assertEqual(session.checkpoint("screening"), "skip_to_cbt")
        self.assertFalse(session.should_interrupt_workflow_wait())
        self.assertFalse(session.should_keep_music_on_interrupted_voice_turn())

    def test_long_press_during_loading_requests_shutdown_confirmation_not_cbt(self):
        session = SessionControl(SessionControlSettings(enabled=True))
        session.request_start("test")

        self.assertEqual(session.handle_long_press(), "shutdown_confirmation")

        self.assertTrue(session.should_interrupt_voice())
        self.assertFalse(session.should_interrupt_workflow_wait())
        self.assertFalse(session.should_discard_interrupted_voice_turn())
        self.assertTrue(session.begin_shutdown_confirmation())
        self.assertEqual(session.handle_shutdown_confirmation_response("no"), "cancelled")

    def test_session_control_publishes_monitor_phase_and_button_events(self):
        events = []
        monitor = type(
            "FakeMonitor",
            (),
            {
                "set_phase": lambda self, phase: events.append(("phase", phase)),
                "set_button_event": lambda self, event: events.append(("button", event)),
            },
        )()
        session = SessionControl(SessionControlSettings(enabled=True), status_monitor=monitor)

        session.handle_short_press()
        session.mark_screening()
        session.handle_long_press()

        self.assertIn(("phase", "loading"), events)
        self.assertIn(("button", "start:button"), events)
        self.assertIn(("phase", "screening"), events)
        self.assertIn(("button", "skip_to_cbt"), events)

    def test_short_press_immediately_pauses_and_next_short_press_resumes(self):
        events = []
        monitor = type(
            "FakeMonitor",
            (),
            {
                "set_phase": lambda self, phase: events.append(("phase", phase)),
                "set_button_event": lambda self, event: events.append(("button", event)),
            },
        )()
        session = SessionControl(SessionControlSettings(enabled=True), status_monitor=monitor)

        session.request_start("test")
        session.mark_screening()
        session.handle_short_press()

        self.assertTrue(session.is_paused())
        self.assertIn(("phase", "paused"), events)
        self.assertIn(("button", "pause"), events)

        session.handle_short_press()

        self.assertFalse(session.is_paused())
        self.assertIn(("phase", "screening"), events)
        self.assertIn(("button", "resume"), events)

    def test_long_press_in_cbt_asks_voice_confirmation_and_closes_on_yes(self):
        events = []
        originals = {
            "log_question": control.log_question,
            "log_system_message": control.log_system_message,
            "get_resp_log": control.get_resp_log,
        }
        try:
            control.log_question = lambda text: events.append(f"question:{text}")
            control.log_system_message = lambda text: events.append(f"system:{text}")
            control.get_resp_log = lambda should_stop=None: "yes"

            session = SessionControl(SessionControlSettings(enabled=True))
            session.request_start("test")
            session.mark_cbt()
            self.assertEqual(session.handle_long_press(), "shutdown_confirmation")

            self.assertTrue(session.should_interrupt_voice())
            self.assertFalse(session.should_interrupt_workflow_wait())
            self.assertFalse(session.should_discard_interrupted_voice_turn())
            self.assertFalse(session.should_keep_music_on_interrupted_voice_turn())
            with self.assertRaises(SessionShutdownRequested):
                session.checkpoint("cbt")
        finally:
            control.log_question = originals["log_question"]
            control.log_system_message = originals["log_system_message"]
            control.get_resp_log = originals["get_resp_log"]

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].startswith("question:Do you want to close Caiti now?"))

    def test_shutdown_confirmation_no_returns_to_previous_phase(self):
        events = []
        monitor = type(
            "FakeMonitor",
            (),
            {
                "set_phase": lambda self, phase: events.append(("phase", phase)),
                "set_button_event": lambda self, event: events.append(("button", event)),
            },
        )()
        session = SessionControl(SessionControlSettings(enabled=True), status_monitor=monitor)
        session.request_start("test")
        session.mark_cbt()

        self.assertEqual(session.handle_long_press(), "shutdown_confirmation")
        self.assertTrue(session.begin_shutdown_confirmation())
        self.assertEqual(session.handle_shutdown_confirmation_response("no"), "cancelled")

        self.assertFalse(session.is_shutdown_requested())
        self.assertFalse(session.should_interrupt_voice())
        self.assertIn(("phase", "cbt"), events)
        self.assertIn(("button", "shutdown_cancelled"), events)

    def test_checkpoint_waits_for_external_shutdown_confirmation_to_finish(self):
        session = SessionControl(SessionControlSettings(enabled=True))
        session.request_start("test")
        session.mark_cbt()

        self.assertEqual(session.handle_long_press(), "shutdown_confirmation")
        self.assertTrue(session.begin_shutdown_confirmation())

        def cancel_confirmation():
            time.sleep(0.05)
            session.handle_shutdown_confirmation_response("no")

        thread = threading.Thread(target=cancel_confirmation)
        thread.start()
        try:
            self.assertEqual(session.checkpoint("cbt"), "continue")
        finally:
            thread.join(timeout=1.0)

        self.assertFalse(session.is_shutdown_requested())
        self.assertFalse(session.should_interrupt_voice())

    def test_three_long_presses_skip_to_cbt_prompt_shutdown_then_close_without_checkpoint_delay(self):
        events = []
        monitor = type(
            "FakeMonitor",
            (),
            {
                "set_phase": lambda self, phase: events.append(("phase", phase)),
                "set_button_event": lambda self, event: events.append(("button", event)),
            },
        )()
        originals = {
            "log_question": control.log_question,
            "log_system_message": control.log_system_message,
            "get_resp_log": control.get_resp_log,
        }

        try:
            control.log_question = lambda text: events.append(("prompt", text))
            control.log_system_message = lambda text: events.append(("system", text))
            control.get_resp_log = lambda should_stop=None: ""

            session = SessionControl(SessionControlSettings(enabled=True), status_monitor=monitor)

            session.request_start("test")
            session.mark_screening()

            self.assertEqual(session.handle_long_press(), "skip_to_cbt")
            self.assertEqual(session.checkpoint("screening"), "skip_to_cbt")

            session.mark_cbt()
            self.assertEqual(session.handle_long_press(), "shutdown_confirmation")
            self.assertEqual(session.handle_long_press(), "shutdown_confirmed_by_long_press")
            with self.assertRaises(SessionShutdownRequested):
                session.checkpoint("cbt")
        finally:
            control.log_question = originals["log_question"]
            control.log_system_message = originals["log_system_message"]
            control.get_resp_log = originals["get_resp_log"]

        self.assertTrue(session.is_shutdown_requested())
        self.assertIn(("button", "skip_to_cbt"), events)
        self.assertIn(("button", "shutdown_confirmation"), events)
        self.assertIn(("button", "shutdown_confirmed_by_long_press"), events)
        self.assertIn(("phase", "closing"), events)
        self.assertFalse(any(event[0] == "prompt" for event in events))
        self.assertFalse(any(event[0] == "system" for event in events))


if __name__ == "__main__":
    unittest.main()
