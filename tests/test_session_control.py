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

        session.handle_long_press()

        self.assertEqual(session.checkpoint("screening"), "skip_to_cbt")

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
            session.handle_long_press()

            with self.assertRaises(SessionShutdownRequested):
                session.checkpoint("cbt")
        finally:
            control.log_question = originals["log_question"]
            control.log_system_message = originals["log_system_message"]
            control.get_resp_log = originals["get_resp_log"]

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].startswith("question:Do you want to close Caiti now?"))


if __name__ == "__main__":
    unittest.main()
