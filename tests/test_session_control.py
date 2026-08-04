import unittest

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

    def test_long_press_during_loading_is_ignored(self):
        session = SessionControl(SessionControlSettings(enabled=True))
        session.request_start("test")

        self.assertEqual(session.handle_long_press(), "ignored_busy")
        self.assertFalse(session.should_interrupt_voice())
        self.assertFalse(session.should_interrupt_workflow_wait())
        self.assertFalse(session.should_discard_interrupted_voice_turn())

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

    def test_short_press_during_user_intake_is_ignored_without_delayed_pause(self):
        session = SessionControl(SessionControlSettings(enabled=True))
        session.request_start("test")
        session.set_phase("user_intake")

        self.assertEqual(session.handle_short_press(), "ignored_busy")
        self.assertFalse(session.is_paused())
        self.assertFalse(session.should_interrupt_voice())

    def test_short_press_during_closing_is_ignored(self):
        session = SessionControl(SessionControlSettings(enabled=True))
        session.request_start("test")
        session.mark_closing()

        self.assertEqual(session.handle_short_press(), "ignored_busy")
        self.assertFalse(session.is_paused())

    def test_busy_phase_ignores_short_and_long_presses(self):
        session = SessionControl(SessionControlSettings(enabled=True))
        session.request_start("test")
        session.set_phase("loading")

        self.assertEqual(session.handle_short_press(), "ignored_busy")
        self.assertEqual(session.handle_long_press(), "ignored_busy")
        self.assertFalse(session.is_paused())
        self.assertFalse(session.should_interrupt_voice())

    def test_reset_for_next_session_clears_started_pause_and_skip_state(self):
        session = SessionControl(SessionControlSettings(enabled=True))
        session.request_start("test")
        session.mark_screening()
        session.handle_short_press()
        session.handle_long_press()

        session.reset_for_next_session()

        self.assertFalse(session.is_paused())
        self.assertFalse(session.should_interrupt_voice())
        self.assertFalse(session.is_shutdown_requested())
        self.assertEqual(session.checkpoint("screening"), "continue")
        self.assertEqual(session.handle_short_press(), "start")
        self.assertTrue(session.wait_for_start(poll_interval_sec=0.01))

    def test_long_press_in_cbt_closes_without_voice_confirmation(self):
        session = SessionControl(SessionControlSettings(enabled=True))
        session.request_start("test")
        session.mark_cbt()

        self.assertEqual(session.handle_long_press(), "shutdown_requested_by_long_press")

        self.assertTrue(session.is_shutdown_requested())
        self.assertTrue(session.should_interrupt_voice())
        self.assertTrue(session.should_interrupt_workflow_wait())
        self.assertTrue(session.should_discard_interrupted_voice_turn())
        self.assertFalse(session.should_keep_music_on_interrupted_voice_turn())
        with self.assertRaises(SessionShutdownRequested):
            session.checkpoint("cbt")

    def test_long_press_outside_screening_closes_without_voice_confirmation(self):
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
        session.set_phase("user_intake")

        self.assertEqual(session.handle_long_press(), "shutdown_requested_by_long_press")

        self.assertTrue(session.is_shutdown_requested())
        self.assertTrue(session.should_interrupt_voice())
        self.assertTrue(session.should_interrupt_workflow_wait())
        self.assertTrue(session.should_discard_interrupted_voice_turn())
        self.assertIn(("phase", "user_intake"), events)
        self.assertIn(("phase", "closing"), events)
        self.assertIn(("button", "shutdown_requested_by_long_press"), events)

    def test_two_long_presses_skip_to_cbt_then_close_without_confirmation_prompt(self):
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

        self.assertEqual(session.handle_long_press(), "skip_to_cbt")
        self.assertEqual(session.checkpoint("screening"), "skip_to_cbt")

        session.mark_cbt()
        self.assertEqual(session.handle_long_press(), "shutdown_requested_by_long_press")
        with self.assertRaises(SessionShutdownRequested):
            session.checkpoint("cbt")

        self.assertTrue(session.is_shutdown_requested())
        self.assertIn(("button", "skip_to_cbt"), events)
        self.assertIn(("button", "shutdown_requested_by_long_press"), events)
        self.assertIn(("phase", "closing"), events)


if __name__ == "__main__":
    unittest.main()
