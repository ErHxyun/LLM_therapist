import unittest

import LLM_therapist_Voice_Application as app


class VoiceApplicationTests(unittest.TestCase):
    def test_main_starts_and_stops_music_around_handler(self):
        events = []

        class FakeMusic:
            def start(self):
                events.append("music.start")

            def stop(self):
                events.append("music.stop")

        class FakeVolumeButtons:
            def start(self):
                events.append("volume.start")

            def stop(self):
                events.append("volume.stop")

        class FakeSessionButton:
            def start(self):
                events.append("session_button.start")
                return True

            def stop(self):
                events.append("session_button.stop")

        class FakeSessionControl:
            settings = type("Settings", (), {"enabled": True})()

            def handle_short_press(self):
                events.append("session.short")

            def handle_long_press(self):
                events.append("session.long")

            def wait_for_start(self):
                events.append("session.wait_start")
                return True

            def request_start(self, source):
                events.append(f"session.request_start:{source}")

            def mark_screening(self):
                events.append("session.mark_screening")

            def mark_closing(self):
                events.append("session.mark_closing")

            def is_shutdown_requested(self):
                events.append("session.is_shutdown_requested")
                return False

        class FakeStatusLEDs:
            def start(self):
                events.append("led.start")

            def stop(self):
                events.append("led.stop")

        class FakeStatusMonitor:
            url = "http://127.0.0.1:8765"

            def start(self):
                events.append("monitor.start")
                return True

            def stop(self):
                events.append("monitor.stop")

            def set_phase(self, phase):
                events.append(f"monitor.phase:{phase}")

        class FakeHandler:
            def __init__(self, *args, **kwargs):
                events.append("handler.init")

            def run(self):
                events.append("handler.run")

        class FakeThread:
            def __init__(self, *args, **kwargs):
                events.append("thread.init")

            def start(self):
                events.append("thread.start")

        originals = {
            "init_record": app.init_record,
            "build_stt": app.build_stt,
            "build_tts": app.build_tts,
            "build_music": app.build_music,
            "build_session_control": app.build_session_control,
            "build_session_button_controller": app.build_session_button_controller,
            "build_status_led_controller": app.build_status_led_controller,
            "build_status_monitor": app.build_status_monitor,
            "build_volume_button_controller": app.build_volume_button_controller,
            "HandlerRL": app.HandlerRL,
            "threading.Thread": app.threading.Thread,
            "threading.Event": app.threading.Event,
            "wait_for_voice_io_drain": app.wait_for_voice_io_drain,
            "time.sleep": app.time.sleep,
            "_preload_llm_runtime": app._preload_llm_runtime,
            "_warm_up_stt": app._warm_up_stt,
        }
        try:
            app.init_record = lambda: events.append("init_record")
            app.build_stt = lambda: "stt"
            app.build_tts = lambda: "tts"
            app.build_music = lambda: FakeMusic()
            app.build_status_monitor = lambda: FakeStatusMonitor()
            app.build_session_control = lambda **_kwargs: FakeSessionControl()
            app.build_session_button_controller = lambda _short, _long: FakeSessionButton()
            app.build_status_led_controller = lambda **_kwargs: FakeStatusLEDs()
            app.build_volume_button_controller = lambda: FakeVolumeButtons()
            app.HandlerRL = FakeHandler
            app._preload_llm_runtime = lambda: events.append("preload_llm_runtime")
            app._warm_up_stt = lambda _stt: events.append("warm_up_stt")
            app.threading.Thread = FakeThread
            app.threading.Event = lambda: type(
                "FakeEvent",
                (),
                {"set": lambda self: events.append("event.set")},
            )()
            app.wait_for_voice_io_drain = lambda *_args, **_kwargs: events.append("drain") or True
            app.time.sleep = lambda _seconds: events.append("sleep")

            app.main()
        finally:
            app.init_record = originals["init_record"]
            app.build_stt = originals["build_stt"]
            app.build_tts = originals["build_tts"]
            app.build_music = originals["build_music"]
            app.build_session_control = originals["build_session_control"]
            app.build_session_button_controller = originals["build_session_button_controller"]
            app.build_status_led_controller = originals["build_status_led_controller"]
            app.build_status_monitor = originals["build_status_monitor"]
            app.build_volume_button_controller = originals["build_volume_button_controller"]
            app.HandlerRL = originals["HandlerRL"]
            app.threading.Thread = originals["threading.Thread"]
            app.threading.Event = originals["threading.Event"]
            app.wait_for_voice_io_drain = originals["wait_for_voice_io_drain"]
            app.time.sleep = originals["time.sleep"]
            app._preload_llm_runtime = originals["_preload_llm_runtime"]
            app._warm_up_stt = originals["_warm_up_stt"]

        self.assertEqual(
            events,
            [
                "init_record",
                "event.set",
                "thread.init",
                "thread.start",
                "monitor.start",
                "led.start",
                "volume.start",
                "monitor.phase:waiting_start",
                "session_button.start",
                "session.wait_start",
                "music.start",
                "preload_llm_runtime",
                "warm_up_stt",
                "session.mark_screening",
                "handler.init",
                "handler.run",
                "session.mark_closing",
                "session.is_shutdown_requested",
                "drain",
                "music.stop",
                "session_button.stop",
                "volume.stop",
                "led.stop",
                "monitor.stop",
                "sleep",
            ],
        )


if __name__ == "__main__":
    unittest.main()
