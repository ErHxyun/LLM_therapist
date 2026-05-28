import unittest
import threading

import LLM_therapist_Voice_Application as app


class VoiceApplicationTests(unittest.TestCase):
    def test_short_press_restarts_music_when_resuming_while_voice_io_idle(self):
        events = []

        class FakeSessionControl:
            def __init__(self):
                self.paused = True

            def is_paused(self):
                events.append(f"is_paused:{self.paused}")
                return self.paused

            def handle_short_press(self):
                events.append("short")
                self.paused = False

        class FakeMusic:
            def start(self):
                events.append("music.start")

            def stop(self):
                events.append("music.stop")

        class FakeEvent:
            def is_set(self):
                events.append("voice_idle")
                return True

        app._handle_short_press_with_music(FakeSessionControl(), FakeMusic(), FakeEvent())

        self.assertEqual(events, ["is_paused:True", "short", "is_paused:False", "voice_idle", "music.start"])

    def test_short_press_does_not_restart_music_when_voice_io_is_not_idle(self):
        events = []

        class FakeSessionControl:
            def __init__(self):
                self.paused = True

            def is_paused(self):
                return self.paused

            def handle_short_press(self):
                events.append("short")
                self.paused = False

        class FakeMusic:
            def start(self):
                events.append("music.start")

            def stop(self):
                events.append("music.stop")

        class FakeEvent:
            def is_set(self):
                events.append("voice_busy")
                return False

        app._handle_short_press_with_music(FakeSessionControl(), FakeMusic(), FakeEvent())

        self.assertEqual(events, ["short", "voice_busy"])

    def test_short_press_restores_music_only_if_it_was_playing_before_pause(self):
        events = []
        restore_music = threading.Event()

        class FakeSessionControl:
            def __init__(self, paused):
                self.paused = paused

            def is_paused(self):
                return self.paused

            def is_shutdown_requested(self):
                return False

            def handle_short_press(self):
                events.append("short")
                self.paused = not self.paused

        class FakeMusic:
            def __init__(self, playing):
                self.playing = playing

            def is_playing(self):
                events.append(f"playing:{self.playing}")
                return self.playing

            def start(self):
                events.append("music.start")

            def stop(self):
                events.append("music.stop")

        class FakeEvent:
            def is_set(self):
                events.append("voice_idle")
                return True

        app._handle_short_press_with_music(
            FakeSessionControl(paused=False),
            FakeMusic(playing=True),
            FakeEvent(),
            restore_music,
        )
        app._handle_short_press_with_music(
            FakeSessionControl(paused=True),
            FakeMusic(playing=False),
            FakeEvent(),
            restore_music,
        )

        self.assertEqual(
            events,
            [
                "playing:True",
                "short",
                "music.stop",
                "playing:False",
                "short",
                "voice_idle",
                "music.start",
            ],
        )

    def test_short_press_pauses_and_resumes_background_music(self):
        events = []
        restore_music = threading.Event()

        class FakeSessionControl:
            def __init__(self, paused):
                self.paused = paused

            def is_paused(self):
                return self.paused

            def is_shutdown_requested(self):
                return False

            def handle_short_press(self):
                events.append("short")
                self.paused = not self.paused

        class FakeMusic:
            def is_playing(self):
                return True

            def pause(self):
                events.append("music.pause")

            def resume(self):
                events.append("music.resume")

            def stop(self):
                events.append("music.stop")

            def start(self):
                events.append("music.start")

        class FakeEvent:
            def is_set(self):
                return True

        music = FakeMusic()
        app._handle_short_press_with_music(
            FakeSessionControl(paused=False),
            music,
            FakeEvent(),
            restore_music,
        )
        app._handle_short_press_with_music(
            FakeSessionControl(paused=True),
            music,
            FakeEvent(),
            restore_music,
        )

        self.assertEqual(events, ["short", "music.pause", "short", "music.resume"])

    def test_shutdown_message_ducks_music_and_uses_tts_led_status(self):
        events = []

        class FakeTTS:
            def speak(self, text):
                events.append(f"tts:{text}")

        class FakeMusic:
            def duck(self):
                events.append("music.duck")

        class FakeStatusLEDs:
            def set_tts_active(self, active):
                events.append(f"blue:{active}")

        app._speak_shutdown_message(FakeTTS(), FakeMusic(), FakeStatusLEDs())

        self.assertEqual(
            events,
            [
                "music.duck",
                "blue:True",
                "tts:Okay, closing Caiti now.",
                "blue:False",
            ],
        )

    def test_shutdown_confirmation_prompt_tells_user_voice_and_button_options(self):
        events = []

        class FakeTTS:
            def speak(self, text):
                events.append(f"tts:{text}")

        class FakeMusic:
            def duck(self):
                events.append("music.duck")

        class FakeStatusLEDs:
            def set_tts_active(self, active):
                events.append(f"blue:{active}")

        app._speak_shutdown_confirmation_prompt(object(), FakeTTS(), FakeMusic(), FakeStatusLEDs())

        self.assertEqual(
            events,
            [
                "music.duck",
                "blue:True",
                "tts:Do you want to close Caiti now? Please say yes to close, say no to keep going, or press and hold the button again for three seconds.",
                "blue:False",
            ],
        )

    def test_shutdown_confirmation_worker_closes_on_voice_yes(self):
        events = []

        class FakeSession:
            settings = type("Settings", (), {"close_cancel_message": "Keep going."})()

            def __init__(self):
                self.shutdown = False

            def begin_shutdown_confirmation(self):
                events.append("begin")
                return True

            def is_shutdown_requested(self):
                return self.shutdown

            def handle_shutdown_confirmation_response(self, response):
                events.append(f"response:{response}")
                self.shutdown = True
                return "shutdown"

        class FakeSTT:
            def listen(self):
                events.append("listen")
                return "yes"

            def set_interrupt_check(self, checker):
                events.append(f"stt.check:{checker is not None}")

        class FakeTTS:
            def speak(self, text):
                events.append(f"tts:{text}")

            def set_interrupt_check(self, checker):
                events.append(f"tts.check:{checker is not None}")

        class FakeMusic:
            def duck(self):
                events.append("music.duck")

            def restore_volume(self):
                events.append("music.restore")

            def stop(self):
                events.append("music.stop")

        class FakeLEDs:
            def set_tts_active(self, active):
                events.append(f"blue:{active}")

            def set_stt_active(self, active):
                events.append(f"green:{active}")

        class FakeEvent:
            def is_set(self):
                return True

        app._run_shutdown_confirmation(
            FakeSession(),
            FakeSTT(),
            FakeTTS(),
            FakeMusic(),
            FakeLEDs(),
            FakeEvent(),
            lambda: events.append("speak_shutdown"),
        )

        self.assertIn("response:yes", events)
        self.assertIn("music.stop", events)
        self.assertIn("speak_shutdown", events)

    def test_shutdown_confirmation_worker_returns_on_voice_no(self):
        events = []

        class FakeSession:
            settings = type("Settings", (), {"close_cancel_message": "Okay, we will keep going."})()

            def begin_shutdown_confirmation(self):
                events.append("begin")
                return True

            def is_shutdown_requested(self):
                return False

            def handle_shutdown_confirmation_response(self, response):
                events.append(f"response:{response}")
                return "cancelled"

        class FakeSTT:
            def listen(self):
                events.append("listen")
                return "no"

            def set_interrupt_check(self, checker):
                return

        class FakeTTS:
            def speak(self, text):
                events.append(f"tts:{text}")

            def set_interrupt_check(self, checker):
                return

        class FakeMusic:
            def duck(self):
                events.append("music.duck")

            def restore_volume(self):
                events.append("music.restore")

            def stop(self):
                events.append("music.stop")

        class FakeLEDs:
            def set_tts_active(self, active):
                return

            def set_stt_active(self, active):
                return

        class FakeEvent:
            def is_set(self):
                return True

        app._run_shutdown_confirmation(
            FakeSession(),
            FakeSTT(),
            FakeTTS(),
            FakeMusic(),
            FakeLEDs(),
            FakeEvent(),
            lambda: events.append("speak_shutdown"),
        )

        self.assertIn("response:no", events)
        self.assertIn("tts:Okay, we will keep going.", events)
        self.assertNotIn("music.stop", events)
        self.assertNotIn("speak_shutdown", events)

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
            paused = False

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

            def checkpoint(self, location):
                events.append(f"session.checkpoint:{location}")
                return "continue"

            def is_paused(self):
                events.append("session.is_paused")
                return self.paused

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

        class FakeEvent:
            def __init__(self):
                self.value = False

            def set(self):
                events.append("event.set")
                self.value = True

            def is_set(self):
                return self.value

        originals = {
            "init_record": app.init_record,
            "build_stt": app.build_stt,
            "build_tts": app.build_tts,
            "build_music": app.build_music,
            "build_intermission_runner": app.build_intermission_runner,
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
            app.build_intermission_runner = lambda **_kwargs: events.append("intermission.build") or "intermission"
            app.build_status_monitor = lambda: FakeStatusMonitor()
            app.build_session_control = lambda **_kwargs: FakeSessionControl()
            app.build_session_button_controller = lambda _short, _long: FakeSessionButton()
            app.build_status_led_controller = lambda **_kwargs: FakeStatusLEDs()
            app.build_volume_button_controller = lambda: FakeVolumeButtons()
            app.HandlerRL = FakeHandler
            app._preload_llm_runtime = lambda: events.append("preload_llm_runtime")
            app._warm_up_stt = lambda _stt: events.append("warm_up_stt")
            app.threading.Thread = FakeThread
            app.threading.Event = FakeEvent
            app.wait_for_voice_io_drain = lambda *_args, **_kwargs: events.append("drain") or True
            app.time.sleep = lambda _seconds: events.append("sleep")

            app.main()
        finally:
            app.init_record = originals["init_record"]
            app.build_stt = originals["build_stt"]
            app.build_tts = originals["build_tts"]
            app.build_music = originals["build_music"]
            app.build_intermission_runner = originals["build_intermission_runner"]
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
                "intermission.build",
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
                "session.checkpoint:loading",
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

    def test_main_speaks_shutdown_message_when_session_shutdown_requested(self):
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

            def __init__(self):
                self.shutdown = False

            def handle_short_press(self):
                return

            def handle_long_press(self):
                return

            def wait_for_start(self):
                events.append("session.wait_start")
                return True

            def mark_screening(self):
                events.append("session.mark_screening")

            def mark_closing(self):
                events.append("session.mark_closing")

            def checkpoint(self, location):
                events.append(f"session.checkpoint:{location}")
                return "continue"

            def is_paused(self):
                return False

            def is_shutdown_requested(self):
                events.append(f"session.is_shutdown_requested:{self.shutdown}")
                return self.shutdown

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

        session = FakeSessionControl()

        class FakeHandler:
            def __init__(self, *args, **kwargs):
                events.append("handler.init")

            def run(self):
                events.append("handler.run")
                session.shutdown = True
                raise app.SessionShutdownRequested("test")

        class FakeThread:
            def __init__(self, *args, **kwargs):
                events.append("thread.init")

            def start(self):
                events.append("thread.start")

        class FakeEvent:
            def __init__(self):
                self.value = False

            def set(self):
                events.append("event.set")
                self.value = True

            def is_set(self):
                return self.value

        originals = {
            "init_record": app.init_record,
            "build_stt": app.build_stt,
            "build_tts": app.build_tts,
            "build_music": app.build_music,
            "build_intermission_runner": app.build_intermission_runner,
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
            "_speak_shutdown_message": app._speak_shutdown_message,
        }
        try:
            app.init_record = lambda: events.append("init_record")
            app.build_stt = lambda: "stt"
            app.build_tts = lambda: "tts"
            app.build_music = lambda: FakeMusic()
            app.build_intermission_runner = lambda **_kwargs: events.append("intermission.build") or "intermission"
            app.build_status_monitor = lambda: FakeStatusMonitor()
            app.build_session_control = lambda **_kwargs: session
            app.build_session_button_controller = lambda _short, _long: FakeSessionButton()
            app.build_status_led_controller = lambda **_kwargs: FakeStatusLEDs()
            app.build_volume_button_controller = lambda: FakeVolumeButtons()
            app.HandlerRL = FakeHandler
            app._preload_llm_runtime = lambda: events.append("preload_llm_runtime")
            app._warm_up_stt = lambda _stt: events.append("warm_up_stt")
            app._speak_shutdown_message = lambda _tts, _music, _status_leds: events.append("speak_shutdown")
            app.threading.Thread = FakeThread
            app.threading.Event = FakeEvent
            app.wait_for_voice_io_drain = lambda *_args, **_kwargs: events.append("drain") or True
            app.time.sleep = lambda _seconds: events.append("sleep")

            app.main()
        finally:
            app.init_record = originals["init_record"]
            app.build_stt = originals["build_stt"]
            app.build_tts = originals["build_tts"]
            app.build_music = originals["build_music"]
            app.build_intermission_runner = originals["build_intermission_runner"]
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
            app._speak_shutdown_message = originals["_speak_shutdown_message"]

        self.assertIn("speak_shutdown", events)
        self.assertNotIn("drain", events)


if __name__ == "__main__":
    unittest.main()
