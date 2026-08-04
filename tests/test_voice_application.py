import unittest
import threading

import LLM_therapist_Voice_Application as app


class VoiceApplicationTests(unittest.TestCase):
    def test_confirm_identity_candidate_accepts_retry_then_yes(self):
        spoken = []
        responses = iter(["maybe", "yes"])
        originals = {
            "_speak_prompt": app._speak_prompt,
            "_listen_with_stt": app._listen_with_stt,
        }
        try:
            app._speak_prompt = lambda *_args, **_kwargs: spoken.append(_args[3])
            app._listen_with_stt = lambda *_args, **_kwargs: next(responses)
            confirmed = app._confirm_identity_candidate(
                stt=object(),
                tts=object(),
                music=object(),
                status_leds=object(),
                candidate="8080",
                confirm_prompt_template=app.USER_ID_CONFIRM_PROMPT,
                retry_prompt=app.USER_ID_CONFIRM_RETRY_PROMPT,
            )
        finally:
            app._speak_prompt = originals["_speak_prompt"]
            app._listen_with_stt = originals["_listen_with_stt"]

        self.assertTrue(confirmed)
        self.assertEqual(
            spoken,
            [
                "I heard your participant ID as 8080. Is that correct? Please say yes or no.",
                "Please say yes if that participant ID is correct, or say no if you want to try again.",
            ],
        )

    def test_collect_confirmed_identity_field_retries_when_confirmation_is_no(self):
        prompts = []
        candidates = iter(["eight zero eight zero", "zero zero one"])
        confirmations = iter([False, True])
        originals = {
            "_collect_identity_field": app._collect_identity_field,
            "_confirm_identity_candidate": app._confirm_identity_candidate,
        }
        try:
            def fake_collect_identity_field(**kwargs):
                prompts.append(kwargs["initial_prompt"])
                return next(candidates)

            def fake_confirm_identity_candidate(**kwargs):
                return next(confirmations)

            app._collect_identity_field = fake_collect_identity_field
            app._confirm_identity_candidate = fake_confirm_identity_candidate

            value = app._collect_confirmed_identity_field(
                stt=object(),
                tts=object(),
                music=object(),
                status_leds=object(),
                initial_prompt=app.USER_ID_PROMPT,
                retry_prompt=app.USER_ID_RETRY_PROMPT,
                confirm_prompt_template=app.USER_ID_CONFIRM_PROMPT,
                confirm_retry_prompt=app.USER_ID_CONFIRM_RETRY_PROMPT,
                reenter_prompt=app.USER_ID_REENTER_PROMPT,
                normalizer=app.normalize_spoken_user_id,
            )
        finally:
            app._collect_identity_field = originals["_collect_identity_field"]
            app._confirm_identity_candidate = originals["_confirm_identity_candidate"]

        self.assertEqual(value, "001")
        self.assertEqual(prompts, [app.USER_ID_PROMPT, app.USER_ID_REENTER_PROMPT])

    def test_wait_for_session_start_persistent_mode_marks_ready_idle_and_waits(self):
        events = []

        class FakeSessionControl:
            def reset_for_next_session(self):
                events.append("session.reset")

            def set_phase(self, phase):
                events.append(f"session.phase:{phase}")

            def wait_for_start(self):
                events.append("session.wait_start")
                return True

        started = app._wait_for_session_start(
            FakeSessionControl(),
            status_monitor=None,
            persistent_loop=True,
        )

        self.assertTrue(started)
        self.assertEqual(
            events,
            [
                "session.phase:ready_idle",
                "session.wait_start",
            ],
        )

    def test_reset_session_runtime_resets_data_and_sets_session_id(self):
        events = []

        class FakeSessionControl:
            def reset_for_next_session(self):
                events.append("session.reset")

        originals = {
            "build_session_id": app.build_session_id,
            "set_session_id": app.set_session_id,
            "reset_record_state": app.reset_record_state,
            "reset_questioner_session_state": app.reset_questioner_session_state,
            "clear_emotion_session_state": app.clear_emotion_session_state,
        }

        try:
            app.build_session_id = lambda subject_id=None: events.append(f"build_session_id:{subject_id}") or "session-123"
            app.set_session_id = lambda session_id: events.append(f"set_session_id:{session_id}") or session_id
            app.reset_record_state = lambda: events.append("reset_record")
            app.reset_questioner_session_state = lambda: events.append("reset_questioner")
            app.clear_emotion_session_state = lambda: events.append("reset_emotion")

            session_id = app._reset_session_runtime(FakeSessionControl(), subject_id="8080")
        finally:
            app.build_session_id = originals["build_session_id"]
            app.set_session_id = originals["set_session_id"]
            app.reset_record_state = originals["reset_record_state"]
            app.reset_questioner_session_state = originals["reset_questioner_session_state"]
            app.clear_emotion_session_state = originals["clear_emotion_session_state"]

        self.assertEqual(session_id, "session-123")
        self.assertEqual(
            events,
            [
                "build_session_id:8080",
                "set_session_id:session-123",
                "reset_record",
                "reset_questioner",
                "reset_emotion",
            ],
        )

    def test_cleanup_after_session_cycle_resets_session_control(self):
        events = []

        class FakeSessionControl:
            def set_phase(self, phase):
                events.append(f"phase:{phase}")

            def reset_for_next_session(self):
                events.append("session.reset")

        class FakeMusic:
            def stop(self):
                events.append("music.stop")

        originals = {
            "wait_for_voice_io_drain": app.wait_for_voice_io_drain,
        }
        try:
            app.wait_for_voice_io_drain = lambda *_args, **_kwargs: events.append("drain") or True
            app._cleanup_after_session_cycle(
                session_control=FakeSessionControl(),
                music=FakeMusic(),
                voice_idle=object(),
            )
        finally:
            app.wait_for_voice_io_drain = originals["wait_for_voice_io_drain"]

        self.assertEqual(events, ["phase:cleanup", "drain", "music.stop", "session.reset"])

    def test_cleanup_after_session_cycle_can_skip_idle_reset_before_poweroff(self):
        events = []

        class FakeSessionControl:
            def set_phase(self, phase):
                events.append(f"phase:{phase}")

            def reset_for_next_session(self):
                events.append("session.reset")

        class FakeMusic:
            def stop(self):
                events.append("music.stop")

        class FakeStatusLeds:
            def reset_for_idle(self):
                events.append("leds.reset")

        class FakeStatusMonitor:
            def reset_for_idle(self):
                events.append("monitor.reset")

        originals = {
            "wait_for_voice_io_drain": app.wait_for_voice_io_drain,
        }
        try:
            app.wait_for_voice_io_drain = lambda *_args, **_kwargs: events.append("drain") or True
            app._cleanup_after_session_cycle(
                session_control=FakeSessionControl(),
                music=FakeMusic(),
                voice_idle=object(),
                status_leds=FakeStatusLeds(),
                status_monitor=FakeStatusMonitor(),
                reset_for_next_session=False,
            )
        finally:
            app.wait_for_voice_io_drain = originals["wait_for_voice_io_drain"]

        self.assertEqual(events, ["phase:cleanup", "drain", "music.stop"])

    def test_auto_poweroff_after_session_complete_reads_config_flag(self):
        original = app.config_loader.SESSION_AUTO_POWEROFF_ON_COMPLETE
        try:
            app.config_loader.SESSION_AUTO_POWEROFF_ON_COMPLETE = True
            self.assertTrue(app._should_auto_poweroff_after_session_complete())
            app.config_loader.SESSION_AUTO_POWEROFF_ON_COMPLETE = False
            self.assertFalse(app._should_auto_poweroff_after_session_complete())
        finally:
            app.config_loader.SESSION_AUTO_POWEROFF_ON_COMPLETE = original

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

    def test_listen_with_stt_pauses_background_music_and_resumes_afterward(self):
        events = []

        class FakeSTT:
            def set_interrupt_check(self, checker):
                events.append(f"interrupt:{checker is not None}")

            def listen(self):
                events.append("stt.listen")
                return "8080"

        class FakeMusic:
            def is_background(self):
                return True

            def is_playing(self):
                return True

            def pause(self):
                events.append("music.pause")

            def resume(self):
                events.append("music.resume")

            def restore_volume(self):
                events.append("music.restore")

        class FakeStatusLEDs:
            def set_stt_active(self, active):
                events.append(f"green:{active}")

        response = app._listen_with_stt(FakeSTT(), FakeMusic(), FakeStatusLEDs())

        self.assertEqual(response, "8080")
        self.assertEqual(
            events,
            [
                "interrupt:False",
                "music.pause",
                "green:True",
                "stt.listen",
                "green:False",
                "music.resume",
                "interrupt:False",
            ],
        )

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

        spoken = app._speak_shutdown_message(FakeTTS(), FakeMusic(), FakeStatusLEDs())

        self.assertTrue(spoken)
        self.assertEqual(
            events,
            [
                "music.duck",
                "blue:True",
                "tts:Caiti is shutting down.",
                "blue:False",
            ],
        )

    def test_shutdown_message_returns_false_when_interrupted(self):
        events = []

        class FakeTTS:
            def speak(self, text):
                events.append(f"tts:{text}")
                raise app.VoiceInterrupted("interrupted")

        class FakeMusic:
            def duck(self):
                events.append("music.duck")

        class FakeStatusLEDs:
            def set_tts_active(self, active):
                events.append(f"blue:{active}")

        spoken = app._speak_shutdown_message(FakeTTS(), FakeMusic(), FakeStatusLEDs())

        self.assertFalse(spoken)
        self.assertEqual(
            events,
            [
                "music.duck",
                "blue:True",
                "tts:Caiti is shutting down.",
                "blue:False",
            ],
        )

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

        class FakeMusicModeButton:
            def start(self):
                events.append("music_mode.start")

            def stop(self):
                events.append("music_mode.stop")

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
            "build_stt": app.build_stt,
            "build_tts": app.build_tts,
            "build_music": app.build_music,
            "_prepare_user_session": app._prepare_user_session,
            "build_intermission_runner": app.build_intermission_runner,
            "build_session_control": app.build_session_control,
            "build_session_button_controller": app.build_session_button_controller,
            "build_status_led_controller": app.build_status_led_controller,
            "build_status_monitor": app.build_status_monitor,
            "build_volume_button_controller": app.build_volume_button_controller,
            "build_music_mode_button_controller": app.build_music_mode_button_controller,
            "HandlerRL": app.HandlerRL,
            "threading.Thread": app.threading.Thread,
            "threading.Event": app.threading.Event,
            "wait_for_voice_io_drain": app.wait_for_voice_io_drain,
            "time.sleep": app.time.sleep,
            "_preload_llm_runtime": app._preload_llm_runtime,
            "_warm_up_tts": app._warm_up_tts,
            "_warm_up_intermission_tts": app._warm_up_intermission_tts,
            "_warm_up_stt": app._warm_up_stt,
            "_request_configured_system_poweroff": app._request_configured_system_poweroff,
        }
        try:
            app.build_stt = lambda: "stt"
            app.build_tts = lambda: "tts"
            app.build_music = lambda: FakeMusic()
            app._prepare_user_session = lambda *_args, **_kwargs: events.append("prepare_user_session") or "session-123"
            app.build_intermission_runner = lambda **_kwargs: events.append("intermission.build") or "intermission"
            app.build_status_monitor = lambda: FakeStatusMonitor()
            app.build_session_control = lambda **_kwargs: FakeSessionControl()
            app.build_session_button_controller = lambda _short, _long: FakeSessionButton()
            app.build_status_led_controller = lambda **_kwargs: FakeStatusLEDs()
            app.build_volume_button_controller = lambda: FakeVolumeButtons()
            app.build_music_mode_button_controller = lambda _press: FakeMusicModeButton()
            app.HandlerRL = FakeHandler
            app._preload_llm_runtime = lambda: events.append("preload_llm_runtime")
            app._warm_up_tts = lambda _tts: events.append("warm_up_tts")
            app._warm_up_intermission_tts = lambda _runner, _tts: events.append("warm_up_intermission_tts")
            app._warm_up_stt = lambda _stt: events.append("warm_up_stt")
            app.threading.Thread = FakeThread
            app.threading.Event = FakeEvent
            app.wait_for_voice_io_drain = lambda *_args, **_kwargs: events.append("drain") or True
            app.time.sleep = lambda _seconds: events.append("sleep")
            app._request_configured_system_poweroff = lambda reason: events.append(f"poweroff:{reason}") or True

            app.main()
        finally:
            app.build_stt = originals["build_stt"]
            app.build_tts = originals["build_tts"]
            app.build_music = originals["build_music"]
            app._prepare_user_session = originals["_prepare_user_session"]
            app.build_intermission_runner = originals["build_intermission_runner"]
            app.build_session_control = originals["build_session_control"]
            app.build_session_button_controller = originals["build_session_button_controller"]
            app.build_status_led_controller = originals["build_status_led_controller"]
            app.build_status_monitor = originals["build_status_monitor"]
            app.build_volume_button_controller = originals["build_volume_button_controller"]
            app.build_music_mode_button_controller = originals["build_music_mode_button_controller"]
            app.HandlerRL = originals["HandlerRL"]
            app.threading.Thread = originals["threading.Thread"]
            app.threading.Event = originals["threading.Event"]
            app.wait_for_voice_io_drain = originals["wait_for_voice_io_drain"]
            app.time.sleep = originals["time.sleep"]
            app._preload_llm_runtime = originals["_preload_llm_runtime"]
            app._warm_up_tts = originals["_warm_up_tts"]
            app._warm_up_intermission_tts = originals["_warm_up_intermission_tts"]
            app._warm_up_stt = originals["_warm_up_stt"]
            app._request_configured_system_poweroff = originals["_request_configured_system_poweroff"]

        self.assertIn("handler.run", events)
        self.assertIn("drain", events)
        self.assertIn("music.start", events)
        self.assertIn("music.stop", events)
        self.assertIn("session.mark_closing", events)
        self.assertIn("poweroff:session complete", events)
        self.assertLess(events.index("music.start"), events.index("handler.run"))
        self.assertLess(events.index("handler.run"), events.index("drain"))
        self.assertLess(events.index("drain"), events.index("poweroff:session complete"))

    def test_main_defers_background_music_autostart_until_voice_io_needs_it(self):
        events = []

        class FakeMusic:
            def is_background(self):
                return True

            def start(self):
                events.append("music.start")

            def stop(self):
                events.append("music.stop")

        class FakeVolumeButtons:
            def start(self):
                return

            def stop(self):
                return

        class FakeMusicModeButton:
            def start(self):
                return

            def stop(self):
                return

        class FakeSessionButton:
            def start(self):
                return True

            def stop(self):
                return

        class FakeSessionControl:
            settings = type("Settings", (), {"enabled": True})()

            def wait_for_start(self):
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
                return False

        class FakeStatusLEDs:
            def start(self):
                return

            def stop(self):
                return

        class FakeStatusMonitor:
            url = "http://127.0.0.1:8765"

            def start(self):
                return False

            def stop(self):
                return

            def set_phase(self, phase):
                events.append(f"monitor.phase:{phase}")

        class FakeHandler:
            def __init__(self, *args, **kwargs):
                return

            def run(self):
                events.append("handler.run")

        class FakeThread:
            def __init__(self, *args, **kwargs):
                return

            def start(self):
                return

        class FakeEvent:
            def __init__(self):
                self.value = False

            def set(self):
                self.value = True

            def is_set(self):
                return self.value

        originals = {
            "build_stt": app.build_stt,
            "build_tts": app.build_tts,
            "build_music": app.build_music,
            "_prepare_user_session": app._prepare_user_session,
            "build_intermission_runner": app.build_intermission_runner,
            "build_session_control": app.build_session_control,
            "build_session_button_controller": app.build_session_button_controller,
            "build_status_led_controller": app.build_status_led_controller,
            "build_status_monitor": app.build_status_monitor,
            "build_volume_button_controller": app.build_volume_button_controller,
            "build_music_mode_button_controller": app.build_music_mode_button_controller,
            "HandlerRL": app.HandlerRL,
            "threading.Thread": app.threading.Thread,
            "threading.Event": app.threading.Event,
            "wait_for_voice_io_drain": app.wait_for_voice_io_drain,
            "time.sleep": app.time.sleep,
            "_preload_llm_runtime": app._preload_llm_runtime,
            "_warm_up_tts": app._warm_up_tts,
            "_warm_up_intermission_tts": app._warm_up_intermission_tts,
            "_warm_up_stt": app._warm_up_stt,
            "_request_configured_system_poweroff": app._request_configured_system_poweroff,
        }
        try:
            app.build_stt = lambda: "stt"
            app.build_tts = lambda: "tts"
            app.build_music = lambda: FakeMusic()
            app._prepare_user_session = lambda *_args, **_kwargs: events.append("prepare_user_session") or "session-123"
            app.build_intermission_runner = lambda **_kwargs: "intermission"
            app.build_status_monitor = lambda: FakeStatusMonitor()
            app.build_session_control = lambda **_kwargs: FakeSessionControl()
            app.build_session_button_controller = lambda _short, _long: FakeSessionButton()
            app.build_status_led_controller = lambda **_kwargs: FakeStatusLEDs()
            app.build_volume_button_controller = lambda: FakeVolumeButtons()
            app.build_music_mode_button_controller = lambda _press: FakeMusicModeButton()
            app.HandlerRL = FakeHandler
            app._preload_llm_runtime = lambda: None
            app._warm_up_tts = lambda _tts: None
            app._warm_up_intermission_tts = lambda _runner, _tts: None
            app._warm_up_stt = lambda _stt: None
            app.threading.Thread = FakeThread
            app.threading.Event = FakeEvent
            app.wait_for_voice_io_drain = lambda *_args, **_kwargs: True
            app.time.sleep = lambda _seconds: None
            app._request_configured_system_poweroff = lambda reason: events.append(f"poweroff:{reason}") or True

            app.main()
        finally:
            app.build_stt = originals["build_stt"]
            app.build_tts = originals["build_tts"]
            app.build_music = originals["build_music"]
            app._prepare_user_session = originals["_prepare_user_session"]
            app.build_intermission_runner = originals["build_intermission_runner"]
            app.build_session_control = originals["build_session_control"]
            app.build_session_button_controller = originals["build_session_button_controller"]
            app.build_status_led_controller = originals["build_status_led_controller"]
            app.build_status_monitor = originals["build_status_monitor"]
            app.build_volume_button_controller = originals["build_volume_button_controller"]
            app.build_music_mode_button_controller = originals["build_music_mode_button_controller"]
            app.HandlerRL = originals["HandlerRL"]
            app.threading.Thread = originals["threading.Thread"]
            app.threading.Event = originals["threading.Event"]
            app.wait_for_voice_io_drain = originals["wait_for_voice_io_drain"]
            app.time.sleep = originals["time.sleep"]
            app._preload_llm_runtime = originals["_preload_llm_runtime"]
            app._warm_up_tts = originals["_warm_up_tts"]
            app._warm_up_intermission_tts = originals["_warm_up_intermission_tts"]
            app._warm_up_stt = originals["_warm_up_stt"]
            app._request_configured_system_poweroff = originals["_request_configured_system_poweroff"]

        self.assertNotIn("music.start", events)
        self.assertIn("handler.run", events)
        self.assertIn("poweroff:session complete", events)

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

        class FakeMusicModeButton:
            def start(self):
                events.append("music_mode.start")

            def stop(self):
                events.append("music_mode.stop")

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
            "build_stt": app.build_stt,
            "build_tts": app.build_tts,
            "build_music": app.build_music,
            "_prepare_user_session": app._prepare_user_session,
            "build_intermission_runner": app.build_intermission_runner,
            "build_session_control": app.build_session_control,
            "build_session_button_controller": app.build_session_button_controller,
            "build_status_led_controller": app.build_status_led_controller,
            "build_status_monitor": app.build_status_monitor,
            "build_volume_button_controller": app.build_volume_button_controller,
            "build_music_mode_button_controller": app.build_music_mode_button_controller,
            "HandlerRL": app.HandlerRL,
            "threading.Thread": app.threading.Thread,
            "threading.Event": app.threading.Event,
            "wait_for_voice_io_drain": app.wait_for_voice_io_drain,
            "time.sleep": app.time.sleep,
            "_preload_llm_runtime": app._preload_llm_runtime,
            "_warm_up_tts": app._warm_up_tts,
            "_warm_up_intermission_tts": app._warm_up_intermission_tts,
            "_warm_up_stt": app._warm_up_stt,
            "_speak_shutdown_message": app._speak_shutdown_message,
            "_request_configured_system_poweroff": app._request_configured_system_poweroff,
        }
        try:
            app.build_stt = lambda: "stt"
            app.build_tts = lambda: "tts"
            app.build_music = lambda: FakeMusic()
            app._prepare_user_session = lambda *_args, **_kwargs: events.append("prepare_user_session") or "session-123"
            app.build_intermission_runner = lambda **_kwargs: events.append("intermission.build") or "intermission"
            app.build_status_monitor = lambda: FakeStatusMonitor()
            app.build_session_control = lambda **_kwargs: session
            app.build_session_button_controller = lambda _short, _long: FakeSessionButton()
            app.build_status_led_controller = lambda **_kwargs: FakeStatusLEDs()
            app.build_volume_button_controller = lambda: FakeVolumeButtons()
            app.build_music_mode_button_controller = lambda _press: FakeMusicModeButton()
            app.HandlerRL = FakeHandler
            app._preload_llm_runtime = lambda: events.append("preload_llm_runtime")
            app._warm_up_tts = lambda _tts: events.append("warm_up_tts")
            app._warm_up_intermission_tts = lambda _runner, _tts: events.append("warm_up_intermission_tts")
            app._warm_up_stt = lambda _stt: events.append("warm_up_stt")
            app._speak_shutdown_message = (
                lambda _tts, _music, _status_leds, **_kwargs: events.append("speak_shutdown") or True
            )
            app.threading.Thread = FakeThread
            app.threading.Event = FakeEvent
            app.wait_for_voice_io_drain = lambda *_args, **_kwargs: events.append("drain") or True
            app.time.sleep = lambda _seconds: events.append("sleep")
            app._request_configured_system_poweroff = lambda reason: events.append(f"poweroff:{reason}") or True

            app.main()
        finally:
            app.build_stt = originals["build_stt"]
            app.build_tts = originals["build_tts"]
            app.build_music = originals["build_music"]
            app._prepare_user_session = originals["_prepare_user_session"]
            app.build_intermission_runner = originals["build_intermission_runner"]
            app.build_session_control = originals["build_session_control"]
            app.build_session_button_controller = originals["build_session_button_controller"]
            app.build_status_led_controller = originals["build_status_led_controller"]
            app.build_status_monitor = originals["build_status_monitor"]
            app.build_volume_button_controller = originals["build_volume_button_controller"]
            app.build_music_mode_button_controller = originals["build_music_mode_button_controller"]
            app.HandlerRL = originals["HandlerRL"]
            app.threading.Thread = originals["threading.Thread"]
            app.threading.Event = originals["threading.Event"]
            app.wait_for_voice_io_drain = originals["wait_for_voice_io_drain"]
            app.time.sleep = originals["time.sleep"]
            app._preload_llm_runtime = originals["_preload_llm_runtime"]
            app._warm_up_tts = originals["_warm_up_tts"]
            app._warm_up_intermission_tts = originals["_warm_up_intermission_tts"]
            app._warm_up_stt = originals["_warm_up_stt"]
            app._speak_shutdown_message = originals["_speak_shutdown_message"]
            app._request_configured_system_poweroff = originals["_request_configured_system_poweroff"]

        self.assertIn("speak_shutdown", events)
        self.assertIn("drain", events)
        self.assertIn("poweroff:button shutdown", events)

    def test_main_converts_user_intake_interrupt_into_button_shutdown(self):
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

        class FakeMusicModeButton:
            def start(self):
                events.append("music_mode.start")

            def stop(self):
                events.append("music_mode.stop")

        class FakeSessionButton:
            def start(self):
                events.append("session_button.start")
                return True

            def stop(self):
                events.append("session_button.stop")

        class FakeSessionControl:
            settings = type("Settings", (), {"enabled": True})()

            def wait_for_start(self):
                events.append("session.wait_start")
                return True

            def mark_closing(self):
                events.append("session.mark_closing")

            def is_shutdown_requested(self):
                events.append("session.is_shutdown_requested")
                return True

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

        session = FakeSessionControl()
        originals = {
            "build_stt": app.build_stt,
            "build_tts": app.build_tts,
            "build_music": app.build_music,
            "_prepare_user_session": app._prepare_user_session,
            "build_intermission_runner": app.build_intermission_runner,
            "build_session_control": app.build_session_control,
            "build_session_button_controller": app.build_session_button_controller,
            "build_status_led_controller": app.build_status_led_controller,
            "build_status_monitor": app.build_status_monitor,
            "build_volume_button_controller": app.build_volume_button_controller,
            "build_music_mode_button_controller": app.build_music_mode_button_controller,
            "threading.Thread": app.threading.Thread,
            "threading.Event": app.threading.Event,
            "wait_for_voice_io_drain": app.wait_for_voice_io_drain,
            "time.sleep": app.time.sleep,
            "_preload_llm_runtime": app._preload_llm_runtime,
            "_warm_up_tts": app._warm_up_tts,
            "_warm_up_intermission_tts": app._warm_up_intermission_tts,
            "_warm_up_stt": app._warm_up_stt,
            "_speak_shutdown_message": app._speak_shutdown_message,
            "_request_configured_system_poweroff": app._request_configured_system_poweroff,
        }
        try:
            app.build_stt = lambda: "stt"
            app.build_tts = lambda: "tts"
            app.build_music = lambda: FakeMusic()
            app._prepare_user_session = (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(app.VoiceInterrupted("shutdown during intake"))
            )
            app.build_intermission_runner = lambda **_kwargs: events.append("intermission.build") or "intermission"
            app.build_status_monitor = lambda: FakeStatusMonitor()
            app.build_session_control = lambda **_kwargs: session
            app.build_session_button_controller = lambda _short, _long: FakeSessionButton()
            app.build_status_led_controller = lambda **_kwargs: FakeStatusLEDs()
            app.build_volume_button_controller = lambda: FakeVolumeButtons()
            app.build_music_mode_button_controller = lambda _press: FakeMusicModeButton()
            app.threading.Thread = FakeThread
            app.threading.Event = FakeEvent
            app.wait_for_voice_io_drain = lambda *_args, **_kwargs: events.append("drain") or True
            app.time.sleep = lambda _seconds: events.append("sleep")
            app._preload_llm_runtime = lambda: events.append("preload_llm_runtime")
            app._warm_up_tts = lambda _tts: events.append("warm_up_tts")
            app._warm_up_intermission_tts = lambda _runner, _tts: events.append("warm_up_intermission_tts")
            app._warm_up_stt = lambda _stt: events.append("warm_up_stt")
            app._speak_shutdown_message = (
                lambda _tts, _music, _status_leds, **_kwargs: events.append("speak_shutdown") or True
            )
            app._request_configured_system_poweroff = lambda reason: events.append(f"poweroff:{reason}") or True

            app.main()
        finally:
            app.build_stt = originals["build_stt"]
            app.build_tts = originals["build_tts"]
            app.build_music = originals["build_music"]
            app._prepare_user_session = originals["_prepare_user_session"]
            app.build_intermission_runner = originals["build_intermission_runner"]
            app.build_session_control = originals["build_session_control"]
            app.build_session_button_controller = originals["build_session_button_controller"]
            app.build_status_led_controller = originals["build_status_led_controller"]
            app.build_status_monitor = originals["build_status_monitor"]
            app.build_volume_button_controller = originals["build_volume_button_controller"]
            app.build_music_mode_button_controller = originals["build_music_mode_button_controller"]
            app.threading.Thread = originals["threading.Thread"]
            app.threading.Event = originals["threading.Event"]
            app.wait_for_voice_io_drain = originals["wait_for_voice_io_drain"]
            app.time.sleep = originals["time.sleep"]
            app._preload_llm_runtime = originals["_preload_llm_runtime"]
            app._warm_up_tts = originals["_warm_up_tts"]
            app._warm_up_intermission_tts = originals["_warm_up_intermission_tts"]
            app._warm_up_stt = originals["_warm_up_stt"]
            app._speak_shutdown_message = originals["_speak_shutdown_message"]
            app._request_configured_system_poweroff = originals["_request_configured_system_poweroff"]

        self.assertIn("speak_shutdown", events)
        self.assertIn("poweroff:button shutdown", events)


if __name__ == "__main__":
    unittest.main()
