import unittest
import tempfile
import sys
from pathlib import Path

import pandas as pd

from src.voice.backends import CommandSTT, CommandTTS, FasterWhisperSTT
import src.voice.io_loop as voice_io
from src.voice.sentence_stream import split_for_tts
from src.utils.io_record import HEADER


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class VoiceIOTests(unittest.TestCase):
    def test_split_for_tts_uses_sentence_boundaries(self):
        text = "First sentence. Second question? Final line\nAnother line"
        self.assertEqual(
            split_for_tts(text),
            ["First sentence.", "Second question?", "Final line", "Another line"],
        )

    def test_clean_spoken_text_removes_internal_labels(self):
        self.assertEqual(
            voice_io.clean_spoken_text("Guide: Please return to the question."),
            "Please return to the question.",
        )
        self.assertEqual(
            voice_io.clean_spoken_text("VALIDATION: I hear you.\n\nHow has your mood been?"),
            "I hear you.\n\nHow has your mood been?",
        )

    def test_parse_voice_prompt_marks_system_message_no_response(self):
        text, expects_response = voice_io.parse_voice_prompt("__CAITI_NO_RESPONSE__\nGoodbye.")
        self.assertEqual(text, "Goodbye.")
        self.assertFalse(expects_response)

    def test_command_stt_returns_stdout_transcript(self):
        calls = []

        def runner(*args, **kwargs):
            calls.append((args, kwargs))
            return _Completed(stdout=" My weight increased recently. \n")

        stt = CommandSTT("local-stt", runner=runner)
        self.assertEqual(stt.listen(), "My weight increased recently.")
        self.assertEqual(calls[0][0][0], "local-stt")
        self.assertTrue(calls[0][1]["shell"])

    def test_command_tts_streams_complete_text_block(self):
        spoken = []

        def runner(*args, **kwargs):
            spoken.append(kwargs.get("input"))
            return _Completed()

        tts = CommandTTS("local-tts", runner=runner)
        tts.speak_stream("One. Two?")
        self.assertEqual(spoken, ["One. Two?"])

    def test_command_tts_uses_playback_markers_for_status_hook(self):
        events = []
        command = (
            f"{sys.executable} -c "
            "\"import os, sys; "
            "sys.stdin.read(); "
            "print('__CAITI_TTS_PLAYBACK_START__', file=sys.stderr, flush=True); "
            "print('__CAITI_TTS_PLAYBACK_END__', file=sys.stderr, flush=True)\""
        )

        tts = CommandTTS(command, timeout_sec=5)
        tts.set_playback_status_hook(lambda active: events.append(active))
        tts.speak("Hello.")

        self.assertEqual(events, [True, False])

    def test_command_backend_raises_on_failure(self):
        def runner(*args, **kwargs):
            return _Completed(returncode=2, stderr="boom")

        with self.assertRaises(RuntimeError):
            CommandSTT("bad-stt", runner=runner).listen()
        with self.assertRaises(RuntimeError):
            CommandTTS("bad-tts", runner=runner).speak("hello")

    def test_faster_whisper_backend_records_once_and_reuses_model(self):
        calls = []
        from src.voice import backends

        original_record = backends.whisper_stt.record_wav_auto_stop
        original_load = backends.whisper_stt.load_whisper_model
        original_transcribe = backends.whisper_stt.transcribe_wav_with_model
        try:
            backends.whisper_stt.record_wav_auto_stop = lambda wav_file, **kwargs: (
                calls.append(("record", kwargs)),
                Path(wav_file).write_bytes(b"RIFF" + b"0" * 80),
            )
            backends.whisper_stt.load_whisper_model = lambda *args, **kwargs: calls.append(("load", args)) or object()
            backends.whisper_stt.transcribe_wav_with_model = (
                lambda model, wav_file, **kwargs: calls.append(("transcribe", kwargs)) or "I see my doctor."
            )

            stt = FasterWhisperSTT(model="base.en", audio_device="plughw:0,0")
            self.assertEqual(stt.listen(), "I see my doctor.")
            self.assertEqual(stt.listen(), "I see my doctor.")
        finally:
            backends.whisper_stt.record_wav_auto_stop = original_record
            backends.whisper_stt.load_whisper_model = original_load
            backends.whisper_stt.transcribe_wav_with_model = original_transcribe

        self.assertEqual([name for name, _ in calls].count("load"), 1)
        self.assertEqual([name for name, _ in calls].count("record"), 2)

    def test_faster_whisper_can_start_music_between_recording_and_transcribe(self):
        events = []
        from src.voice import backends

        original_record = backends.whisper_stt.record_wav_auto_stop
        original_load = backends.whisper_stt.load_whisper_model
        original_transcribe = backends.whisper_stt.transcribe_wav_with_model

        class FakeMusic:
            def start(self):
                events.append("music.start")

            def stop(self):
                events.append("music.stop")

        try:
            backends.whisper_stt.record_wav_auto_stop = lambda wav_file, **kwargs: (
                events.append("record"),
                Path(wav_file).write_bytes(b"RIFF" + b"0" * 80),
            )
            backends.whisper_stt.load_whisper_model = lambda *args, **kwargs: events.append("load") or object()
            backends.whisper_stt.transcribe_wav_with_model = (
                lambda model, wav_file, **kwargs: events.append("transcribe") or "I feel okay."
            )

            stt = FasterWhisperSTT(model="base.en", audio_device="plughw:0,0")
            self.assertEqual(stt.listen_with_waiting_music(FakeMusic()), "I feel okay.")
        finally:
            backends.whisper_stt.record_wav_auto_stop = original_record
            backends.whisper_stt.load_whisper_model = original_load
            backends.whisper_stt.transcribe_wav_with_model = original_transcribe

        self.assertEqual(events, ["record", "music.start", "load", "transcribe"])

    def test_process_voice_turn_bridges_question_to_transcript(self):
        class FakeSTT:
            def listen(self):
                return "My weight increased recently."

        class FakeTTS:
            def __init__(self):
                self.spoken = []

            def speak(self, text):
                self.spoken.append(text)

            def speak_stream(self, text):
                self.spoken.append(text)

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["Guide: First sentence. Second question?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                tts = FakeTTS()
                processed = voice_io.process_voice_turn(FakeSTT(), tts)
                result = pd.read_csv(record_path)
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertTrue(processed)
        self.assertEqual(tts.spoken, ["First sentence. Second question?"])
        self.assertEqual(result.loc[0, "Resp"], "My weight increased recently.")
        self.assertEqual(int(result.loc[0, "Question_Lock"]), 0)
        self.assertEqual(int(result.loc[0, "Resp_Lock"]), 0)

    def test_process_voice_turn_stops_and_restarts_waiting_music(self):
        events = []

        class FakeSTT:
            def listen(self):
                events.append("listen")
                return "No current issue."

        class FakeTTS:
            def speak(self, text):
                events.append(f"speak:{text}")

            def speak_stream(self, text):
                events.append(f"speak:{text}")

        class FakeMusic:
            def stop(self):
                events.append("music.stop")

            def start(self):
                events.append("music.start")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["How has your mood been?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            original_settle_sec = voice_io.MUSIC_STOP_SETTLE_SEC
            voice_io.RECORD_CSV = record_path
            voice_io.MUSIC_STOP_SETTLE_SEC = 0
            try:
                processed = voice_io.process_voice_turn(FakeSTT(), FakeTTS(), music=FakeMusic())
            finally:
                voice_io.RECORD_CSV = original_record_csv
                voice_io.MUSIC_STOP_SETTLE_SEC = original_settle_sec

        self.assertTrue(processed)
        self.assertEqual(
            events,
            [
                "music.stop",
                "speak:How has your mood been?",
                "listen",
                "music.start",
            ],
        )

    def test_process_voice_turn_runs_intermission_after_user_answer_until_next_question_ready(self):
        events = []

        class FakeSTT:
            last_audio_duration_sec = 12.5

            def listen(self):
                events.append("listen")
                return "No current issue."

        class FakeTTS:
            def speak(self, text):
                events.append(f"speak:{text}")

            def speak_stream(self, text):
                events.append(f"speak:{text}")

        class FakeIntermission:
            def __init__(self, record_path):
                self.record_path = record_path

            def run_until_ready(self, is_ready, should_stop=None, user_speech_duration_sec=None):
                events.append(f"intermission.duration:{user_speech_duration_sec}")
                events.append(f"intermission.ready_before:{is_ready()}")
                df = pd.read_csv(self.record_path)
                df.loc[0, "Question"] = "Next main question?"
                df.loc[0, "Question_Lock"] = 1
                df.to_csv(self.record_path, columns=HEADER, index=False)
                events.append(f"intermission.ready_after:{is_ready()}")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["How has your mood been?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(
                    FakeSTT(),
                    FakeTTS(),
                    intermission_runner=FakeIntermission(record_path),
                )
                result = pd.read_csv(record_path)
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertTrue(processed)
        self.assertEqual(
            events,
            [
                "speak:How has your mood been?",
                "listen",
                "intermission.duration:12.5",
                "intermission.ready_before:False",
                "intermission.ready_after:True",
            ],
        )
        self.assertEqual(result.loc[0, "Question"], "Next main question?")
        self.assertEqual(int(result.loc[0, "Question_Lock"]), 1)

    def test_process_voice_turn_skips_intermission_for_empty_transcript(self):
        events = []

        class FakeSTT:
            last_audio_duration_sec = 20.0

            def listen(self):
                events.append("listen")
                return ""

        class FakeTTS:
            def speak(self, text):
                events.append(f"speak:{text}")

            def speak_stream(self, text):
                events.append(f"speak:{text}")

        class FakeIntermission:
            def run_until_ready(self, is_ready, should_stop=None, user_speech_duration_sec=None):
                raise AssertionError("Empty main transcripts should not start intermission")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["How has your mood been?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(
                    FakeSTT(),
                    FakeTTS(),
                    empty_transcript_retries=0,
                    intermission_runner=FakeIntermission(),
                )
                result = pd.read_csv(record_path)
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertTrue(processed)
        self.assertEqual(events, ["speak:How has your mood been?", "listen"])
        self.assertTrue(pd.isna(result.loc[0, "Resp"]) or result.loc[0, "Resp"] == "")
        self.assertEqual(int(result.loc[0, "Resp_Lock"]), 0)

    def test_process_voice_turn_ducks_background_music_for_tts_and_stt(self):
        events = []

        class FakeSTT:
            def listen(self):
                events.append("listen")
                return "No current issue."

        class FakeTTS:
            def speak(self, text):
                events.append(f"speak:{text}")

            def speak_stream(self, text):
                events.append(f"speak:{text}")

        class FakeMusic:
            def is_background(self):
                return True

            def start(self):
                events.append("music.start")

            def duck(self):
                events.append("music.duck")

            def restore_volume(self):
                events.append("music.restore")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["How has your mood been?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(FakeSTT(), FakeTTS(), music=FakeMusic())
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertTrue(processed)
        self.assertEqual(
            events,
            [
                "music.start",
                "music.duck",
                "speak:How has your mood been?",
                "music.duck",
                "listen",
                "music.restore",
                "music.start",
            ],
        )

    def test_process_voice_turn_updates_status_leds(self):
        events = []

        class FakeSTT:
            def listen(self):
                events.append("listen")
                return "I sleep okay."

        class FakeTTS:
            def speak(self, text):
                events.append(f"speak:{text}")

            def speak_stream(self, text):
                raise AssertionError("status-led path should speak text through speak() directly")

        class FakeStatusLEDs:
            def set_tts_active(self, active):
                events.append(f"blue:{active}")

            def set_stt_active(self, active):
                events.append(f"green:{active}")

            def mark_session_started(self):
                events.append("yellow:on")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["First sentence. Second question?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(
                    FakeSTT(),
                    FakeTTS(),
                    status_leds=FakeStatusLEDs(),
                )
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertTrue(processed)
        self.assertEqual(
            events,
            [
                "yellow:on",
                "blue:True",
                "speak:First sentence. Second question?",
                "blue:False",
                "green:True",
                "listen",
                "green:False",
            ],
        )

    def test_process_voice_turn_uses_playback_hook_for_status_led_timing(self):
        events = []

        class FakeSTT:
            def listen(self):
                events.append("listen")
                return "I sleep okay."

        class FakeTTS:
            def __init__(self):
                self.hook = None

            def set_playback_status_hook(self, hook):
                self.hook = hook

            def speak(self, text):
                events.append(f"prepare:{text}")
                self.hook(True)
                events.append(f"speak:{text}")
                self.hook(False)

            def speak_stream(self, text):
                raise AssertionError("status-led path should speak text through speak() directly")

        class FakeStatusLEDs:
            def set_tts_active(self, active):
                events.append(f"blue:{active}")

            def set_stt_active(self, active):
                events.append(f"green:{active}")

            def mark_session_started(self):
                events.append("yellow:on")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["First sentence. Second question?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(
                    FakeSTT(),
                    FakeTTS(),
                    status_leds=FakeStatusLEDs(),
                )
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertTrue(processed)
        self.assertEqual(
            events,
            [
                "prepare:First sentence. Second question?",
                "yellow:on",
                "blue:True",
                "speak:First sentence. Second question?",
                "blue:False",
                "green:True",
                "listen",
                "green:False",
            ],
        )

    def test_process_voice_turn_interruption_restores_question_for_replay(self):
        events = []

        class FakeSessionControl:
            def __init__(self):
                self.paused = False
                self.waits = 0

            def is_paused(self):
                return self.paused

            def is_shutdown_requested(self):
                return False

            def wait_while_paused(self, poll_interval_sec=0.05):
                events.append("wait_paused")
                self.waits += 1
                self.paused = False

        class FakeSTT:
            def set_interrupt_check(self, checker):
                events.append(f"stt.check:{checker is not None}")

            def listen(self):
                raise AssertionError("Interrupted TTS should not reach STT")

        class FakeTTS:
            def __init__(self, session):
                self.session = session

            def set_interrupt_check(self, checker):
                events.append(f"tts.check:{checker is not None}")

            def set_playback_status_hook(self, hook):
                self.hook = hook

            def speak(self, text):
                events.append(f"speak:{text}")
                self.session.paused = True
                raise voice_io.VoiceInterrupted("pause")

            def speak_stream(self, text):
                raise AssertionError("status-led path should speak text through speak() directly")

        class FakeStatusLEDs:
            def set_tts_active(self, active):
                events.append(f"blue:{active}")

            def set_stt_active(self, active):
                events.append(f"green:{active}")

            def mark_session_started(self):
                events.append("yellow:on")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            question = "Please answer this question."
            pd.DataFrame(
                [[question, 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            session = FakeSessionControl()
            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(
                    FakeSTT(),
                    FakeTTS(session),
                    status_leds=FakeStatusLEDs(),
                    session_control=session,
                )
                result = pd.read_csv(record_path)
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertFalse(processed)
        self.assertEqual(result.loc[0, "Question"], question)
        self.assertEqual(int(result.loc[0, "Question_Lock"]), 1)
        self.assertEqual(int(result.loc[0, "Resp_Lock"]), 1)
        self.assertIn("wait_paused", events)
        self.assertEqual(events[-2:], ["tts.check:False", "stt.check:False"])

    def test_process_voice_turn_discards_question_for_workflow_override(self):
        events = []

        class FakeSessionControl:
            def __init__(self):
                self.checks = 0

            def should_interrupt_voice(self):
                self.checks += 1
                return self.checks > 1

            def should_discard_interrupted_voice_turn(self):
                return True

            def is_paused(self):
                return False

            def is_shutdown_requested(self):
                return False

        class FakeSTT:
            def set_interrupt_check(self, checker):
                events.append(f"stt.check:{checker is not None}")

            def listen(self):
                raise AssertionError("Interrupted TTS should not reach STT")

        class FakeTTS:
            def set_interrupt_check(self, checker):
                events.append(f"tts.check:{checker is not None}")

            def speak_stream(self, text):
                raise AssertionError("Interrupted voice turn should not speak")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["Current question?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(
                    FakeSTT(),
                    FakeTTS(),
                    session_control=FakeSessionControl(),
                )
                result = pd.read_csv(record_path)
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertFalse(processed)
        self.assertTrue(pd.isna(result.loc[0, "Question"]) or result.loc[0, "Question"] == "")
        self.assertEqual(int(result.loc[0, "Question_Lock"]), 0)
        self.assertEqual(int(result.loc[0, "Resp_Lock"]), 1)
        self.assertEqual(events[-2:], ["tts.check:False", "stt.check:False"])

    def test_process_voice_turn_keeps_background_music_during_skip_to_cbt_override(self):
        events = []

        class FakeSessionControl:
            def __init__(self):
                self.checks = 0

            def should_interrupt_voice(self):
                self.checks += 1
                return self.checks > 1

            def should_discard_interrupted_voice_turn(self):
                return True

            def should_keep_music_on_interrupted_voice_turn(self):
                return True

            def is_paused(self):
                return False

            def is_shutdown_requested(self):
                return False

        class FakeMusic:
            def is_background(self):
                return True

            def start(self):
                events.append("music.start")

            def duck(self):
                events.append("music.duck")

            def restore_volume(self):
                events.append("music.restore")

            def pause(self):
                events.append("music.pause")

            def stop(self):
                events.append("music.stop")

        class FakeSTT:
            def set_interrupt_check(self, checker):
                events.append(f"stt.check:{checker is not None}")

            def listen(self):
                raise AssertionError("Interrupted TTS should not reach STT")

        class FakeTTS:
            def set_interrupt_check(self, checker):
                events.append(f"tts.check:{checker is not None}")

            def speak_stream(self, text):
                raise AssertionError("Interrupted voice turn should not speak")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["Current question?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(
                    FakeSTT(),
                    FakeTTS(),
                    music=FakeMusic(),
                    session_control=FakeSessionControl(),
                )
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertFalse(processed)
        self.assertIn("music.restore", events)
        self.assertNotIn("music.pause", events)
        self.assertNotIn("music.stop", events)

    def test_process_voice_turn_clears_pending_question_when_override_already_active(self):
        class FakeSessionControl:
            def should_interrupt_voice(self):
                return True

            def should_discard_interrupted_voice_turn(self):
                return True

            def is_paused(self):
                return False

            def is_shutdown_requested(self):
                return False

        class FakeSTT:
            def listen(self):
                raise AssertionError("Override should not reach STT")

        class FakeTTS:
            def speak_stream(self, text):
                raise AssertionError("Override should not reach TTS")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["Already queued?", 1, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(
                    FakeSTT(),
                    FakeTTS(),
                    session_control=FakeSessionControl(),
                )
                result = pd.read_csv(record_path)
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertFalse(processed)
        self.assertTrue(pd.isna(result.loc[0, "Question"]) or result.loc[0, "Question"] == "")
        self.assertEqual(int(result.loc[0, "Question_Lock"]), 0)
        self.assertEqual(int(result.loc[0, "Resp_Lock"]), 1)

    def test_process_voice_turn_system_message_skips_stt_and_music_restart(self):
        events = []

        class FakeSTT:
            def listen(self):
                raise AssertionError("System message should not collect STT")

        class FakeTTS:
            def speak(self, text):
                events.append(f"speak:{text}")

            def speak_stream(self, text):
                events.append(f"speak:{text}")

        class FakeMusic:
            def stop(self):
                events.append("music.stop")

            def start(self):
                events.append("music.start")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["__CAITI_NO_RESPONSE__\nGoodbye. Take care.", 1, "stale", 0]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            original_settle_sec = voice_io.MUSIC_STOP_SETTLE_SEC
            voice_io.RECORD_CSV = record_path
            voice_io.MUSIC_STOP_SETTLE_SEC = 0
            try:
                processed = voice_io.process_voice_turn(FakeSTT(), FakeTTS(), music=FakeMusic())
                result = pd.read_csv(record_path)
            finally:
                voice_io.RECORD_CSV = original_record_csv
                voice_io.MUSIC_STOP_SETTLE_SEC = original_settle_sec

        self.assertTrue(processed)
        self.assertEqual(events, ["music.stop", "speak:Goodbye. Take care."])
        self.assertEqual(int(result.loc[0, "Question_Lock"]), 0)
        self.assertEqual(int(result.loc[0, "Resp_Lock"]), 1)
        self.assertTrue(pd.isna(result.loc[0, "Resp"]) or result.loc[0, "Resp"] == "")

    def test_process_voice_turn_system_message_restores_background_music(self):
        events = []

        class FakeSTT:
            def listen(self):
                raise AssertionError("System message should not collect STT")

        class FakeTTS:
            def speak(self, text):
                events.append(f"speak:{text}")

            def speak_stream(self, text):
                events.append(f"speak:{text}")

        class FakeMusic:
            def is_background(self):
                return True

            def start(self):
                events.append("music.start")

            def duck(self):
                events.append("music.duck")

            def restore_volume(self):
                events.append("music.restore")

        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["__CAITI_NO_RESPONSE__\nGoodbye. Take care.", 1, "stale", 0]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                processed = voice_io.process_voice_turn(FakeSTT(), FakeTTS(), music=FakeMusic())
            finally:
                voice_io.RECORD_CSV = original_record_csv

        self.assertTrue(processed)
        self.assertEqual(
            events,
            [
                "music.start",
                "music.duck",
                "speak:Goodbye. Take care.",
                "music.restore",
            ],
        )

    def test_wait_for_voice_io_drain_waits_for_pending_question_to_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record_path = str(Path(tmpdir) / "record.csv")
            pd.DataFrame(
                [["Pending message", 0, "", 1]],
                columns=HEADER,
            ).to_csv(record_path, index=False)

            original_record_csv = voice_io.RECORD_CSV
            voice_io.RECORD_CSV = record_path
            try:
                self.assertTrue(voice_io.wait_for_voice_io_drain(timeout_sec=0.2, poll_interval_sec=0.01))
            finally:
                voice_io.RECORD_CSV = original_record_csv


if __name__ == "__main__":
    unittest.main()
