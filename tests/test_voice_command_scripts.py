import tempfile
import unittest
import wave
import struct
import math
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from scripts import faster_whisper_stt_command as stt_script
from scripts import piper_tts_command as tts_script
from scripts import smoke_test_voice as smoke_script


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _ChunkStdout:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def read(self, _size):
        if not self.chunks:
            return b""
        return self.chunks.pop(0)


class _ChunkStderr:
    def __init__(self, payload: bytes = b""):
        self.payload = payload

    def read(self):
        return self.payload


class _FakePopen:
    def __init__(self, chunks, stderr: bytes = b""):
        self.stdout = _ChunkStdout(chunks)
        self.stderr = _ChunkStderr(stderr)
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if not self.terminated and not self.killed else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.terminated = True
        return 0


def _pcm_chunk(value: int, samples: int = 100) -> bytes:
    return struct.pack("<h", value) * samples


def _write_test_wav(path: Path, sample_value: int, frames: int = 1600, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack("<h", sample_value) * frames)


class VoiceCommandScriptTests(unittest.TestCase):
    def test_whisper_transcription_keeps_internal_timestamp_segmentation(self):
        calls = []

        class Segment:
            def __init__(self, text):
                self.text = text

        class FakeWhisper:
            def transcribe(self, *_args, **kwargs):
                calls.append(kwargs)
                return [Segment(" First part. "), Segment(" Second part. ")], object()

        transcript = stt_script.transcribe_wav_with_model(FakeWhisper(), "long.wav")

        self.assertEqual(transcript, "First part. Second part.")
        self.assertFalse(calls[0]["without_timestamps"])
        self.assertFalse(calls[0]["condition_on_previous_text"])

    def test_piper_command_uses_stdin_model_and_output_file(self):
        command = tts_script.build_piper_command(
            "piper",
            "models/piper/en_US-amy-medium.onnx",
            "/tmp/out.wav",
            0.8,
            0.4,
        )

        self.assertEqual(command[0], "piper")
        self.assertIn("--model", command)
        self.assertIn("models/piper/en_US-amy-medium.onnx", command)
        self.assertIn("--output_file", command)
        self.assertIn("/tmp/out.wav", command)
        self.assertIn("--sentence_silence", command)
        self.assertIn("0.4", command)

    def test_player_command_defaults_aplay_quiet(self):
        self.assertEqual(tts_script.build_player_command("aplay", "/tmp/out.wav"), ["aplay", "-q", "/tmp/out.wav"])
        self.assertEqual(tts_script.build_player_command("paplay", "/tmp/out.wav"), ["paplay", "/tmp/out.wav"])
        self.assertEqual(
            tts_script.build_player_command("aplay -D plughw:3,0 -q", "/tmp/out.wav"),
            ["aplay", "-D", "plughw:3,0", "-q", "/tmp/out.wav"],
        )

    def test_normalize_speech_text_collapses_blank_lines(self):
        self.assertEqual(
            tts_script.normalize_speech_text("Hello there.\n\nHow are you today?\n"),
            "Hello there. How are you today?",
        )

    def test_prepare_aplay_wav_converts_to_48k_stereo(self):
        try:
            import numpy as np
            import soundfile  # noqa: F401
            import scipy.signal  # noqa: F401
        except Exception:
            self.skipTest("audio conversion dependencies are unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "mono.wav"
            rate = 22050
            duration_sec = 0.2
            frames = int(rate * duration_sec)
            with wave.open(str(source), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(rate)
                samples = bytearray()
                for index in range(frames):
                    value = int(12000 * math.sin((2 * math.pi * 440 * index) / rate))
                    samples.extend(struct.pack("<h", value))
                writer.writeframes(bytes(samples))

            converted = tts_script._prepare_aplay_wav(str(source))
            self.assertNotEqual(converted, str(source))
            with wave.open(converted, "rb") as reader:
                self.assertEqual(reader.getframerate(), 48000)
                self.assertEqual(reader.getnchannels(), 2)

            Path(converted).unlink(missing_ok=True)

    def test_piper_speak_runs_piper_then_player(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "voice.onnx"
            model.write_bytes(b"model")
            Path(f"{model}.json").write_text("{}", encoding="utf-8")

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                if command[0] == "piper":
                    output_file = command[command.index("--output_file") + 1]
                    Path(output_file).write_bytes(b"RIFF" + b"0" * 80)
                return _Completed()

            tts_script.speak_text(
                "Hello.",
                model_path=str(model),
                executable="piper",
                player="aplay",
                runner=runner,
            )

        self.assertEqual(calls[0][0][0], "piper")
        self.assertEqual(calls[0][1]["input"], "Hello.")
        self.assertEqual(calls[1][0][0], "aplay")

    def test_piper_speak_can_emit_playback_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "voice.onnx"
            model.write_bytes(b"model")
            Path(f"{model}.json").write_text("{}", encoding="utf-8")

            def runner(command, **kwargs):
                if command[0] == "piper":
                    output_file = command[command.index("--output_file") + 1]
                    Path(output_file).write_bytes(b"RIFF" + b"0" * 80)
                return _Completed()

            stderr = StringIO()
            with redirect_stderr(stderr):
                tts_script.speak_text(
                    "Hello.",
                    model_path=str(model),
                    executable="piper",
                    player="aplay",
                    emit_playback_markers=True,
                    runner=runner,
                )

        self.assertEqual(
            stderr.getvalue().splitlines(),
            [tts_script.PLAYBACK_START_MARKER, tts_script.PLAYBACK_END_MARKER],
        )

    def test_piper_voice_validation_requires_json_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "voice.onnx"
            model.write_bytes(b"model")

            with self.assertRaises(FileNotFoundError):
                tts_script.validate_piper_voice(str(model))

    def test_piper_speak_can_use_cached_fallback(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "missing.onnx"
            fallback = Path(tmpdir) / "fallback.wav"
            fallback.write_bytes(b"RIFF" + b"0" * 80)

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                return _Completed()

            tts_script.speak_text(
                "Hello.",
                model_path=str(model),
                executable="piper",
                player="aplay",
                fallback_executable="",
                cached_fallback_wav=str(fallback),
                runner=runner,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][0], "aplay")

    def test_piper_speak_reuses_synthesized_cache(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "voice.onnx"
            model.write_bytes(b"model")
            Path(f"{model}.json").write_text("{}", encoding="utf-8")
            cache_dir = root / "cache"

            def runner(command, **kwargs):
                calls.append((command, kwargs))
                if command[0] == "piper":
                    output_file = command[command.index("--output_file") + 1]
                    Path(output_file).write_bytes(b"RIFF" + b"0" * 80)
                return _Completed()

            for _ in range(2):
                tts_script.speak_text(
                    "Hello.",
                    model_path=str(model),
                    executable="piper",
                    player="aplay",
                    cache_dir=str(cache_dir),
                    runner=runner,
                )

        piper_calls = [call for call in calls if call[0][0] == "piper"]
        player_calls = [call for call in calls if call[0][0] == "aplay"]
        self.assertEqual(len(piper_calls), 1)
        self.assertEqual(len(player_calls), 2)

    def test_arecord_command_is_mono_16k_wav(self):
        command = stt_script.build_arecord_command(
            "/tmp/input.wav",
            seconds=5.2,
            sample_rate=16000,
            channels=1,
            device="hw:1,0",
        )

        self.assertEqual(command[:2], ["arecord", "-q"])
        self.assertIn("S16_LE", command)
        self.assertIn("16000", command)
        self.assertIn("6", command)
        self.assertIn("hw:1,0", command)
        self.assertEqual(command[-1], "/tmp/input.wav")

    def test_arecord_raw_command_streams_stdout_for_auto_stop(self):
        command = stt_script.build_arecord_raw_command(
            seconds=5.2,
            sample_rate=16000,
            channels=1,
            device="plughw:0,0",
        )

        self.assertEqual(command[:2], ["arecord", "-q"])
        self.assertIn("-t", command)
        self.assertIn("raw", command)
        self.assertIn("6", command)
        self.assertIn("plughw:0,0", command)

    def test_analyze_wav_reports_basic_audio_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "input.wav"
            with wave.open(str(wav_path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes((1000).to_bytes(2, byteorder="little", signed=True) * 16000)

            metrics = stt_script.analyze_wav(str(wav_path))

        self.assertAlmostEqual(metrics["duration_sec"], 1.0, places=2)
        self.assertEqual(metrics["sample_rate"], 16000.0)
        self.assertEqual(metrics["channels"], 1.0)
        self.assertGreater(metrics["rms_dbfs"], -40.0)

    def test_auto_stop_records_until_speech_then_trailing_silence(self):
        speech = _pcm_chunk(8000)
        silence = _pcm_chunk(0)
        chunks = [speech, speech, silence, silence, silence, speech]

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "auto.wav"
            stt_script.record_wav_auto_stop(
                str(wav_path),
                max_seconds=10,
                sample_rate=1000,
                channels=1,
                vad_detector="energy",
                silence_threshold_dbfs=-45,
                silence_timeout_sec=0.2,
                trailing_pad_sec=0.1,
                min_speech_seconds=0.1,
                min_record_seconds=0.1,
                no_speech_timeout_sec=1.0,
                chunk_ms=100,
                popen_factory=lambda *_args, **_kwargs: _FakePopen(chunks),
            )
            with wave.open(str(wav_path), "rb") as reader:
                self.assertEqual(reader.getnframes(), 500)

    def test_auto_stop_no_speech_timeout_stops_early(self):
        chunks = [_pcm_chunk(0) for _ in range(10)]

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "silence.wav"
            stt_script.record_wav_auto_stop(
                str(wav_path),
                max_seconds=10,
                sample_rate=1000,
                channels=1,
                vad_detector="energy",
                silence_threshold_dbfs=-45,
                no_speech_timeout_sec=0.3,
                chunk_ms=100,
                popen_factory=lambda *_args, **_kwargs: _FakePopen(chunks),
            )
            with wave.open(str(wav_path), "rb") as reader:
                self.assertLessEqual(reader.getnframes(), 400)

    def test_record_wav_raises_on_arecord_failure(self):
        def runner(command, **kwargs):
            return _Completed(returncode=1, stderr="no mic")

        with self.assertRaises(RuntimeError):
            stt_script.record_wav(
                "/tmp/missing.wav",
                seconds=1,
                runner=runner,
            )

    def test_record_wav_retries_with_default_when_configured_device_is_missing(self):
        calls = []

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "input.wav"

            def runner(command, **kwargs):
                calls.append(command)
                if "-D" in command and command[command.index("-D") + 1] == "plughw:0,0":
                    return _Completed(returncode=1, stderr="arecord: main:831: audio open error: No such file or directory")
                wav_path.write_bytes(b"RIFF" + b"0" * 80)
                return _Completed()

            stt_script.record_wav(
                str(wav_path),
                seconds=1,
                device="plughw:0,0",
                runner=runner,
            )

        self.assertEqual(calls[0][calls[0].index("-D") + 1], "plughw:0,0")
        self.assertEqual(calls[1][calls[1].index("-D") + 1], "default")

    def test_record_wav_retries_with_default_when_first_device_captures_silence(self):
        calls = []

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "input.wav"

            def runner(command, **kwargs):
                calls.append(command)
                device = command[command.index("-D") + 1] if "-D" in command else ""
                if device == "pulse":
                    _write_test_wav(wav_path, 0)
                else:
                    _write_test_wav(wav_path, 1000)
                return _Completed()

            stt_script.record_wav(
                str(wav_path),
                seconds=1,
                device="pulse",
                runner=runner,
            )

        self.assertEqual(calls[0][calls[0].index("-D") + 1], "pulse")
        self.assertEqual(calls[1][calls[1].index("-D") + 1], "default")

    def test_auto_stop_retries_with_default_when_configured_device_is_missing(self):
        calls = []

        def popen_factory(command, **_kwargs):
            calls.append(command)
            if "-D" in command and command[command.index("-D") + 1] == "plughw:0,0":
                return _FakePopen([], stderr=b"arecord: main:831: audio open error: No such file or directory")
            return _FakePopen([_pcm_chunk(8000), _pcm_chunk(8000), _pcm_chunk(0), _pcm_chunk(0), _pcm_chunk(0)])

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "auto.wav"
            stt_script.record_wav_auto_stop(
                str(wav_path),
                max_seconds=10,
                sample_rate=1000,
                channels=1,
                device="plughw:0,0",
                vad_detector="energy",
                silence_threshold_dbfs=-45,
                silence_timeout_sec=0.2,
                trailing_pad_sec=0.1,
                min_speech_seconds=0.1,
                min_record_seconds=0.1,
                no_speech_timeout_sec=1.0,
                chunk_ms=100,
                popen_factory=popen_factory,
            )

        self.assertEqual(calls[0][calls[0].index("-D") + 1], "plughw:0,0")
        self.assertEqual(calls[1][calls[1].index("-D") + 1], "default")

    def test_auto_stop_retries_with_default_when_first_device_captures_silence(self):
        calls = []

        def popen_factory(command, **_kwargs):
            calls.append(command)
            device = command[command.index("-D") + 1] if "-D" in command else ""
            if device == "pulse":
                return _FakePopen([_pcm_chunk(0), _pcm_chunk(0), _pcm_chunk(0), _pcm_chunk(0), _pcm_chunk(0)])
            return _FakePopen([_pcm_chunk(8000), _pcm_chunk(8000), _pcm_chunk(0), _pcm_chunk(0), _pcm_chunk(0)])

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "auto.wav"
            stt_script.record_wav_auto_stop(
                str(wav_path),
                max_seconds=10,
                sample_rate=1000,
                channels=1,
                device="pulse",
                vad_detector="energy",
                silence_threshold_dbfs=-45,
                silence_timeout_sec=0.2,
                trailing_pad_sec=0.1,
                min_speech_seconds=0.1,
                min_record_seconds=0.1,
                no_speech_timeout_sec=1.0,
                chunk_ms=100,
                popen_factory=popen_factory,
            )

        self.assertEqual(calls[0][calls[0].index("-D") + 1], "pulse")
        self.assertEqual(calls[1][calls[1].index("-D") + 1], "default")

    def test_smoke_test_command_option_parses_space_and_equals_forms(self):
        self.assertEqual(
            smoke_script.command_option("python scripts/piper_tts_command.py --model models/piper/test.onnx", "--model"),
            "models/piper/test.onnx",
        )
        self.assertEqual(
            smoke_script.command_option("python scripts/stt.py --audio-device=plughw:0,0", "--audio-device"),
            "plughw:0,0",
        )

    def test_smoke_test_resolves_repo_relative_paths(self):
        resolved = smoke_script.resolve_repo_path("scripts/piper_tts_command.py")
        self.assertTrue(resolved.is_absolute())
        self.assertTrue(resolved.exists())


if __name__ == "__main__":
    unittest.main()
