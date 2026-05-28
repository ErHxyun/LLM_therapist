import tempfile
import unittest
import wave
import struct
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
    def read(self):
        return b""


class _FakePopen:
    def __init__(self, chunks):
        self.stdout = _ChunkStdout(chunks)
        self.stderr = _ChunkStderr()
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


class VoiceCommandScriptTests(unittest.TestCase):
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
