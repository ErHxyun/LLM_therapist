import tempfile
import unittest
from pathlib import Path

from src.voice.tts.piper import PersistentPiperTTS


class _AudioChunk:
    sample_rate = 16000
    sample_width = 2
    sample_channels = 1
    audio_int16_bytes = b"\x00\x00" * 1600


class _FakeVoice:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, syn_config=None):
        self.calls.append((text, syn_config))
        yield _AudioChunk()


class _FakeProcess:
    def __init__(self, command, **_kwargs):
        self.command = command
        self.stderr = None
        self._polled = False
        self.terminated = False
        self.killed = False

    def poll(self):
        if self._polled:
            return 0
        self._polled = True
        return None

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class PersistentPiperTTSTests(unittest.TestCase):
    def test_speak_reuses_cached_wav_for_same_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "voice.onnx"
            model.write_bytes(b"model")
            Path(f"{model}.json").write_text("{}", encoding="utf-8")
            voice = _FakeVoice()
            played = []

            def popen_factory(command, **kwargs):
                played.append(command)
                return _FakeProcess(command, **kwargs)

            tts = PersistentPiperTTS(
                model_path=str(model),
                player="true",
                cache_dir=str(root / "cache"),
                popen_factory=popen_factory,
                synthesis_config_factory=lambda length_scale: {"length_scale": length_scale},
            )
            tts._voice = voice

            tts.speak("Hello there.")
            first_timing = dict(tts.last_timing)
            tts.speak("Hello there.")
            second_timing = dict(tts.last_timing)

        self.assertEqual(len(voice.calls), 1)
        self.assertEqual(len(played), 2)
        self.assertFalse(first_timing["cache_hit"])
        self.assertTrue(second_timing["cache_hit"])
        self.assertEqual(second_timing["text_length"], len("Hello there."))

    def test_playback_status_hook_wraps_player_lifetime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "voice.onnx"
            model.write_bytes(b"model")
            Path(f"{model}.json").write_text("{}", encoding="utf-8")
            tts = PersistentPiperTTS(
                model_path=str(model),
                player="true",
                cache_dir=None,
                popen_factory=lambda command, **kwargs: _FakeProcess(command, **kwargs),
                synthesis_config_factory=lambda length_scale: {"length_scale": length_scale},
            )
            tts._voice = _FakeVoice()
            events = []
            tts.set_playback_status_hook(lambda active: events.append(active))

            tts.speak("Hello.")

        self.assertEqual(events, [True, False])
        self.assertTrue(tts.last_timing["used_playback_markers"])


if __name__ == "__main__":
    unittest.main()
