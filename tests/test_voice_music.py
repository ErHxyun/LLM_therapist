import tempfile
import time
import unittest
from pathlib import Path

from src.voice.music import CommandMusic, NullMusic, format_music_command


class _FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self._running = True

    def poll(self):
        return None if self._running else 0

    def terminate(self):
        self.terminated = True
        self._running = False

    def kill(self):
        self.killed = True
        self._running = False

    def wait(self, timeout=None):
        self._running = False
        return 0


class VoiceMusicTests(unittest.TestCase):
    def test_format_music_command_uses_placeholder_or_appends_path(self):
        self.assertEqual(
            format_music_command("aplay -q {path}", "assets/audio/waiting music.wav"),
            "aplay -q 'assets/audio/waiting music.wav'",
        )
        self.assertEqual(
            format_music_command("paplay", "assets/audio/music.wav"),
            "paplay assets/audio/music.wav",
        )

    def test_null_music_is_noop(self):
        music = NullMusic()
        music.start()
        music.stop()

    def test_command_music_starts_and_stops_process(self):
        processes = []

        def popen_factory(*_args, **_kwargs):
            process = _FakeProcess()
            processes.append(process)
            return process

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "music.wav"
            path.write_bytes(b"RIFF" + b"0" * 80)
            music = CommandMusic(
                path=str(path),
                command="aplay -q {path}",
                popen_factory=popen_factory,
                poll_interval_sec=0.01,
            )
            music.start()
            deadline = time.time() + 1.0
            while not processes and time.time() < deadline:
                time.sleep(0.01)
            music.stop()

        self.assertTrue(processes)
        self.assertTrue(processes[0].terminated)


if __name__ == "__main__":
    unittest.main()
