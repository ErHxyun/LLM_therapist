import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.voice.music import (
    CommandMusic,
    MPVBackgroundMusic,
    MusicMode,
    NullMusic,
    build_mpv_command_args,
    build_music_command_args,
    format_music_command,
)


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

    def test_build_music_command_args_avoids_shell_and_preserves_paths(self):
        self.assertEqual(
            build_music_command_args("aplay -q {path}", "assets/audio/waiting music.wav"),
            ["aplay", "-q", "assets/audio/waiting music.wav"],
        )

    def test_build_mpv_command_args_uses_ipc_and_looping(self):
        self.assertEqual(
            build_mpv_command_args("assets/audio/music.wav", "/tmp/caiti.sock", 35),
            [
                "mpv",
                "--no-video",
                "--quiet",
                "--force-window=no",
                "--loop-file=inf",
                "--input-ipc-server=/tmp/caiti.sock",
                "--volume=35",
                "assets/audio/music.wav",
            ],
        )

    def test_null_music_is_noop(self):
        music = NullMusic()
        music.start()
        music.stop()

    def test_command_music_starts_and_stops_process(self):
        processes = []
        popen_kwargs = []

        def popen_factory(*args, **kwargs):
            process = _FakeProcess()
            processes.append(process)
            processes.append(args[0])
            popen_kwargs.append(kwargs)
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
            self.assertTrue(music.is_playing())
            music.stop()

        self.assertTrue(processes)
        self.assertTrue(processes[0].terminated)
        self.assertFalse(music.is_playing())
        self.assertEqual(processes[1], ["aplay", "-q", str(path)])
        self.assertNotIn("shell", popen_kwargs[0])
        self.assertTrue(popen_kwargs[0]["start_new_session"])

    def test_mpv_background_music_ducks_pauses_resumes_and_stops(self):
        processes = []
        commands = []
        popen_args = []

        def popen_factory(*args, **kwargs):
            process = _FakeProcess()
            process.pid = 12345
            processes.append(process)
            popen_args.append(args[0])
            return process

        def ipc_sender(_ipc_path, command):
            commands.append(command)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "music.wav"
            path.write_bytes(b"RIFF" + b"0" * 80)
            music = MPVBackgroundMusic(
                path=str(path),
                ipc_path="/tmp/test-caiti-mpv.sock",
                volume_percent=35,
                duck_volume_percent=7,
                popen_factory=popen_factory,
                ipc_sender=ipc_sender,
            )

            music.start()
            self.assertTrue(music.is_background())
            self.assertTrue(music.is_playing())
            music.duck()
            music.restore_volume()
            music.pause()
            self.assertFalse(music.is_playing())
            music.resume()
            self.assertTrue(music.is_playing())
            music.stop()

        self.assertEqual(
            popen_args[0],
            [
                "mpv",
                "--no-video",
                "--quiet",
                "--force-window=no",
                "--loop-file=inf",
                "--input-ipc-server=/tmp/test-caiti-mpv.sock",
                "--volume=35",
                str(path),
            ],
        )
        self.assertIn(["set_property", "volume", 7], commands)
        self.assertIn(["set_property", "volume", 35], commands)
        self.assertIn(["set_property", "pause", True], commands)
        self.assertIn(["set_property", "pause", False], commands)
        self.assertIn(["quit"], commands)
        self.assertTrue(processes[0].terminated)

    def test_mpv_background_music_cycles_modes_without_session_side_effects(self):
        popen_args = []

        def popen_factory(*args, **kwargs):
            process = _FakeProcess()
            process.pid = 12345
            popen_args.append(args[0])
            return process

        with tempfile.TemporaryDirectory() as tmpdir:
            music_path = Path(tmpdir) / "music.wav"
            fireplace_path = Path(tmpdir) / "fireplace.wav"
            seawaves_path = Path(tmpdir) / "seawaves.wav"
            for path in (music_path, fireplace_path, seawaves_path):
                path.write_bytes(b"RIFF" + b"0" * 80)

            music = MPVBackgroundMusic(
                path=str(music_path),
                ipc_path="/tmp/test-caiti-mpv-modes.sock",
                volume_percent=35,
                duck_volume_percent=7,
                modes=[
                    MusicMode("music", str(music_path)),
                    MusicMode("fireplace", str(fireplace_path)),
                    MusicMode("seawaves", str(seawaves_path)),
                    MusicMode("off", ""),
                ],
                popen_factory=popen_factory,
                ipc_sender=lambda _ipc_path, _command: None,
            )
            music._force_stop_matching_mpv_processes = lambda: None

            music.start()
            self.assertEqual(popen_args[-1][-1], str(music_path))
            self.assertEqual(music.cycle_mode(), "fireplace")
            self.assertEqual(popen_args[-1][-1], str(fireplace_path))
            self.assertEqual(music.cycle_mode(), "seawaves")
            self.assertEqual(popen_args[-1][-1], str(seawaves_path))
            self.assertEqual(music.cycle_mode(), "off")
            self.assertFalse(music.is_playing())
            self.assertEqual(music.cycle_mode(), "music")
            self.assertTrue(music.is_playing())
            self.assertEqual(popen_args[-1][-1], str(music_path))
            music.stop()

    def test_restore_volume_clears_ducked_state_when_music_is_off(self):
        popen_args = []

        def popen_factory(*args, **kwargs):
            process = _FakeProcess()
            process.pid = 12345
            popen_args.append(args[0])
            return process

        with tempfile.TemporaryDirectory() as tmpdir:
            music_path = Path(tmpdir) / "music.wav"
            music_path.write_bytes(b"RIFF" + b"0" * 80)

            music = MPVBackgroundMusic(
                path=str(music_path),
                ipc_path="/tmp/test-caiti-mpv-off.sock",
                volume_percent=35,
                duck_volume_percent=7,
                modes=[MusicMode("music", str(music_path)), MusicMode("off", "")],
                popen_factory=popen_factory,
                ipc_sender=lambda _ipc_path, _command: None,
            )
            music._force_stop_matching_mpv_processes = lambda: None

            music.start()
            music.duck()
            self.assertEqual(music.cycle_mode(), "off")
            music.restore_volume()
            self.assertEqual(music.cycle_mode(), "music")
            music.start()

        self.assertIn("--volume=35", popen_args[-1])

    def test_mpv_orphan_scan_matches_only_this_music_ipc_socket(self):
        output = "\n".join(
            [
                " 111 mpv mpv --no-video --input-ipc-server=/tmp/caiti_mpv_music.sock assets/audio/music.wav",
                " 222 mpv mpv --no-video --input-ipc-server=/tmp/other.sock assets/audio/music.wav",
                " 333 bash bash -lc mpv --input-ipc-server=/tmp/caiti_mpv_music.sock",
            ]
        )
        with patch("src.voice.music.subprocess.check_output", return_value=output):
            pids = MPVBackgroundMusic._find_matching_mpv_pids(
                "--input-ipc-server=/tmp/caiti_mpv_music.sock"
            )

        self.assertEqual(pids, [111])

    def test_mpv_stop_scans_for_orphan_even_without_process_handle(self):
        calls = []
        music = MPVBackgroundMusic(path="missing.wav", ipc_path="/tmp/caiti-test.sock")
        music._force_stop_matching_mpv_processes = lambda: calls.append("force")

        music.stop()

        self.assertEqual(calls, ["force"])


if __name__ == "__main__":
    unittest.main()
