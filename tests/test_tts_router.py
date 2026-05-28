import tempfile
import unittest
from pathlib import Path

from src.voice.tts.command import CommandTTS, ConsoleTTS
from src.voice.tts.router import TTSRouteSettings, build_role_tts, build_tts_from_settings, extract_command_option


class TTSRouterTests(unittest.TestCase):
    def test_extract_command_option_handles_space_and_equals_forms(self):
        self.assertEqual(
            extract_command_option("python scripts/piper_tts_command.py --model models/voice.onnx", "--model"),
            "models/voice.onnx",
        )
        self.assertEqual(
            extract_command_option("python scripts/piper_tts_command.py --model=models/voice.onnx", "--model"),
            "models/voice.onnx",
        )

    def test_console_backend_builds_console_tts(self):
        tts = build_tts_from_settings(TTSRouteSettings(role="primary", backend="console"))
        self.assertIsInstance(tts, ConsoleTTS)

    def test_primary_and_cbt_roles_reuse_primary_tts(self):
        primary = object()
        self.assertIs(build_role_tts("primary", primary_tts=primary), primary)
        self.assertIs(build_role_tts("cbt", primary_tts=primary), primary)

    def test_command_backend_builds_command_tts_for_valid_piper_voice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "voice.onnx"
            model.write_bytes(b"model")
            Path(f"{model}.json").write_text("{}", encoding="utf-8")
            tts = build_tts_from_settings(
                TTSRouteSettings(
                    role="intermission",
                    backend="command",
                    command=f"python scripts/piper_tts_command.py --model {model}",
                    timeout_sec=12,
                    fallback_to_primary=True,
                ),
                primary_tts=object(),
            )

        self.assertIsInstance(tts, CommandTTS)
        self.assertEqual(tts.timeout_sec, 12)

    def test_intermission_invalid_piper_voice_falls_back_to_primary(self):
        primary = object()
        tts = build_tts_from_settings(
            TTSRouteSettings(
                role="intermission",
                backend="command",
                command="python scripts/piper_tts_command.py --model /tmp/missing.onnx",
                fallback_to_primary=True,
            ),
            primary_tts=primary,
        )

        self.assertIs(tts, primary)

    def test_intermission_invalid_piper_voice_can_disable_tts(self):
        tts = build_tts_from_settings(
            TTSRouteSettings(
                role="intermission",
                backend="command",
                command="python scripts/piper_tts_command.py --model /tmp/missing.onnx",
                fallback_to_primary=False,
            )
        )

        self.assertIsNone(tts)

    def test_strict_primary_keeps_command_even_if_piper_voice_is_invalid(self):
        tts = build_tts_from_settings(
            TTSRouteSettings(
                role="primary",
                backend="command",
                command="python scripts/piper_tts_command.py --model /tmp/missing.onnx",
                strict=True,
            )
        )

        self.assertIsInstance(tts, CommandTTS)


if __name__ == "__main__":
    unittest.main()
