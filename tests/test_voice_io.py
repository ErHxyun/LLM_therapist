import unittest

from src.voice.backends import CommandSTT, CommandTTS
from src.voice.sentence_stream import split_for_tts


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

    def test_command_stt_returns_stdout_transcript(self):
        calls = []

        def runner(*args, **kwargs):
            calls.append((args, kwargs))
            return _Completed(stdout=" My weight increased recently. \n")

        stt = CommandSTT("local-stt", runner=runner)
        self.assertEqual(stt.listen(), "My weight increased recently.")
        self.assertEqual(calls[0][0][0], "local-stt")
        self.assertTrue(calls[0][1]["shell"])

    def test_command_tts_streams_text_chunks_to_stdin(self):
        spoken = []

        def runner(*args, **kwargs):
            spoken.append(kwargs.get("input"))
            return _Completed()

        tts = CommandTTS("local-tts", runner=runner)
        tts.speak_stream("One. Two?")
        self.assertEqual(spoken, ["One.", "Two?"])

    def test_command_backend_raises_on_failure(self):
        def runner(*args, **kwargs):
            return _Completed(returncode=2, stderr="boom")

        with self.assertRaises(RuntimeError):
            CommandSTT("bad-stt", runner=runner).listen()
        with self.assertRaises(RuntimeError):
            CommandTTS("bad-tts", runner=runner).speak("hello")


if __name__ == "__main__":
    unittest.main()
