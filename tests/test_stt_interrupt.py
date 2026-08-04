import unittest

from src.voice.backends import FasterWhisperSTT
from src.voice.exceptions import VoiceInterrupted


class FasterWhisperInterruptTests(unittest.TestCase):
    def test_interruption_after_recording_skips_waiting_music_and_transcription(self):
        events = []
        interrupted = [False]
        stt = FasterWhisperSTT()
        stt._make_wav_file = lambda: ("unused.wav", False)

        class FakeMusic:
            def start(self):
                events.append("music.start")

        def record(_wav_file):
            events.append("record")
            interrupted[0] = True

        stt._record = record
        stt._transcribe = lambda _wav_file: events.append("transcribe") or "unexpected"
        stt.set_interrupt_check(lambda: interrupted[0])

        with self.assertRaises(VoiceInterrupted):
            stt.listen_with_waiting_music(FakeMusic())

        self.assertEqual(events, ["record"])


if __name__ == "__main__":
    unittest.main()
