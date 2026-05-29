import math
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts import benchmark_stt as bench


def _write_silence_wav(path: Path, seconds: float = 0.1) -> None:
    sample_rate = 16000
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


class STTBenchmarkScriptTest(unittest.TestCase):
    def test_collect_audio_files_sorts_and_filters_wav_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ignore.txt").write_text("nope", encoding="utf-8")
            _write_silence_wav(root / "b.wav")
            _write_silence_wav(root / "a.WAV")

            paths = bench.collect_audio_files(root, [], recursive=False)

            self.assertEqual([path.name for path in paths], ["a.WAV", "b.wav"])

    def test_vad_filter_values_supports_current_and_both(self):
        self.assertEqual(bench.vad_filter_values("current", True), [True])
        self.assertEqual(bench.vad_filter_values("current", False), [False])
        self.assertEqual(bench.vad_filter_values("both", True), [True, False])
        self.assertEqual(bench.vad_filter_values("on", False), [True])
        self.assertEqual(bench.vad_filter_values("off", True), [False])

    def test_parse_positive_int_values_supports_repeated_and_csv_values(self):
        self.assertEqual(bench.parse_positive_int_values([], 3), [3])
        self.assertEqual(bench.parse_positive_int_values(["1,2", "2", "3"], 5), [1, 2, 3])
        with self.assertRaises(ValueError):
            bench.parse_positive_int_values(["0"], 1)

    def test_prompt_variants_supports_current_none_and_both(self):
        self.assertEqual(bench.prompt_variants("current", "prompt"), [("current", "prompt")])
        self.assertEqual(bench.prompt_variants("none", "prompt"), [("none", "")])
        self.assertEqual(bench.prompt_variants("both", "prompt"), [("current", "prompt"), ("none", "")])

    def test_run_benchmark_reuses_model_and_records_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            _write_silence_wav(audio)
            calls = []
            original_load = bench.whisper_stt.load_whisper_model
            original_transcribe = bench.whisper_stt.transcribe_wav_with_model
            try:
                bench.whisper_stt.load_whisper_model = (
                    lambda *args, **kwargs: calls.append(("load", args, kwargs)) or object()
                )
                bench.whisper_stt.transcribe_wav_with_model = (
                    lambda *args, **kwargs: calls.append(("transcribe", args, kwargs)) or "yes I am sleeping"
                )

                samples = bench.run_benchmark(
                    audio_paths=[audio],
                    models=["tiny.en"],
                    device="cpu",
                    compute_type="int8",
                    beam_sizes=[1, 2],
                    best_of_values=[1],
                    language="en",
                    prompts=[("none", "")],
                    vad_filters=[True],
                    iterations=2,
                )
            finally:
                bench.whisper_stt.load_whisper_model = original_load
                bench.whisper_stt.transcribe_wav_with_model = original_transcribe

            self.assertEqual(len([call for call in calls if call[0] == "load"]), 1)
            self.assertEqual(len(samples), 4)
            self.assertEqual(samples[0].transcript, "yes I am sleeping")
            self.assertEqual(samples[0].transcript_chars, len("yes I am sleeping"))
            self.assertEqual(samples[0].prompt_label, "none")
            self.assertTrue(math.isfinite(samples[0].realtime_factor))

    def test_summarize_groups_by_stt_parameters(self):
        sample = bench.STTBenchmarkSample(
            audio_path="sample.wav",
            audio_name="sample.wav",
            model="small.en",
            device="cpu",
            compute_type="int8",
            beam_size=1,
            best_of=1,
            vad_filter=True,
            prompt_label="current",
            iteration=1,
            model_load_sec=1.0,
            audio_duration_sec=2.0,
            sample_rate=16000,
            channels=1,
            rms_dbfs=-30.0,
            peak_dbfs=-10.0,
            transcribe_sec=0.5,
            realtime_factor=0.25,
            transcript_chars=2,
            transcript="ok",
        )

        summary = bench.summarize([sample])[0]

        self.assertEqual(summary["beam_size"], 1)
        self.assertEqual(summary["best_of"], 1)
        self.assertEqual(summary["prompt_label"], "current")
        self.assertEqual(summary["transcribe_sec_mean"], 0.5)

    def test_list_audio_does_not_require_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_silence_wav(root / "sample.wav")

            with redirect_stdout(StringIO()) as stdout:
                code = bench.main(["--audio-dir", str(root), "--list-audio"])

            self.assertEqual(code, 0)
            self.assertIn("sample.wav", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
