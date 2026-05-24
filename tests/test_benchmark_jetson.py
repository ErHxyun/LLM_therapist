import unittest
from contextlib import redirect_stdout
from io import StringIO

from scripts import benchmark_jetson as bench


class BenchmarkJetsonScriptTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(bench.percentile([], 50), None)
        self.assertEqual(bench.percentile([4.0], 95), 4.0)
        self.assertEqual(bench.percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertAlmostEqual(bench.percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)

    def test_summarize_samples_counts_failures_and_contract_failures(self):
        samples = [
            bench.BenchmarkSample(
                name="case",
                task="task",
                adapter="adapter",
                iteration=1,
                latency_ms=10.0,
                switch_latency_ms=1.0,
                memory_before={},
                memory_after={},
                output_chars=3,
                normalized_output="ok",
                contract_ok=True,
                contract_message="ok",
            ),
            bench.BenchmarkSample(
                name="case",
                task="task",
                adapter="adapter",
                iteration=2,
                latency_ms=20.0,
                switch_latency_ms=2.0,
                memory_before={},
                memory_after={},
                output_chars=3,
                normalized_output="bad",
                contract_ok=False,
                contract_message="bad contract",
            ),
            bench.BenchmarkSample(
                name="case",
                task="task",
                adapter="adapter",
                iteration=3,
                latency_ms=None,
                switch_latency_ms=None,
                memory_before={},
                memory_after={},
                output_chars=0,
                normalized_output="",
                contract_ok=False,
                contract_message="failed",
                error="traceback",
            ),
        ]

        summary = bench.summarize_samples(samples)["case"]

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["contract_failures"], 1)
        self.assertEqual(summary["latency_ms_p50"], 15.0)
        self.assertEqual(summary["switch_latency_ms_mean"], 1.5)

    def test_list_cases_does_not_require_model_load(self):
        with redirect_stdout(StringIO()):
            self.assertEqual(bench.main(["--list-cases"]), 0)


if __name__ == "__main__":
    unittest.main()
