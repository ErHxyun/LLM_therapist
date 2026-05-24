"""Benchmark local CaiTI runtime latency and memory on Jetson.

The benchmark runs the same adapter prompts used by smoke_test_adapters.py, but
records load time, per-task latency, adapter switch latency, memory snapshots,
and contract failures in a JSON report.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_test_adapters import build_smoke_cases, missing_runtime_dependencies  # noqa: E402
from src.local_llm.routing import resolve_adapter  # noqa: E402
from src.local_llm.runtime import LocalCaiTIRuntime, RuntimeSettings  # noqa: E402
from src.local_llm.types import GenerationConfig, LLMTask  # noqa: E402
from src.utils.config_loader import (  # noqa: E402
    LOCAL_LLM_BASE_SUBDIR,
    LOCAL_LLM_DEFAULT_MAX_NEW_TOKENS,
    LOCAL_LLM_DEVICE_MAP,
    LOCAL_LLM_MAX_INPUT_TOKENS,
    LOCAL_LLM_MODEL_ID,
    LOCAL_LLM_TEMPERATURE,
    LOCAL_LLM_TOKENIZER_ID,
    LOCAL_LLM_TOKENIZER_SUBDIR,
    LOCAL_LLM_TOP_P,
    LOCAL_LLM_TORCH_DTYPE,
)


@dataclass(frozen=True)
class BenchmarkSample:
    name: str
    task: str
    adapter: str | None
    iteration: int
    latency_ms: float | None
    switch_latency_ms: float | None
    memory_before: dict[str, float | None]
    memory_after: dict[str, float | None]
    output_chars: int
    normalized_output: str
    contract_ok: bool
    contract_message: str
    error: str | None = None


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        model_id=LOCAL_LLM_MODEL_ID,
        base_subdir=LOCAL_LLM_BASE_SUBDIR,
        tokenizer_id=LOCAL_LLM_TOKENIZER_ID,
        tokenizer_subdir=LOCAL_LLM_TOKENIZER_SUBDIR,
        device_map=LOCAL_LLM_DEVICE_MAP,
        torch_dtype=LOCAL_LLM_TORCH_DTYPE,
    )


def package_versions() -> dict[str, str | None]:
    packages = ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _read_proc_status_mb() -> dict[str, float | None]:
    values = {"process_rss_mb": None, "process_peak_rss_mb": None}
    key_map = {"VmRSS": "process_rss_mb", "VmHWM": "process_peak_rss_mb"}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in key_map:
                amount = rest.strip().split()[0]
                values[key_map[key]] = float(amount) / 1024.0
    except OSError:
        pass
    return values


def _read_meminfo_mb() -> dict[str, float | None]:
    wanted = {
        "MemTotal": "system_mem_total_mb",
        "MemAvailable": "system_mem_available_mb",
        "SwapTotal": "system_swap_total_mb",
        "SwapFree": "system_swap_free_mb",
    }
    values = {name: None for name in wanted.values()}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in wanted:
                amount = rest.strip().split()[0]
                values[wanted[key]] = float(amount) / 1024.0
    except OSError:
        pass
    return values


def _read_cuda_memory_mb() -> dict[str, float | None]:
    values = {
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
        "cuda_peak_allocated_mb": None,
        "cuda_peak_reserved_mb": None,
    }
    try:
        import torch
    except ImportError:
        return values
    if not torch.cuda.is_available():
        return values
    values["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024.0 / 1024.0
    values["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024.0 / 1024.0
    values["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    values["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024.0 / 1024.0
    return values


def snapshot_memory() -> dict[str, float | None]:
    snapshot: dict[str, float | None] = {}
    snapshot.update(_read_proc_status_mb())
    snapshot.update(_read_meminfo_mb())
    snapshot.update(_read_cuda_memory_mb())
    return snapshot


def synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(samples: list[BenchmarkSample]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    names = sorted({sample.name for sample in samples})
    for name in names:
        group = [sample for sample in samples if sample.name == name]
        latencies = [sample.latency_ms for sample in group if sample.latency_ms is not None and sample.error is None]
        switches = [
            sample.switch_latency_ms
            for sample in group
            if sample.switch_latency_ms is not None and sample.error is None
        ]
        summary[name] = {
            "count": len(group),
            "successes": len(latencies),
            "failures": sum(1 for sample in group if sample.error),
            "contract_failures": sum(1 for sample in group if not sample.contract_ok and not sample.error),
            "latency_ms_min": min(latencies) if latencies else None,
            "latency_ms_mean": (sum(latencies) / len(latencies)) if latencies else None,
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p95": percentile(latencies, 95),
            "latency_ms_max": max(latencies) if latencies else None,
            "switch_latency_ms_mean": (sum(switches) / len(switches)) if switches else None,
        }
    return summary


def _format_prompt(system_prompt: str, payload: str) -> str:
    return f"{system_prompt.rstrip()}\n{payload.lstrip()}"


def run_base_sample(runtime: LocalCaiTIRuntime, iteration: int) -> BenchmarkSample:
    memory_before = snapshot_memory()
    try:
        synchronize_cuda()
        start = time.perf_counter()
        result = runtime.generate_base(
            "You are a concise CaiTI benchmark assistant.",
            "Say hello in one short sentence.",
            GenerationConfig(
                max_new_tokens=min(16, LOCAL_LLM_DEFAULT_MAX_NEW_TOKENS),
                temperature=LOCAL_LLM_TEMPERATURE,
                top_p=LOCAL_LLM_TOP_P,
                do_sample=False,
                use_chat_template=True,
                max_input_tokens=LOCAL_LLM_MAX_INPUT_TOKENS,
            ),
        )
        synchronize_cuda()
        latency_ms = (time.perf_counter() - start) * 1000.0
        output = result.text.strip()
        return BenchmarkSample(
            name="base_generation",
            task=LLMTask.BASE.value,
            adapter=None,
            iteration=iteration,
            latency_ms=latency_ms,
            switch_latency_ms=None,
            memory_before=memory_before,
            memory_after=snapshot_memory(),
            output_chars=len(output),
            normalized_output=output,
            contract_ok=bool(output),
            contract_message="non-empty base output" if output else "base output was empty",
        )
    except Exception:
        return BenchmarkSample(
            name="base_generation",
            task=LLMTask.BASE.value,
            adapter=None,
            iteration=iteration,
            latency_ms=None,
            switch_latency_ms=None,
            memory_before=memory_before,
            memory_after=snapshot_memory(),
            output_chars=0,
            normalized_output="",
            contract_ok=False,
            contract_message="base generation failed",
            error=traceback.format_exc(),
        )


def run_adapter_sample(runtime: LocalCaiTIRuntime, case, iteration: int) -> BenchmarkSample:
    adapter = resolve_adapter(case.task)
    memory_before = snapshot_memory()
    switch_latency_ms: float | None = None
    try:
        synchronize_cuda()
        switch_start = time.perf_counter()
        runtime.model.set_adapter(case.task.value)
        synchronize_cuda()
        switch_latency_ms = (time.perf_counter() - switch_start) * 1000.0

        prompt = _format_prompt(case.system_prompt, case.payload)
        start = time.perf_counter()
        result = runtime.generate_adapter(
            case.task,
            prompt,
            GenerationConfig(
                max_new_tokens=case.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                use_chat_template=False,
                max_input_tokens=LOCAL_LLM_MAX_INPUT_TOKENS,
            ),
        )
        synchronize_cuda()
        latency_ms = (time.perf_counter() - start) * 1000.0
        normalized = case.normalize(result.text)
        contract_ok, contract_message = case.validate(result.text)
        return BenchmarkSample(
            name=case.name,
            task=case.task.value,
            adapter=adapter,
            iteration=iteration,
            latency_ms=latency_ms,
            switch_latency_ms=switch_latency_ms,
            memory_before=memory_before,
            memory_after=snapshot_memory(),
            output_chars=len(result.text),
            normalized_output=normalized,
            contract_ok=contract_ok,
            contract_message=contract_message,
        )
    except Exception:
        return BenchmarkSample(
            name=case.name,
            task=case.task.value,
            adapter=adapter,
            iteration=iteration,
            latency_ms=None,
            switch_latency_ms=switch_latency_ms,
            memory_before=memory_before,
            memory_after=snapshot_memory(),
            output_chars=0,
            normalized_output="",
            contract_ok=False,
            contract_message="adapter benchmark failed",
            error=traceback.format_exc(),
        )


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "data" / "results" / f"jetson_benchmark_{stamp}.json"


def build_report(
    args: argparse.Namespace,
    load_latency_ms: float,
    memory_before_load: dict[str, float | None],
    memory_after_load: dict[str, float | None],
    samples: list[BenchmarkSample],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "package_versions": package_versions(),
        "settings": asdict(runtime_settings()),
        "benchmark_args": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "include_base": not args.skip_base,
        },
        "load": {
            "latency_ms": load_latency_ms,
            "memory_before": memory_before_load,
            "memory_after": memory_after_load,
        },
        "summary": summarize_samples(samples),
        "samples": [asdict(sample) for sample in samples],
        "failures": [asdict(sample) for sample in samples if sample.error or not sample.contract_ok],
    }


def print_summary(report: dict[str, Any]) -> None:
    print("\n=== Load ===")
    print(f"latency_ms: {report['load']['latency_ms']:.1f}")
    before = report["load"]["memory_before"].get("process_rss_mb")
    after = report["load"]["memory_after"].get("process_rss_mb")
    if before is not None and after is not None:
        print(f"process_rss_mb: {before:.1f} -> {after:.1f}")

    print("\n=== Per-task latency summary ===")
    header = f"{'case':32} {'ok':>4} {'fail':>4} {'p50_ms':>10} {'p95_ms':>10} {'mean_ms':>10} {'switch_ms':>10}"
    print(header)
    print("-" * len(header))
    for name, stats in report["summary"].items():
        print(
            f"{name:32} "
            f"{stats['successes']:>4} "
            f"{stats['failures'] + stats['contract_failures']:>4} "
            f"{_fmt_ms(stats['latency_ms_p50']):>10} "
            f"{_fmt_ms(stats['latency_ms_p95']):>10} "
            f"{_fmt_ms(stats['latency_ms_mean']):>10} "
            f"{_fmt_ms(stats['switch_latency_ms_mean']):>10}"
        )

    failures = report["failures"]
    if failures:
        print("\n=== Failures ===")
        for failure in failures:
            print(f"- {failure['name']} iter={failure['iteration']}: {failure['contract_message']}")


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the local CaiTI runtime on Jetson.")
    parser.add_argument("--iterations", type=int, default=3, help="Measured iterations per case.")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup iterations per case.")
    parser.add_argument("--skip-base", action="store_true", help="Skip base-model generation benchmark.")
    parser.add_argument("--output", type=Path, default=None, help="JSON report path.")
    parser.add_argument("--list-cases", action="store_true", help="Print benchmark cases and exit without loading model.")
    parser.add_argument(
        "--ignore-missing-deps",
        action="store_true",
        help="Try running even if Python dependency probes fail.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = build_smoke_cases()
    if args.list_cases:
        if not args.skip_base:
            print("base_generation")
        for case in cases:
            print(case.name)
        return 0

    if args.iterations < 1:
        print("--iterations must be at least 1", file=sys.stderr)
        return 2
    if args.warmup < 0:
        print("--warmup must be non-negative", file=sys.stderr)
        return 2

    missing = missing_runtime_dependencies()
    if missing and not args.ignore_missing_deps:
        print("Cannot run benchmark because runtime dependencies are missing.")
        print("Missing packages: " + ", ".join(missing))
        return 2

    print("Loading local CaiTI runtime...")
    memory_before_load = snapshot_memory()
    start = time.perf_counter()
    runtime = LocalCaiTIRuntime(runtime_settings())
    synchronize_cuda()
    load_latency_ms = (time.perf_counter() - start) * 1000.0
    memory_after_load = snapshot_memory()
    reset_cuda_peak_memory()

    samples: list[BenchmarkSample] = []
    all_case_names = (["base_generation"] if not args.skip_base else []) + [case.name for case in cases]
    print(f"Running {len(all_case_names)} cases with warmup={args.warmup}, iterations={args.iterations}.")

    for iteration in range(args.warmup):
        if not args.skip_base:
            run_base_sample(runtime, iteration=-(iteration + 1))
        for case in cases:
            run_adapter_sample(runtime, case, iteration=-(iteration + 1))

    for iteration in range(1, args.iterations + 1):
        print(f"Measured iteration {iteration}/{args.iterations}")
        if not args.skip_base:
            samples.append(run_base_sample(runtime, iteration))
        for case in cases:
            samples.append(run_adapter_sample(runtime, case, iteration))

    report = build_report(args, load_latency_ms, memory_before_load, memory_after_load, samples)
    output_path = args.output or default_output_path()
    write_report(report, output_path)
    print_summary(report)
    print(f"\nWrote benchmark report: {output_path}")

    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
