"""Benchmark Faster-Whisper STT on fixed local WAV files.

This script does not record audio and does not call the CaiTI session pipeline.
It reuses the same Faster-Whisper helpers as the voice runtime so STT settings
can be compared safely on a fixed audio set.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import faster_whisper_stt_command as whisper_stt  # noqa: E402
from src.utils import config_loader  # noqa: E402


AUDIO_EXTENSIONS = {".wav", ".wave"}


@dataclass(frozen=True)
class STTBenchmarkSample:
    audio_path: str
    audio_name: str
    model: str
    device: str
    compute_type: str
    beam_size: int
    best_of: int
    vad_filter: bool
    prompt_label: str
    iteration: int
    model_load_sec: float | None
    audio_duration_sec: float | None
    sample_rate: int | None
    channels: int | None
    rms_dbfs: float | None
    peak_dbfs: float | None
    transcribe_sec: float | None
    realtime_factor: float | None
    transcript_chars: int
    transcript: str
    error: str | None = None


def default_output_path(suffix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "data" / "results" / f"stt_benchmark_{stamp}.{suffix}"


def collect_audio_files(audio_dir: Path | None, audio_files: Sequence[Path], recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    if audio_dir:
        pattern = "**/*" if recursive else "*"
        for path in audio_dir.glob(pattern):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(path)
    files.extend(audio_files)
    unique = {path.resolve(): path for path in files if path.suffix.lower() in AUDIO_EXTENSIONS}
    return [unique[key] for key in sorted(unique)]


def vad_filter_values(mode: str, current_value: bool) -> list[bool]:
    if mode == "current":
        return [current_value]
    if mode == "on":
        return [True]
    if mode == "off":
        return [False]
    if mode == "both":
        return [True, False]
    raise ValueError(f"Unsupported vad filter mode: {mode}")


def parse_positive_int_values(values: Sequence[str], default_value: int) -> list[int]:
    raw_values = list(values) or [str(default_value)]
    parsed: list[int] = []
    for raw_value in raw_values:
        for part in str(raw_value).split(","):
            part = part.strip()
            if not part:
                continue
            value = int(part)
            if value < 1:
                raise ValueError("values must be >= 1")
            if value not in parsed:
                parsed.append(value)
    return parsed


def prompt_variants(mode: str, default_prompt: str) -> list[tuple[str, str]]:
    if mode == "current":
        return [("current", default_prompt)]
    if mode == "none":
        return [("none", "")]
    if mode == "both":
        return [("current", default_prompt), ("none", "")]
    raise ValueError(f"Unsupported prompt mode: {mode}")


def _round(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _audio_metrics(path: Path) -> dict[str, float | int | None]:
    metrics = whisper_stt.analyze_wav(str(path))
    return {
        "audio_duration_sec": _round(float(metrics.get("duration_sec", 0.0))),
        "sample_rate": int(metrics.get("sample_rate", 0)),
        "channels": int(metrics.get("channels", 0)),
        "rms_dbfs": _round(float(metrics.get("rms_dbfs", -120.0)), 1),
        "peak_dbfs": _round(float(metrics.get("peak_dbfs", -120.0)), 1),
    }


def run_benchmark(
    audio_paths: Sequence[Path],
    models: Sequence[str],
    device: str,
    compute_type: str,
    beam_sizes: Sequence[int],
    best_of_values: Sequence[int],
    language: str,
    prompts: Sequence[tuple[str, str]],
    vad_filters: Sequence[bool],
    iterations: int,
) -> list[STTBenchmarkSample]:
    samples: list[STTBenchmarkSample] = []
    audio_info = {path: _audio_metrics(path) for path in audio_paths}

    for model_name in models:
        load_started_at = time.monotonic()
        model = None
        load_error: str | None = None
        try:
            model = whisper_stt.load_whisper_model(model_name, device=device, compute_type=compute_type)
            model_load_sec = _round(time.monotonic() - load_started_at)
        except Exception:
            model_load_sec = _round(time.monotonic() - load_started_at)
            load_error = traceback.format_exc(limit=5)

        for vad_filter in vad_filters:
            for beam_size in beam_sizes:
                for best_of in best_of_values:
                    for prompt_label, initial_prompt in prompts:
                        for path in audio_paths:
                            metrics = audio_info[path]
                            for iteration in range(1, iterations + 1):
                                if load_error is not None or model is None:
                                    samples.append(
                                        STTBenchmarkSample(
                                            audio_path=str(path),
                                            audio_name=path.name,
                                            model=model_name,
                                            device=device,
                                            compute_type=compute_type,
                                            beam_size=beam_size,
                                            best_of=best_of,
                                            vad_filter=vad_filter,
                                            prompt_label=prompt_label,
                                            iteration=iteration,
                                            model_load_sec=model_load_sec,
                                            audio_duration_sec=metrics["audio_duration_sec"],
                                            sample_rate=metrics["sample_rate"],
                                            channels=metrics["channels"],
                                            rms_dbfs=metrics["rms_dbfs"],
                                            peak_dbfs=metrics["peak_dbfs"],
                                            transcribe_sec=None,
                                            realtime_factor=None,
                                            transcript_chars=0,
                                            transcript="",
                                            error=load_error,
                                        )
                                    )
                                    continue

                                started_at = time.monotonic()
                                transcript = ""
                                error = None
                                try:
                                    transcript = whisper_stt.transcribe_wav_with_model(
                                        model,
                                        str(path),
                                        beam_size=beam_size,
                                        best_of=best_of,
                                        language=language,
                                        initial_prompt=initial_prompt,
                                        vad_filter=vad_filter,
                                    ).strip()
                                except Exception:
                                    error = traceback.format_exc(limit=5)
                                transcribe_sec = _round(time.monotonic() - started_at)
                                audio_duration = metrics["audio_duration_sec"]
                                realtime_factor = None
                                if audio_duration and transcribe_sec is not None:
                                    realtime_factor = _round(transcribe_sec / float(audio_duration))

                                samples.append(
                                    STTBenchmarkSample(
                                        audio_path=str(path),
                                        audio_name=path.name,
                                        model=model_name,
                                        device=device,
                                        compute_type=compute_type,
                                        beam_size=beam_size,
                                        best_of=best_of,
                                        vad_filter=vad_filter,
                                        prompt_label=prompt_label,
                                        iteration=iteration,
                                        model_load_sec=model_load_sec,
                                        audio_duration_sec=metrics["audio_duration_sec"],
                                        sample_rate=metrics["sample_rate"],
                                        channels=metrics["channels"],
                                        rms_dbfs=metrics["rms_dbfs"],
                                        peak_dbfs=metrics["peak_dbfs"],
                                        transcribe_sec=transcribe_sec,
                                        realtime_factor=realtime_factor,
                                        transcript_chars=len(transcript),
                                        transcript=transcript,
                                        error=error,
                                    )
                                )
    return samples


def write_jsonl(samples: Iterable[STTBenchmarkSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")


def write_csv(samples: Sequence[STTBenchmarkSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(samples[0]).keys()) if samples else list(STTBenchmarkSample.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def summarize(samples: Sequence[STTBenchmarkSample]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, bool, int, int, str], list[STTBenchmarkSample]] = {}
    for sample in samples:
        grouped.setdefault(
            (sample.model, sample.vad_filter, sample.beam_size, sample.best_of, sample.prompt_label), []
        ).append(sample)

    rows: list[dict[str, object]] = []
    for (model, vad_filter, beam_size, best_of, prompt_label), group in sorted(grouped.items()):
        successful = [sample for sample in group if sample.transcribe_sec is not None and sample.error is None]
        latencies = [float(sample.transcribe_sec) for sample in successful if sample.transcribe_sec is not None]
        realtime = [float(sample.realtime_factor) for sample in successful if sample.realtime_factor is not None]
        rows.append(
            {
                "model": model,
                "vad_filter": vad_filter,
                "beam_size": beam_size,
                "best_of": best_of,
                "prompt_label": prompt_label,
                "count": len(group),
                "successes": len(successful),
                "failures": len(group) - len(successful),
                "transcribe_sec_mean": round(statistics.mean(latencies), 3) if latencies else None,
                "transcribe_sec_median": round(statistics.median(latencies), 3) if latencies else None,
                "realtime_factor_mean": round(statistics.mean(realtime), 3) if realtime else None,
            }
        )
    return rows


def print_summary(samples: Sequence[STTBenchmarkSample]) -> None:
    print("STT benchmark summary")
    for row in summarize(samples):
        print(
            f"- model={row['model']} vad_filter={row['vad_filter']} "
            f"beam={row['beam_size']} best_of={row['best_of']} prompt={row['prompt_label']} "
            f"successes={row['successes']}/{row['count']} "
            f"mean={row['transcribe_sec_mean']}s "
            f"median={row['transcribe_sec_median']}s "
            f"rtf={row['realtime_factor_mean']}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Faster-Whisper STT on fixed WAV files.")
    parser.add_argument("--audio-dir", type=Path, default=REPO_ROOT / "data" / "benchmark" / "stt")
    parser.add_argument("--audio-file", type=Path, action="append", default=[])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--model", action="append", default=[], help="Model to test. Repeat for multiple models.")
    parser.add_argument("--stt-device", default=config_loader.VOICE_STT_DEVICE)
    parser.add_argument("--compute-type", default=config_loader.VOICE_STT_COMPUTE_TYPE)
    parser.add_argument(
        "--beam-size",
        action="append",
        default=[],
        help="Beam size to test. Repeat or pass comma-separated values, e.g. --beam-size 1,2.",
    )
    parser.add_argument(
        "--best-of",
        action="append",
        default=[],
        help="Best-of value to test. Repeat or pass comma-separated values, e.g. --best-of 1,2.",
    )
    parser.add_argument("--language", default=config_loader.VOICE_STT_LANGUAGE)
    parser.add_argument("--initial-prompt", default=config_loader.VOICE_STT_INITIAL_PROMPT)
    parser.add_argument(
        "--prompt-mode",
        choices=["current", "none", "both"],
        default="current",
        help="Use the configured initial prompt, no prompt, or compare both.",
    )
    parser.add_argument(
        "--vad-filter-mode",
        choices=["current", "on", "off", "both"],
        default="current",
        help="Use config value, force on/off, or test both Faster-Whisper internal VAD settings.",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--list-audio", action="store_true", help="List resolved audio files and exit.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    models = args.model or [config_loader.VOICE_STT_WHISPER_MODEL]
    audio_paths = collect_audio_files(args.audio_dir, args.audio_file, recursive=args.recursive)

    if args.list_audio:
        for path in audio_paths:
            print(path)
        return 0

    if not audio_paths:
        print(
            "No WAV files found. Add fixed STT benchmark audio under "
            f"{args.audio_dir} or pass --audio-file.",
            file=sys.stderr,
        )
        return 1
    if args.iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 1

    try:
        beam_sizes = parse_positive_int_values(args.beam_size, config_loader.VOICE_STT_BEAM_SIZE)
        best_of_values = parse_positive_int_values(args.best_of, config_loader.VOICE_STT_BEST_OF)
        vad_filters = vad_filter_values(args.vad_filter_mode, config_loader.VOICE_STT_VAD_FILTER)
        prompts = prompt_variants(args.prompt_mode, args.initial_prompt)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    samples = run_benchmark(
        audio_paths=audio_paths,
        models=models,
        device=args.stt_device,
        compute_type=args.compute_type,
        beam_sizes=beam_sizes,
        best_of_values=best_of_values,
        language=args.language,
        prompts=prompts,
        vad_filters=vad_filters,
        iterations=args.iterations,
    )

    output_jsonl = args.output_jsonl or default_output_path("jsonl")
    output_csv = args.output_csv or output_jsonl.with_suffix(".csv")
    write_jsonl(samples, output_jsonl)
    write_csv(samples, output_csv)
    print_summary(samples)
    print(f"Wrote JSONL: {output_jsonl}")
    print(f"Wrote CSV: {output_csv}")

    return 1 if any(sample.error for sample in samples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
