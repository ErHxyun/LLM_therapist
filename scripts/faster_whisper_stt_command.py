"""Command-backend adapter for local Faster-Whisper STT.

Records a short mono WAV with arecord unless --input-wav is supplied, then
prints only the final transcript to stdout. Designed for CAITI_STT_COMMAND.
"""

from __future__ import annotations

import argparse
import audioop
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.voice.vad import build_vad, dbfs

DEVICE_OPEN_ERROR_SNIPPETS = (
    "audio open error",
    "no such file or directory",
    "no such device",
    "cannot get card index",
    "unknown pcm",
    "device or resource busy",
)


def str_to_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_arecord_command(
    output_file: str,
    seconds: float,
    sample_rate: int,
    channels: int,
    device: str,
) -> list[str]:
    duration_sec = max(1, int(math.ceil(float(seconds))))
    command = [
        "arecord",
        "-q",
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        "-d",
        str(duration_sec),
    ]
    if device:
        command.extend(["-D", device])
    command.append(output_file)
    return command


def build_arecord_raw_command(
    seconds: float,
    sample_rate: int,
    channels: int,
    device: str,
) -> list[str]:
    duration_sec = max(1, int(math.ceil(float(seconds))))
    command = [
        "arecord",
        "-q",
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        "-t",
        "raw",
        "-d",
        str(duration_sec),
    ]
    if device:
        command.extend(["-D", device])
    return command


def _capture_device_candidates(device: str) -> list[str]:
    normalized = str(device or "").strip()
    candidates: list[str] = []
    if normalized:
        candidates.append(normalized)
    for fallback in ("default", "pulse", ""):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _is_recoverable_device_error(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    return any(snippet in normalized for snippet in DEVICE_OPEN_ERROR_SNIPPETS)


def _report_device_retry(failed_device: str, next_device: str, error: str) -> None:
    failed_label = failed_device or "<default>"
    next_label = next_device or "<implicit-default>"
    print(
        f"arecord device {failed_label!r} failed ({error}). Retrying with {next_label!r}.",
        file=sys.stderr,
    )


def _wav_is_effectively_silent(
    wav_file: str,
    *,
    rms_threshold_dbfs: float = -95.0,
    peak_threshold_dbfs: float = -95.0,
) -> bool:
    try:
        metrics = analyze_wav(wav_file)
    except Exception:
        return False
    return (
        float(metrics.get("rms_dbfs", -120.0)) <= rms_threshold_dbfs
        and float(metrics.get("peak_dbfs", -120.0)) <= peak_threshold_dbfs
    )


def _dbfs(audio: bytes, sample_width: int = 2) -> float:
    return dbfs(audio, sample_width=sample_width)


def _write_wav(output_file: str, audio: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(output_file, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(audio)


def record_wav(
    output_file: str,
    seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
    device: str = "",
    timeout_sec: int = 30,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    candidates = _capture_device_candidates(device)
    last_error: RuntimeError | None = None
    for index, candidate in enumerate(candidates):
        command = build_arecord_command(output_file, seconds, sample_rate, channels, candidate)
        completed = runner(command, capture_output=True, text=True, timeout=timeout_sec)
        has_more_candidates = index + 1 < len(candidates)
        if completed.returncode == 0 and Path(output_file).exists() and Path(output_file).stat().st_size > 44:
            if has_more_candidates and _wav_is_effectively_silent(output_file):
                _report_device_retry(candidate, candidates[index + 1], "recorded audio was silent")
                continue
            return

        error = (completed.stderr or "").strip() or "Recorded WAV is empty or invalid."
        last_error = RuntimeError(error)
        if has_more_candidates and _is_recoverable_device_error(error):
            _report_device_retry(candidate, candidates[index + 1], error)
            continue
        raise last_error

    if last_error is not None:
        raise last_error
    raise RuntimeError("Recorded WAV is empty or invalid.")


def record_wav_auto_stop(
    output_file: str,
    max_seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
    device: str = "",
    vad_detector: str = "auto",
    vad_aggressiveness: int = 3,
    silence_threshold_dbfs: float = -45.0,
    silence_timeout_sec: float = 1.2,
    trailing_pad_sec: float = 0.4,
    min_speech_seconds: float = 0.25,
    min_record_seconds: float = 1.0,
    no_speech_timeout_sec: float = 5.0,
    chunk_ms: int = 30,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """
    Record from arecord until speech is followed by enough silence.

    This keeps live turns responsive: users no longer need to wait for a fixed
    recording window after they finish answering. If no speech is detected, it
    stops early so the normal empty-transcript retry can prompt again.
    """
    candidates = _capture_device_candidates(device)
    last_error: RuntimeError | None = None
    for index, candidate in enumerate(candidates):
        try:
            _record_wav_auto_stop_once(
                output_file,
                max_seconds=max_seconds,
                sample_rate=sample_rate,
                channels=channels,
                device=candidate,
                vad_detector=vad_detector,
                vad_aggressiveness=vad_aggressiveness,
                silence_threshold_dbfs=silence_threshold_dbfs,
                silence_timeout_sec=silence_timeout_sec,
                trailing_pad_sec=trailing_pad_sec,
                min_speech_seconds=min_speech_seconds,
                min_record_seconds=min_record_seconds,
                no_speech_timeout_sec=no_speech_timeout_sec,
                chunk_ms=chunk_ms,
                popen_factory=popen_factory,
                should_stop=should_stop,
            )
            has_more_candidates = index + 1 < len(candidates)
            if has_more_candidates and _wav_is_effectively_silent(output_file):
                _report_device_retry(candidate, candidates[index + 1], "recorded audio was silent")
                continue
            return
        except RuntimeError as exc:
            last_error = exc
            has_more_candidates = index + 1 < len(candidates)
            if has_more_candidates and _is_recoverable_device_error(str(exc)):
                _report_device_retry(candidate, candidates[index + 1], str(exc))
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Recorded WAV is empty or invalid.")


def _record_wav_auto_stop_once(
    output_file: str,
    max_seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
    device: str = "",
    vad_detector: str = "auto",
    vad_aggressiveness: int = 3,
    silence_threshold_dbfs: float = -45.0,
    silence_timeout_sec: float = 1.2,
    trailing_pad_sec: float = 0.4,
    min_speech_seconds: float = 0.25,
    min_record_seconds: float = 1.0,
    no_speech_timeout_sec: float = 5.0,
    chunk_ms: int = 30,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    command = build_arecord_raw_command(max_seconds, sample_rate, channels, device)
    process = popen_factory(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("arecord stdout pipe was not available.")

    bytes_per_second = sample_rate * channels * 2
    chunk_ms = int(max(10, chunk_ms))
    chunk_size = max(channels * 2, int(bytes_per_second * chunk_ms / 1000))
    vad = build_vad(
        vad_detector,
        sample_rate=sample_rate,
        channels=channels,
        chunk_ms=chunk_ms,
        aggressiveness=vad_aggressiveness,
        threshold_dbfs=silence_threshold_dbfs,
    )
    frames: list[bytes] = []
    recorded_sec = 0.0
    silence_after_speech_sec = 0.0
    speech_candidate_sec = 0.0
    speech_started = False
    stop_silence_sec = max(0.0, silence_timeout_sec) + max(0.0, trailing_pad_sec)
    started_at = time.monotonic()

    try:
        while recorded_sec < max_seconds:
            if should_stop is not None and should_stop():
                break
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            frames.append(chunk)
            chunk_sec = len(chunk) / bytes_per_second
            recorded_sec += chunk_sec

            if vad.is_speech(chunk):
                speech_candidate_sec += chunk_sec
                if speech_candidate_sec >= min_speech_seconds:
                    speech_started = True
                    silence_after_speech_sec = 0.0
            elif speech_started:
                silence_after_speech_sec += chunk_sec
            else:
                speech_candidate_sec = 0.0

            if not speech_started and recorded_sec >= no_speech_timeout_sec:
                break
            if (
                speech_started
                and recorded_sec >= min_record_seconds
                and silence_after_speech_sec >= stop_silence_sec
            ):
                break
            if time.monotonic() - started_at > max_seconds + 2:
                break
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    audio = b"".join(frames)
    if not audio:
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read().decode(errors="replace").strip()
        raise RuntimeError(stderr or "Recorded WAV is empty or invalid.")
    _write_wav(output_file, audio, sample_rate, channels)


def analyze_wav(wav_file: str) -> dict[str, float]:
    with wave.open(wav_file, "rb") as reader:
        sample_width = reader.getsampwidth()
        frame_rate = reader.getframerate()
        channels = reader.getnchannels()
        frames = reader.getnframes()
        audio = reader.readframes(frames)

    duration_sec = frames / frame_rate if frame_rate else 0.0
    if not audio or sample_width <= 0:
        return {
            "duration_sec": duration_sec,
            "channels": float(channels),
            "sample_rate": float(frame_rate),
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
        }

    max_amplitude = float((1 << (8 * sample_width - 1)) - 1)
    rms = float(audioop.rms(audio, sample_width))
    peak = float(audioop.max(audio, sample_width))

    def to_dbfs(value: float) -> float:
        if value <= 0 or max_amplitude <= 0:
            return -120.0
        return 20.0 * math.log10(value / max_amplitude)

    return {
        "duration_sec": duration_sec,
        "channels": float(channels),
        "sample_rate": float(frame_rate),
        "rms_dbfs": to_dbfs(rms),
        "peak_dbfs": to_dbfs(peak),
    }


def load_whisper_model(model: str, device: str = "cpu", compute_type: str = "int8"):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install faster-whisper, not openai-whisper."
        ) from exc

    return WhisperModel(model, device=device, compute_type=compute_type, num_workers=1)


def transcribe_wav_with_model(
    whisper,
    wav_file: str,
    beam_size: int = 5,
    best_of: int = 5,
    language: str = "en",
    initial_prompt: str = "",
    vad_filter: bool = True,
) -> str:
    segments, _info = whisper.transcribe(
        wav_file,
        language=language or None,
        beam_size=beam_size,
        best_of=best_of,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt or None,
        vad_filter=vad_filter,
        without_timestamps=True,
    )
    return " ".join(segment.text.strip() for segment in segments if segment.text).strip()


def transcribe_wav(
    wav_file: str,
    model: str,
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    best_of: int = 5,
    language: str = "en",
    initial_prompt: str = "",
    vad_filter: bool = True,
) -> str:
    whisper = load_whisper_model(model, device=device, compute_type=compute_type)
    return transcribe_wav_with_model(
        whisper,
        wav_file,
        beam_size=beam_size,
        best_of=best_of,
        language=language,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
    )


def listen_once(
    model: str,
    input_wav: str = "",
    record_seconds: float = 30.0,
    sample_rate: int = 16000,
    channels: int = 1,
    audio_device: str = "",
    stt_device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    best_of: int = 5,
    language: str = "en",
    initial_prompt: str = "",
    vad_filter: bool = True,
    auto_stop: bool = False,
    vad_detector: str = "auto",
    vad_aggressiveness: int = 3,
    silence_threshold_dbfs: float = -45.0,
    silence_timeout_sec: float = 1.2,
    trailing_pad_sec: float = 0.4,
    min_speech_seconds: float = 0.25,
    min_record_seconds: float = 1.0,
    no_speech_timeout_sec: float = 5.0,
    vad_chunk_ms: int = 30,
    save_wav: str = "",
    debug_audio: bool = False,
    timeout_sec: int = 30,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    if input_wav:
        if debug_audio:
            metrics = analyze_wav(input_wav)
            print(
                "audio "
                f"duration={metrics['duration_sec']:.2f}s "
                f"sample_rate={metrics['sample_rate']:.0f} "
                f"channels={metrics['channels']:.0f} "
                f"rms={metrics['rms_dbfs']:.1f}dBFS "
                f"peak={metrics['peak_dbfs']:.1f}dBFS",
                file=sys.stderr,
            )
        return transcribe_wav(
            input_wav,
            model,
            stt_device,
            compute_type,
            beam_size,
            best_of,
            language,
            initial_prompt,
            vad_filter,
        )

    if save_wav:
        wav_file = save_wav
        Path(wav_file).parent.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_file = tmp.name
        cleanup = True
    try:
        if auto_stop:
            record_wav_auto_stop(
                wav_file,
                max_seconds=record_seconds,
                sample_rate=sample_rate,
                channels=channels,
                device=audio_device,
                vad_detector=vad_detector,
                vad_aggressiveness=vad_aggressiveness,
                silence_threshold_dbfs=silence_threshold_dbfs,
                silence_timeout_sec=silence_timeout_sec,
                trailing_pad_sec=trailing_pad_sec,
                min_speech_seconds=min_speech_seconds,
                min_record_seconds=min_record_seconds,
                no_speech_timeout_sec=no_speech_timeout_sec,
                chunk_ms=vad_chunk_ms,
            )
        else:
            record_wav(
                wav_file,
                seconds=record_seconds,
                sample_rate=sample_rate,
                channels=channels,
                device=audio_device,
                timeout_sec=timeout_sec,
                runner=runner,
            )
        if debug_audio:
            metrics = analyze_wav(wav_file)
            print(
                "audio "
                f"duration={metrics['duration_sec']:.2f}s "
                f"sample_rate={metrics['sample_rate']:.0f} "
                f"channels={metrics['channels']:.0f} "
                f"rms={metrics['rms_dbfs']:.1f}dBFS "
                f"peak={metrics['peak_dbfs']:.1f}dBFS",
                file=sys.stderr,
            )
        return transcribe_wav(
            wav_file,
            model,
            stt_device,
            compute_type,
            beam_size,
            best_of,
            language,
            initial_prompt,
            vad_filter,
        )
    finally:
        if cleanup:
            try:
                os.remove(wav_file)
            except OSError:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio locally and print a Faster-Whisper transcript.")
    parser.add_argument("--model", default=os.environ.get("CAITI_WHISPER_MODEL", "base.en"))
    parser.add_argument("--input-wav", default="")
    parser.add_argument("--record-seconds", type=float, default=float(os.environ.get("CAITI_STT_RECORD_SECONDS", "30")))
    parser.add_argument("--sample-rate", type=int, default=int(os.environ.get("CAITI_STT_SAMPLE_RATE", "16000")))
    parser.add_argument("--channels", type=int, default=int(os.environ.get("CAITI_STT_CHANNELS", "1")))
    parser.add_argument("--audio-device", default=os.environ.get("CAITI_STT_AUDIO_DEVICE", ""))
    parser.add_argument("--stt-device", default=os.environ.get("CAITI_WHISPER_DEVICE", "cpu"))
    parser.add_argument("--compute-type", default=os.environ.get("CAITI_WHISPER_COMPUTE_TYPE", "int8"))
    parser.add_argument("--beam-size", type=int, default=int(os.environ.get("CAITI_WHISPER_BEAM_SIZE", "5")))
    parser.add_argument("--best-of", type=int, default=int(os.environ.get("CAITI_WHISPER_BEST_OF", "5")))
    parser.add_argument("--language", default=os.environ.get("CAITI_WHISPER_LANGUAGE", "en"))
    parser.add_argument(
        "--initial-prompt",
        default=os.environ.get(
            "CAITI_WHISPER_INITIAL_PROMPT",
            "The speaker is answering CaiTI daily functioning screening questions in English. "
            "Common words include doctor, therapist, case manager, medication, mood, sleep, "
            "appetite, hygiene, chores, work, school, family, alcohol, drugs, safety, and daily life.",
        ),
    )
    parser.set_defaults(vad_filter=str_to_bool(os.environ.get("CAITI_WHISPER_VAD_FILTER", "1")))
    parser.add_argument("--vad-filter", dest="vad_filter", action="store_true")
    parser.add_argument("--no-vad-filter", dest="vad_filter", action="store_false")
    parser.set_defaults(auto_stop=str_to_bool(os.environ.get("CAITI_STT_AUTO_STOP", "0")))
    parser.add_argument("--auto-stop", dest="auto_stop", action="store_true")
    parser.add_argument("--no-auto-stop", dest="auto_stop", action="store_false")
    parser.add_argument(
        "--vad-detector",
        choices=["auto", "webrtc", "energy"],
        default=os.environ.get("CAITI_STT_VAD_DETECTOR", "auto"),
        help="Voice activity detector for recording auto-stop.",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=int(os.environ.get("CAITI_STT_VAD_AGGRESSIVENESS", "3")),
        help="WebRTC VAD aggressiveness from 0 to 3.",
    )
    parser.add_argument(
        "--silence-threshold-dbfs",
        type=float,
        default=float(os.environ.get("CAITI_STT_SILENCE_THRESHOLD_DBFS", "-45")),
    )
    parser.add_argument(
        "--silence-timeout-sec",
        type=float,
        default=float(os.environ.get("CAITI_STT_SILENCE_TIMEOUT_SEC", "1.2")),
    )
    parser.add_argument(
        "--trailing-pad-sec",
        type=float,
        default=float(os.environ.get("CAITI_STT_TRAILING_PAD_SEC", "0.4")),
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=float(os.environ.get("CAITI_STT_MIN_SPEECH_SECONDS", "0.25")),
    )
    parser.add_argument(
        "--min-record-seconds",
        type=float,
        default=float(os.environ.get("CAITI_STT_MIN_RECORD_SECONDS", "1.0")),
    )
    parser.add_argument(
        "--no-speech-timeout-sec",
        type=float,
        default=float(os.environ.get("CAITI_STT_NO_SPEECH_TIMEOUT_SEC", "5.0")),
    )
    parser.add_argument(
        "--vad-chunk-ms",
        type=int,
        default=int(os.environ.get("CAITI_STT_VAD_CHUNK_MS", "30")),
    )
    parser.add_argument("--save-wav", default=os.environ.get("CAITI_STT_SAVE_WAV", ""))
    parser.add_argument(
        "--debug-audio",
        action="store_true",
        default=str_to_bool(os.environ.get("CAITI_STT_DEBUG_AUDIO", "0")),
    )
    parser.add_argument("--timeout-sec", type=int, default=int(os.environ.get("CAITI_STT_TIMEOUT_SEC", "30")))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        transcript = listen_once(
            model=args.model,
            input_wav=args.input_wav,
            record_seconds=args.record_seconds,
            sample_rate=args.sample_rate,
            channels=args.channels,
            audio_device=args.audio_device,
            stt_device=args.stt_device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            best_of=args.best_of,
            language=args.language,
            initial_prompt=args.initial_prompt,
            vad_filter=args.vad_filter,
            auto_stop=args.auto_stop,
            vad_detector=args.vad_detector,
            vad_aggressiveness=args.vad_aggressiveness,
            silence_threshold_dbfs=args.silence_threshold_dbfs,
            silence_timeout_sec=args.silence_timeout_sec,
            trailing_pad_sec=args.trailing_pad_sec,
            min_speech_seconds=args.min_speech_seconds,
            min_record_seconds=args.min_record_seconds,
            no_speech_timeout_sec=args.no_speech_timeout_sec,
            vad_chunk_ms=args.vad_chunk_ms,
            save_wav=args.save_wav,
            debug_audio=args.debug_audio,
            timeout_sec=args.timeout_sec,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if transcript:
        print(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
