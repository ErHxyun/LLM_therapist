"""Command-backend adapter for local Piper TTS.

Reads text from stdin, synthesizes one temporary WAV with Piper, then plays it
with a local audio player. Designed for CAITI_TTS_COMMAND.
"""

from __future__ import annotations

import argparse
import math
import hashlib
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import wave
from contextlib import closing
from pathlib import Path
from typing import Callable, Sequence

PLAYBACK_START_MARKER = "__CAITI_TTS_PLAYBACK_START__"
PLAYBACK_END_MARKER = "__CAITI_TTS_PLAYBACK_END__"
ESPEAK_FALLBACK = "espeak-ng"
REPO_ROOT = Path(__file__).resolve().parents[1]
CACHED_FALLBACK_WAV = REPO_ROOT / "assets" / "audio" / "tts_fallback.wav"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache" / "tts"
TARGET_APLAY_SAMPLE_RATE = 48000
TARGET_APLAY_CHANNELS = 2


def build_piper_command(
    executable: str,
    model_path: str,
    output_file: str,
    length_scale: float,
    sentence_silence: float,
) -> list[str]:
    return [
        executable,
        "--model",
        model_path,
        "--length_scale",
        str(length_scale),
        "--sentence_silence",
        str(sentence_silence),
        "--output_file",
        output_file,
    ]


def build_player_command(player: str, wav_file: str) -> list[str]:
    normalized = str(player or "").strip()
    if not normalized:
        normalized = "aplay"
    if normalized == "aplay":
        return ["aplay", "-q", wav_file]
    if normalized == "paplay":
        return ["paplay", wav_file]
    parts = shlex.split(normalized)
    if not parts:
        return ["aplay", "-q", wav_file]
    return [*parts, wav_file]


def normalize_speech_text(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    normalized = " ".join(lines)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _should_prepare_aplay_wav(player: str) -> bool:
    try:
        parts = shlex.split(str(player or "").strip())
    except ValueError:
        return False
    return bool(parts) and Path(parts[0]).name == "aplay"


def _prepare_aplay_wav(
    wav_file: str,
    target_rate: int = TARGET_APLAY_SAMPLE_RATE,
    target_channels: int = TARGET_APLAY_CHANNELS,
) -> str:
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except Exception:
        return wav_file

    try:
        with closing(wave.open(wav_file, "rb")) as wav_reader:
            input_rate = int(wav_reader.getframerate())
            input_channels = int(wav_reader.getnchannels())
            sample_width = int(wav_reader.getsampwidth())
    except Exception:
        return wav_file

    if input_rate == target_rate and input_channels == target_channels and sample_width == 2:
        return wav_file

    try:
        data, rate = sf.read(wav_file, always_2d=True, dtype="float32")
        if int(rate) != target_rate:
            divisor = math.gcd(int(rate), int(target_rate)) or 1
            data = resample_poly(data, target_rate // divisor, int(rate) // divisor, axis=0)
        if data.shape[1] == 1 and target_channels == 2:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > target_channels:
            data = data[:, :target_channels]
        elif data.shape[1] < target_channels:
            data = np.pad(data, ((0, 0), (0, target_channels - data.shape[1])), mode="edge")
        data = np.clip(data, -1.0, 1.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted:
            converted_path = converted.name
        sf.write(converted_path, data, target_rate, subtype="PCM_16")
        return converted_path
    except Exception:
        return wav_file


def validate_piper_voice(model_path: str) -> None:
    if not model_path:
        raise ValueError("Piper model path is required.")
    model = Path(model_path)
    if not model.exists():
        raise FileNotFoundError(f"Piper model not found: {model_path}")
    if model.stat().st_size == 0:
        raise RuntimeError(f"Piper model is empty: {model_path}")

    config = Path(f"{model_path}.json")
    if not config.exists():
        raise FileNotFoundError(f"Piper model config not found: {config}")
    if config.stat().st_size == 0:
        raise RuntimeError(f"Piper model config is empty: {config}")
    try:
        json.loads(config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Piper model config is invalid JSON: {config}") from exc


def build_espeak_command(executable: str, text: str, output_file: str) -> list[str]:
    return [executable, "-v", "en-us", "-s", "140", "-w", output_file, text]


def synthesize_espeak_wav(
    text: str,
    output_file: str,
    executable: str = ESPEAK_FALLBACK,
    timeout_sec: int = 60,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    if not executable or shutil.which(executable) is None:
        return False
    completed = runner(
        build_espeak_command(executable, text, output_file),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return completed.returncode == 0 and _is_valid_wav(output_file)


def copy_cached_fallback_wav(output_file: str, cached_fallback_wav: str | os.PathLike[str] | None = None) -> bool:
    source = Path(cached_fallback_wav) if cached_fallback_wav else CACHED_FALLBACK_WAV
    if not source.exists() or source.stat().st_size <= 44:
        return False
    shutil.copyfile(source, output_file)
    return _is_valid_wav(output_file)


def _is_valid_wav(wav_file: str) -> bool:
    return Path(wav_file).exists() and Path(wav_file).stat().st_size > 44


def _file_fingerprint(path: Path) -> dict[str, str | int | None]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "size": None, "mtime_ns": None}
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_cache_key(
    text: str,
    model_path: str,
    executable: str,
    length_scale: float,
    sentence_silence: float,
) -> str:
    model = Path(model_path).expanduser().resolve()
    payload = {
        "text": text,
        "model": _file_fingerprint(model),
        "model_config": _file_fingerprint(Path(f"{model}.json")),
        "executable": executable,
        "length_scale": length_scale,
        "sentence_silence": sentence_silence,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_wav_path(
    cache_dir: str | os.PathLike[str] | None,
    text: str,
    model_path: str,
    executable: str,
    length_scale: float,
    sentence_silence: float,
) -> Path | None:
    if not cache_dir:
        return None
    key = build_cache_key(text, model_path, executable, length_scale, sentence_silence)
    return Path(cache_dir).expanduser() / f"{key}.wav"


def store_wav_in_cache(source_wav: str, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    shutil.copyfile(source_wav, tmp_path)
    os.replace(tmp_path, cache_path)


def speak_text(
    text: str,
    model_path: str,
    executable: str = "piper",
    player: str = "aplay",
    length_scale: float = 0.8,
    sentence_silence: float = 0.4,
    timeout_sec: int = 60,
    emit_playback_markers: bool = False,
    fallback_executable: str = ESPEAK_FALLBACK,
    cached_fallback_wav: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    text = normalize_speech_text(text)
    if not text:
        return

    cache_path = cache_wav_path(cache_dir, text, model_path, executable, length_scale, sentence_silence)
    cleanup = False
    if cache_path is not None and _is_valid_wav(str(cache_path)):
        wav_file = str(cache_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_file = tmp.name
        cleanup = True

    try:
        piper_error: Exception | None = None
        if cleanup:
            try:
                validate_piper_voice(model_path)
                piper_cmd = build_piper_command(
                    executable,
                    model_path,
                    wav_file,
                    length_scale,
                    sentence_silence,
                )
                completed = runner(
                    piper_cmd,
                    input=text,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if completed.returncode != 0:
                    raise RuntimeError((completed.stderr or "").strip() or "Piper TTS failed.")
                if not _is_valid_wav(wav_file):
                    raise RuntimeError("Piper produced an empty or invalid WAV file.")
                if cache_path is not None:
                    store_wav_in_cache(wav_file, cache_path)
            except Exception as exc:
                piper_error = exc
                if not synthesize_espeak_wav(
                    text,
                    wav_file,
                    executable=fallback_executable,
                    timeout_sec=timeout_sec,
                    runner=runner,
                ) and not copy_cached_fallback_wav(wav_file, cached_fallback_wav):
                    raise RuntimeError(f"Piper TTS failed and no fallback audio is available: {exc}") from exc

        playback_wav = wav_file
        playback_cleanup = False
        if _should_prepare_aplay_wav(player):
            prepared_wav = _prepare_aplay_wav(wav_file)
            if prepared_wav != wav_file:
                playback_wav = prepared_wav
                playback_cleanup = True
        play_cmd = build_player_command(player, playback_wav)
        if emit_playback_markers:
            print(PLAYBACK_START_MARKER, file=sys.stderr, flush=True)
        try:
            completed = runner(
                play_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        finally:
            if emit_playback_markers:
                print(PLAYBACK_END_MARKER, file=sys.stderr, flush=True)
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or "").strip() or "Audio playback failed.")
        if piper_error is not None:
            print(f"Piper TTS fallback used: {piper_error}", file=sys.stderr)
        if playback_cleanup:
            try:
                os.remove(playback_wav)
            except OSError:
                pass
    finally:
        if cleanup:
            try:
                os.remove(wav_file)
            except OSError:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read stdin, synthesize with Piper, and play WAV locally.")
    parser.add_argument("--model", default=os.environ.get("CAITI_PIPER_MODEL", ""))
    parser.add_argument("--executable", default=os.environ.get("CAITI_PIPER_EXECUTABLE", "piper"))
    parser.add_argument("--player", default=os.environ.get("CAITI_AUDIO_PLAYER", "aplay"))
    parser.add_argument("--length-scale", type=float, default=float(os.environ.get("CAITI_PIPER_LENGTH_SCALE", "0.8")))
    parser.add_argument(
        "--sentence-silence",
        type=float,
        default=float(os.environ.get("CAITI_PIPER_SENTENCE_SILENCE", "0.4")),
    )
    parser.add_argument("--timeout-sec", type=int, default=int(os.environ.get("CAITI_TTS_TIMEOUT_SEC", "60")))
    parser.add_argument("--fallback-executable", default=os.environ.get("CAITI_TTS_FALLBACK_EXECUTABLE", ESPEAK_FALLBACK))
    parser.add_argument("--cache-dir", default=os.environ.get("CAITI_TTS_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument(
        "--playback-markers",
        action="store_true",
        default=os.environ.get("CAITI_TTS_PLAYBACK_MARKERS", "0").strip().lower() in {"1", "true", "yes", "on"},
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        speak_text(
            sys.stdin.read(),
            model_path=args.model,
            executable=args.executable,
            player=args.player,
            length_scale=args.length_scale,
            sentence_silence=args.sentence_silence,
            timeout_sec=args.timeout_sec,
            emit_playback_markers=args.playback_markers,
            fallback_executable=args.fallback_executable,
            cache_dir=None if args.no_cache else args.cache_dir,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
