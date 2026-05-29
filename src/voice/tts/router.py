from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts import piper_tts_command
from src.utils import config_loader
from src.utils.log_util import get_logger
from src.voice.tts.command import CommandTTS, ConsoleTTS
from src.voice.tts.piper import PersistentPiperTTS

logger = get_logger("TTSRouter")


@dataclass(frozen=True)
class TTSRouteSettings:
    role: str
    backend: str
    command: str = ""
    timeout_sec: int = 60
    fallback_to_primary: bool = False
    strict: bool = False


def build_primary_tts():
    return build_tts_from_settings(
        TTSRouteSettings(
            role="primary",
            backend=config_loader.VOICE_TTS_BACKEND,
            command=config_loader.VOICE_TTS_COMMAND,
            timeout_sec=config_loader.VOICE_TTS_TIMEOUT_SEC,
            strict=True,
        )
    )


def build_role_tts(role: str, primary_tts: Any = None):
    normalized = str(role or "primary").strip().lower()
    if normalized in {"primary", "main", "cbt"}:
        return primary_tts if primary_tts is not None else build_primary_tts()
    if normalized == "intermission":
        return build_tts_from_settings(
            TTSRouteSettings(
                role="intermission",
                backend=config_loader.INTERMISSION_TTS_BACKEND,
                command=config_loader.INTERMISSION_TTS_COMMAND,
                timeout_sec=config_loader.VOICE_TTS_TIMEOUT_SEC,
                fallback_to_primary=config_loader.INTERMISSION_FALLBACK_TO_PRIMARY_TTS,
            ),
            primary_tts=primary_tts,
        )
    raise ValueError(f"Unsupported TTS role: {role}")


def build_tts_from_settings(settings: TTSRouteSettings, primary_tts: Any = None):
    backend = str(settings.backend or "").strip().lower()
    if backend == "primary":
        if primary_tts is not None:
            return primary_tts
        return _handle_unavailable(settings, "primary TTS was requested but is not available", primary_tts)
    if backend == "console":
        return ConsoleTTS()
    if backend == "command":
        command = str(settings.command or "").strip()
        if not command:
            if settings.strict:
                return CommandTTS(command, timeout_sec=settings.timeout_sec)
            return _handle_unavailable(settings, f"{settings.role} TTS command is empty", primary_tts)
        unavailable = _piper_command_unavailable_reason(command)
        if unavailable and not settings.strict:
            return _handle_unavailable(settings, unavailable, primary_tts)
        if unavailable:
            logger.warning("%s; command fallback may be used at playback time.", unavailable)
        return CommandTTS(command, timeout_sec=settings.timeout_sec)
    if backend in {"piper", "persistent_piper", "persistent-piper"}:
        unavailable = _piper_command_unavailable_reason(str(settings.command or ""))
        if unavailable:
            if settings.strict:
                raise RuntimeError(unavailable)
            return _handle_unavailable(settings, unavailable, primary_tts)
        return _build_persistent_piper_tts(settings)
    if settings.strict:
        raise ValueError(f"Unsupported TTS backend: {settings.backend}")
    return _handle_unavailable(settings, f"unsupported {settings.role} TTS backend: {settings.backend!r}", primary_tts)


def _handle_unavailable(settings: TTSRouteSettings, reason: str, primary_tts: Any = None):
    if settings.fallback_to_primary and primary_tts is not None:
        logger.warning("%s; using primary TTS.", reason)
        return primary_tts
    logger.warning("%s; %s TTS disabled.", reason, settings.role)
    return None


def _piper_command_unavailable_reason(command: str) -> str:
    if "piper_tts_command.py" not in command:
        return ""
    model_path = extract_command_option(command, "--model")
    if not model_path:
        return "Piper TTS model path is empty"
    try:
        piper_tts_command.validate_piper_voice(str(_resolve_path(model_path)))
    except Exception as exc:
        return f"Piper TTS voice is unavailable: {exc}"
    return ""


def _build_persistent_piper_tts(settings: TTSRouteSettings) -> PersistentPiperTTS:
    command = str(settings.command or "")
    model_path = extract_command_option(command, "--model")
    player = extract_command_option(command, "--player") or "aplay"
    length_scale = _extract_float_option(command, "--length-scale", 0.8)
    sentence_silence = _extract_float_option(command, "--sentence-silence", 0.4)
    cache_dir = extract_command_option(command, "--cache-dir") or str(piper_tts_command.DEFAULT_CACHE_DIR)
    if _command_has_flag(command, "--no-cache"):
        cache_dir = ""
    return PersistentPiperTTS(
        model_path=str(_resolve_path(model_path)),
        player=player,
        length_scale=length_scale,
        sentence_silence=sentence_silence,
        timeout_sec=settings.timeout_sec,
        cache_dir=cache_dir or None,
        use_cuda=_command_has_flag(command, "--cuda"),
    )


def extract_command_option(command: str, option: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return ""
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        if part.startswith(option + "="):
            return part.split("=", 1)[1]
    return ""


def _extract_float_option(command: str, option: str, default: float) -> float:
    value = extract_command_option(command, option)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring invalid %s value for persistent Piper TTS: %r", option, value)
        return default


def _command_has_flag(command: str, flag: str) -> bool:
    try:
        return flag in shlex.split(command)
    except ValueError:
        return False


def _resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate
