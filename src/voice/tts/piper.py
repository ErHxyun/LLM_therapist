from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from scripts import piper_tts_command
from src.utils.log_util import get_logger
from src.voice.exceptions import VoiceInterrupted

logger = get_logger("VoiceTTS")


@dataclass
class PersistentPiperTTS:
    model_path: str
    player: str = "aplay"
    length_scale: float = 0.8
    sentence_silence: float = 0.4
    timeout_sec: int = 60
    cache_dir: str | os.PathLike[str] | None = piper_tts_command.DEFAULT_CACHE_DIR
    use_cuda: bool = False
    popen_factory: Callable = subprocess.Popen
    synthesis_config_factory: Callable[[float], object] | None = None
    _playback_status_hook: Callable[[bool], None] | None = field(default=None, init=False, repr=False)
    _interrupt_check: Callable[[], bool] | None = field(default=None, init=False, repr=False)
    _voice: object | None = field(default=None, init=False, repr=False)
    _voice_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    last_timing: dict[str, float | int | bool] = field(default_factory=dict, init=False)

    def set_playback_status_hook(self, hook: Callable[[bool], None] | None) -> None:
        self._playback_status_hook = hook

    def set_interrupt_check(self, checker: Callable[[], bool] | None) -> None:
        self._interrupt_check = checker

    def _should_interrupt(self) -> bool:
        return bool(self._interrupt_check is not None and self._interrupt_check())

    def warm_up(self) -> None:
        self._get_voice()

    def _get_voice(self):
        if self._voice is not None:
            return self._voice
        with self._voice_lock:
            if self._voice is None:
                if self._should_interrupt():
                    raise VoiceInterrupted("TTS interrupted before Piper model load.")
                logger.info("Loading persistent Piper voice model %s", self.model_path)
                try:
                    from piper import PiperVoice
                except ImportError as exc:
                    raise RuntimeError("Python package piper is not installed.") from exc
                piper_tts_command.validate_piper_voice(self.model_path)
                self._voice = PiperVoice.load(self.model_path, use_cuda=self.use_cuda)
                logger.info("Persistent Piper voice model ready: %s", self.model_path)
                logger.info("Persistent Piper playback player configured: %s", self.player)
        return self._voice

    def speak(self, text: str) -> None:
        chunk = str(text or "").strip()
        if not chunk:
            return
        if self._should_interrupt():
            raise VoiceInterrupted("TTS interrupted before synthesis.")

        started_at = time.monotonic()
        wav_file, cleanup, cache_hit = self._prepare_wav(chunk)
        prepare_finished_at = time.monotonic()
        try:
            if self._should_interrupt():
                raise VoiceInterrupted("TTS interrupted before playback.")
            self._play_wav(wav_file)
        finally:
            if cleanup:
                try:
                    os.remove(wav_file)
                except OSError:
                    pass
        finished_at = time.monotonic()
        self.last_timing = {
            "total_duration_sec": round(finished_at - started_at, 3),
            "prepare_duration_sec": round(prepare_finished_at - started_at, 3),
            "playback_duration_sec": round(finished_at - prepare_finished_at, 3),
            "text_length": len(chunk),
            "cache_hit": cache_hit,
            "used_playback_markers": self._playback_status_hook is not None,
        }
        logger.info(
            "TTS timing total=%.3fs prepare=%.3fs playback=%.3fs text_length=%s markers=%s cache_hit=%s backend=persistent_piper",
            self.last_timing["total_duration_sec"],
            self.last_timing["prepare_duration_sec"],
            self.last_timing["playback_duration_sec"],
            len(chunk),
            self._playback_status_hook is not None,
            cache_hit,
        )
        logger.info("Persistent Piper TTS spoke block length=%s", len(chunk))

    def speak_stream(self, text: str) -> None:
        self.speak(text)

    def _prepare_wav(self, text: str) -> tuple[str, bool, bool]:
        cache_path = piper_tts_command.cache_wav_path(
            self.cache_dir,
            text,
            self.model_path,
            "persistent_piper",
            self.length_scale,
            self.sentence_silence,
        )
        if cache_path is not None and piper_tts_command._is_valid_wav(str(cache_path)):
            return str(cache_path), False, True

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_file = tmp.name
        try:
            self._synthesize_wav(text, wav_file)
            if cache_path is not None:
                piper_tts_command.store_wav_in_cache(wav_file, cache_path)
        except Exception:
            try:
                os.remove(wav_file)
            except OSError:
                pass
            raise
        return wav_file, True, False

    def _synthesize_wav(self, text: str, wav_file: str) -> None:
        voice = self._get_voice()
        syn_config = self._make_synthesis_config()
        first_chunk = True
        with wave.open(wav_file, "wb") as writer:
            for audio_chunk in voice.synthesize(text, syn_config=syn_config):
                if self._should_interrupt():
                    raise VoiceInterrupted("TTS interrupted during synthesis.")
                if first_chunk:
                    writer.setframerate(audio_chunk.sample_rate)
                    writer.setsampwidth(audio_chunk.sample_width)
                    writer.setnchannels(audio_chunk.sample_channels)
                    first_chunk = False
                writer.writeframes(audio_chunk.audio_int16_bytes)
                silence_frames = int(float(self.sentence_silence) * int(audio_chunk.sample_rate))
                if silence_frames > 0:
                    silence_bytes = b"\x00" * silence_frames * int(audio_chunk.sample_width) * int(
                        audio_chunk.sample_channels
                    )
                    writer.writeframes(silence_bytes)
        if not piper_tts_command._is_valid_wav(wav_file):
            raise RuntimeError("Persistent Piper produced an empty or invalid WAV file.")

    def _make_synthesis_config(self):
        if self.synthesis_config_factory is not None:
            return self.synthesis_config_factory(self.length_scale)
        try:
            from piper import SynthesisConfig
        except ImportError as exc:
            raise RuntimeError("Python package piper is not installed.") from exc
        return SynthesisConfig(length_scale=self.length_scale)

    def _play_wav(self, wav_file: str) -> None:
        command = piper_tts_command.build_player_command(self.player, wav_file)
        process = self.popen_factory(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        playback_active = False

        def set_playback_active(active: bool) -> None:
            nonlocal playback_active
            if playback_active == active:
                return
            playback_active = active
            if self._playback_status_hook is not None:
                self._playback_status_hook(active)

        try:
            set_playback_active(True)
            deadline = time.monotonic() + self.timeout_sec
            while process.poll() is None:
                if self._should_interrupt():
                    self._terminate_process(process)
                    raise VoiceInterrupted("TTS interrupted during playback.")
                if time.monotonic() > deadline:
                    self._terminate_process(process, force=True)
                    raise subprocess.TimeoutExpired(command, self.timeout_sec)
                time.sleep(0.05)
            returncode = process.wait(timeout=0.1)
        finally:
            set_playback_active(False)

        if returncode != 0:
            stderr = ""
            if process.stderr is not None:
                stderr = process.stderr.read().strip()
            raise RuntimeError(f"Audio playback failed with code {returncode}: {stderr}")

    @staticmethod
    def _terminate_process(process: subprocess.Popen, force: bool = False) -> None:
        if process.poll() is not None:
            return
        try:
            if force:
                process.kill()
            else:
                process.terminate()
        except Exception:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass
            process.wait(timeout=1)
