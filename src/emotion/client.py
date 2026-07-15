from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error, request

from src.emotion.followup import register_emotion_result
from src.runtime.status_monitor import get_active_status_monitor
from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("EmotionSideChannel")


Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]


@dataclass(frozen=True)
class EmotionSettings:
    enabled: bool
    service_url: str
    user_id: str
    language: str
    timeout_sec: float
    results_jsonl_path: str
    audio_dir: str
    keep_audio: bool


class NullEmotionSideChannel:
    enabled = False

    def analyze_async(
        self,
        *,
        audio_file_path: str,
        transcript: str,
        sample_rate: int,
        duration_seconds: float,
    ) -> None:
        return


class EmotionSideChannel:
    """Send completed STT turns to the external emo_module API.

    This is intentionally a side channel: it never changes the transcript,
    record.csv, RL state, CBT state, or prompts sent to the LLM.
    """

    enabled = True

    def __init__(self, settings: EmotionSettings, transport: Transport | None = None) -> None:
        self.settings = settings
        self._transport = transport or post_json
        self._write_lock = threading.Lock()

    def analyze_async(
        self,
        *,
        audio_file_path: str,
        transcript: str,
        sample_rate: int,
        duration_seconds: float,
    ) -> None:
        utterance_id, payload, copied_audio_path = self._prepare_request(
            audio_file_path=audio_file_path,
            transcript=transcript,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )
        thread = threading.Thread(
            target=self._send_request,
            args=(utterance_id, payload, copied_audio_path),
            name="caiti-emotion-analysis",
            daemon=True,
        )
        thread.start()

    def analyze(
        self,
        *,
        audio_file_path: str,
        transcript: str,
        sample_rate: int,
        duration_seconds: float,
    ) -> dict[str, Any]:
        utterance_id, payload, copied_audio_path = self._prepare_request(
            audio_file_path=audio_file_path,
            transcript=transcript,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )
        return self._send_request(utterance_id, payload, copied_audio_path)

    def _prepare_request(
        self,
        *,
        audio_file_path: str,
        transcript: str,
        sample_rate: int,
        duration_seconds: float,
    ) -> tuple[str, dict[str, Any], str]:
        utterance_id = self._utterance_id()
        copied_audio_path = self._copy_audio(audio_file_path, utterance_id)
        payload = {
            "utterance_id": utterance_id,
            "transcript": str(transcript or ""),
            "audio_file_path": copied_audio_path or audio_file_path,
            "language": self.settings.language,
            "user_id": self.settings.user_id,
            "sample_rate": int(sample_rate or 0),
            "duration_seconds": float(duration_seconds or 0.0),
        }
        return utterance_id, payload, copied_audio_path

    def _send_request(
        self,
        utterance_id: str,
        payload: dict[str, Any],
        copied_audio_path: str,
    ) -> dict[str, Any]:
        started_at = time.time()
        record: dict[str, Any]
        try:
            response = self._transport(self.settings.service_url, payload, self.settings.timeout_sec)
            record = {
                "status": "ok",
                "utterance_id": utterance_id,
                "created_at": started_at,
                "latency_sec": round(time.time() - started_at, 3),
                "request": payload,
                "response": response,
            }
            logger.info(
                "Emotion scores utterance_id=%s latency=%.3fs %s",
                utterance_id,
                record["latency_sec"],
                json.dumps(_emotion_log_summary(response), ensure_ascii=False, sort_keys=True),
            )
        except Exception as exc:
            record = {
                "status": "error",
                "utterance_id": utterance_id,
                "created_at": started_at,
                "latency_sec": round(time.time() - started_at, 3),
                "request": payload,
                "error": str(exc),
            }
            logger.warning("Emotion analysis failed for utterance_id=%s: %s", utterance_id, exc)
        finally:
            if copied_audio_path and not self.settings.keep_audio:
                try:
                    os.remove(copied_audio_path)
                except OSError:
                    pass

        self._append_result(record)
        register_emotion_result(record)
        monitor = get_active_status_monitor()
        if monitor is not None:
            setter = getattr(monitor, "set_emotion_result", None)
            if callable(setter):
                try:
                    setter(record)
                except Exception:
                    pass
        return record

    def _utterance_id(self) -> str:
        return f"{self.settings.user_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    def _copy_audio(self, audio_file_path: str, utterance_id: str) -> str:
        if not audio_file_path or not os.path.exists(audio_file_path):
            return ""
        audio_dir = os.path.abspath(self.settings.audio_dir)
        os.makedirs(audio_dir, exist_ok=True)
        target = os.path.join(audio_dir, f"{utterance_id}.wav")
        shutil.copyfile(audio_file_path, target)
        return target

    def _append_result(self, record: dict[str, Any]) -> None:
        path = self.settings.results_jsonl_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._write_lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _emotion_log_summary(response: dict[str, Any]) -> dict[str, Any]:
    final_assessment = response.get("final_assessment", {})
    final_result = response.get("final_result", {})
    comparison = response.get("emotion_comparison", {})
    audio_scores = response.get("audio_scores", {})
    text_scores = response.get("text_scores", {})
    return {
        "risk": final_result.get("credibility_risk", final_assessment.get("credibility_risk")),
        "risk_level": final_result.get("risk_level", final_assessment.get("risk_level")),
        "confidence": final_assessment.get("confidence"),
        "uncertainty": final_assessment.get("uncertainty"),
        "audio_emotion": response.get("audio_emotion"),
        "context_emotion": response.get("context_emotion"),
        "consistent": comparison.get("audio_vs_context_consistent"),
        "arousal_conflict": comparison.get("arousal_conflict"),
        "valence_conflict": comparison.get("valence_conflict"),
        "contradiction_or_sarcasm": comparison.get("contradiction_or_sarcasm"),
        "audio_arousal": audio_scores.get("arousal"),
        "audio_tension": audio_scores.get("tension"),
        "audio_stability": audio_scores.get("stability"),
        "text_valence": text_scores.get("text_valence"),
        "certainty": text_scores.get("certainty"),
    }


def post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            data = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    return json.loads(data or "{}")


def build_emotion_side_channel() -> EmotionSideChannel | NullEmotionSideChannel:
    if not config_loader.EMOTION_ENABLED or not config_loader.EMOTION_SERVICE_URL:
        return NullEmotionSideChannel()
    settings = EmotionSettings(
        enabled=True,
        service_url=config_loader.EMOTION_SERVICE_URL,
        user_id=config_loader.EMOTION_USER_ID,
        language=config_loader.EMOTION_LANGUAGE,
        timeout_sec=config_loader.EMOTION_TIMEOUT_SEC,
        results_jsonl_path=config_loader.EMOTION_RESULTS_JSONL_PATH,
        audio_dir=config_loader.EMOTION_AUDIO_DIR,
        keep_audio=config_loader.EMOTION_KEEP_AUDIO,
    )
    return EmotionSideChannel(settings)
