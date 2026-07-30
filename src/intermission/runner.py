"""Parallel intermission runner for CaiTI voice sessions.

The runner speaks short mini-tasks while the main CaiTI pipeline is thinking.
It never writes to record.csv, never calls the LLM, and stores PHQ/GAD answers
only in isolated intermission tables so they cannot affect RL, CBT, reports,
or adapter prompts.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from src.intermission.tasks import (
    BREATHING_TASKS,
    MINDFULNESS_TASKS,
    SCREENING_ITEMS,
    MiniTaskKind,
    ScriptedTask,
    ScreeningItem,
    classify_screening_response,
)
from src.intermission.storage import IntermissionScreeningStore, build_intermission_store
from src.runtime.status_monitor import get_active_status_monitor
from src.utils import config_loader
from src.utils.log_util import get_logger
from src.voice.backends import TTSBackend
from src.voice.tts import TTSRouteSettings, build_tts_from_settings

logger = get_logger("IntermissionRunner")
AUDIO_RELEASE_SETTLE_SEC = 0.15


@dataclass(frozen=True)
class IntermissionSettings:
    enabled: bool = False
    tts_backend: str = "command"
    tts_command: str = ""
    fallback_to_primary_tts: bool = True
    screening_enabled: bool = True
    breathing_enabled: bool = True
    mindfulness_enabled: bool = True
    max_seconds: float = 45.0
    poll_interval_sec: float = 0.1
    max_screening_items_per_turn: int = 1
    trigger_min_user_speech_sec: float = 10.0
    trigger_min_interval_turns: int = 2
    trigger_probability: float = 0.5
    cooldown_turns: int = 1
    persist_results: bool = True
    db_path: str = ""
    results_json_path: str = ""
    lead_in_text: str = (
        "Let's do a brief check-in together."
    )
    bridge_text: str = "Let's go back to the main session."
    transition_delay_sec: float = 2.0


@dataclass
class NullIntermissionRunner:
    def run_until_ready(
        self,
        is_ready: Callable[[], bool],
        should_stop: Callable[[], bool] | None = None,
        user_speech_duration_sec: float | None = None,
    ) -> None:
        return

    def begin_session(
        self,
        *,
        db_path: str,
        results_json_path: str,
        session_id: str,
        resume: bool = False,
    ) -> None:
        return

    def end_session(self) -> None:
        return


@dataclass
class IntermissionRunner:
    settings: IntermissionSettings
    tts: TTSBackend
    stt: object | None = None
    music: object | None = None
    status_leds: object | None = None
    store: IntermissionScreeningStore | None = None
    _screening_index: int = field(default=0, init=False)
    _screening_answers: dict[str, int | None] = field(default_factory=dict, init=False)
    _script_index: int = field(default=0, init=False)
    _cooldown_remaining: int = field(default=0, init=False)
    _eligible_turns_since_activity: int = field(default=0, init=False)

    def begin_session(
        self,
        *,
        db_path: str,
        results_json_path: str,
        session_id: str,
        resume: bool = False,
    ) -> None:
        """Bind private screening state and storage to one explicit session."""
        self._screening_index = 0
        self._screening_answers = {}
        self._script_index = 0
        self._cooldown_remaining = 0
        self._eligible_turns_since_activity = 0
        self.store = (
            build_intermission_store(
                db_path,
                results_json_path,
                session_id=session_id,
            )
            if self.settings.persist_results
            else None
        )
        if resume and self.store is not None:
            stored_items = self.store.fetch_items()
            attempted_ids: set[str] = set()
            for record in stored_items:
                item_id = str(record.get("item_id", ""))
                if not item_id:
                    continue
                attempted_ids.add(item_id)
                score = record.get("score")
                self._screening_answers[item_id] = None if score is None else int(score)
            for index, item in enumerate(SCREENING_ITEMS):
                if item.item_id not in attempted_ids:
                    self._screening_index = index
                    break
            else:
                self._screening_index = len(SCREENING_ITEMS)
        logger.info(
            "Intermission bound to session %s: resume=%s screening_index=%s.",
            session_id,
            bool(resume),
            self._screening_index,
        )

    def end_session(self) -> None:
        """Drop all session-specific state while keeping the warmed TTS backend."""
        self._screening_index = 0
        self._screening_answers = {}
        self._script_index = 0
        self._cooldown_remaining = 0
        self._eligible_turns_since_activity = 0
        self.store = None

    def run_until_ready(
        self,
        is_ready: Callable[[], bool],
        should_stop: Callable[[], bool] | None = None,
        user_speech_duration_sec: float | None = None,
    ) -> None:
        if not self.settings.enabled or self._should_stop(should_stop) or is_ready():
            return
        if not self._should_run_this_turn(user_speech_duration_sec):
            return

        deadline = time.monotonic() + max(0.0, self.settings.max_seconds)
        did_activity = False
        lead_in_spoken = False
        screening_count = 0

        while time.monotonic() < deadline and not is_ready() and not self._should_stop(should_stop):
            activity = self._next_activity(screening_count)
            if activity is None:
                self._sleep_until_ready(deadline, is_ready, should_stop)
                break

            did_activity = True
            if not lead_in_spoken:
                lead_in_spoken = True
                self._speak(self.settings.lead_in_text)
            if isinstance(activity, ScreeningItem):
                screening_count += 1
                self._run_screening_item(activity)
            else:
                self._run_scripted_task(activity, deadline, is_ready, should_stop)

        if did_activity and is_ready() and not self._should_stop(should_stop):
            self._sleep_before_bridge(should_stop)
            if not self._should_stop(should_stop):
                self._speak(self.settings.bridge_text)
        if did_activity:
            self._cooldown_remaining = max(0, int(self.settings.cooldown_turns))
            self._eligible_turns_since_activity = 0

    def _should_run_this_turn(self, user_speech_duration_sec: float | None) -> bool:
        if self._cooldown_remaining > 0:
            logger.info(
                "Skipping intermission; cooldown has %s turn(s) remaining.",
                self._cooldown_remaining,
            )
            self._cooldown_remaining -= 1
            return False

        threshold = max(0.0, float(self.settings.trigger_min_user_speech_sec))
        duration = 0.0 if user_speech_duration_sec is None else max(0.0, float(user_speech_duration_sec))
        if duration < threshold:
            logger.info(
                "Skipping intermission; user speech %.2fs is below %.2fs trigger.",
                duration,
                threshold,
            )
            return False

        self._eligible_turns_since_activity += 1
        min_interval_turns = max(0, int(self.settings.trigger_min_interval_turns))
        if self._eligible_turns_since_activity <= min_interval_turns:
            logger.info(
                "Skipping intermission; eligible interval counter is %s/%s.",
                self._eligible_turns_since_activity,
                min_interval_turns,
            )
            return False

        probability = min(1.0, max(0.0, float(self.settings.trigger_probability)))
        roll = random.random()
        if roll >= probability:
            logger.info(
                "Skipping intermission; random trigger roll %.3f is above probability %.3f.",
                roll,
                probability,
            )
            return False
        return True

    @property
    def screening_answers(self) -> dict[str, int | None]:
        return dict(self._screening_answers)

    @property
    def screening_totals(self) -> dict[str, dict[str, int | bool]]:
        summaries: dict[str, dict[str, int | bool]] = {}
        for scale in sorted({item.scale for item in SCREENING_ITEMS}):
            item_ids = [item.item_id for item in SCREENING_ITEMS if item.scale == scale]
            scores = [
                self._screening_answers[item_id]
                for item_id in item_ids
                if self._screening_answers.get(item_id) is not None
            ]
            summaries[scale] = {
                "total": sum(int(score) for score in scores),
                "answered": len(scores),
                "expected": len(item_ids),
                "complete": len(scores) == len(item_ids),
            }
        return summaries

    def _next_activity(self, screening_count: int) -> ScreeningItem | ScriptedTask | None:
        if (
            self.settings.screening_enabled
            and screening_count < self.settings.max_screening_items_per_turn
            and self._screening_index < len(SCREENING_ITEMS)
        ):
            item = SCREENING_ITEMS[self._screening_index]
            self._screening_index += 1
            return item

        scripted: list[ScriptedTask] = []
        if self.settings.breathing_enabled:
            scripted.extend(BREATHING_TASKS)
        if self.settings.mindfulness_enabled:
            scripted.extend(MINDFULNESS_TASKS)
        if not scripted:
            return None
        task = scripted[self._script_index % len(scripted)]
        self._script_index += 1
        return task

    def _run_screening_item(self, item: ScreeningItem) -> None:
        self._speak(item.prompt, source="intermission_screening", expects_response=True)
        if self.stt is None:
            self._screening_answers[item.item_id] = None
            self._persist_screening_result(
                item,
                status="UNRESOLVED",
                score=None,
                response_text="",
                reason="stt_unavailable",
            )
            return
        listen_error = None
        try:
            self._duck_music()
            self._set_stt_active(True)
            listen = getattr(self.stt, "listen")
            response = str(listen() or "")
        except Exception as exc:
            logger.info("Private intermission screening item %s was not resolved: %s", item.item_id, exc)
            listen_error = exc
            response = ""
        finally:
            self._set_stt_active(False)
            self._restore_music()
        result = classify_screening_response(response)
        reason = result.reason
        if listen_error is not None and result.status == "UNRESOLVED":
            reason = f"stt_error:{type(listen_error).__name__}"
        self._screening_answers[item.item_id] = result.score
        self._persist_screening_result(
            item,
            status=result.status,
            score=result.score,
            response_text=response,
            reason=reason,
        )
        self._speak("Thank you.")

    def _run_scripted_task(
        self,
        task: ScriptedTask,
        deadline: float,
        is_ready: Callable[[], bool],
        should_stop: Callable[[], bool] | None,
    ) -> None:
        source = f"intermission_{task.kind.value}"
        for step in task.iter_steps():
            if time.monotonic() >= deadline or is_ready() or self._should_stop(should_stop):
                return
            self._speak(step.text, source=source)
            self._sleep_between_scripted_steps(
                step.pause_after_sec,
                deadline,
                is_ready,
                should_stop,
            )

    def _persist_screening_result(
        self,
        item: ScreeningItem,
        *,
        status: str,
        score: int | None,
        response_text: str,
        reason: str,
    ) -> None:
        if self.store is None:
            return
        try:
            self.store.upsert_item(
                item_id=item.item_id,
                scale=item.scale,
                status=status,
                score=score,
                response_text=response_text,
                reason=reason,
            )
            self.store.upsert_summary(self.screening_totals)
            monitor = get_active_status_monitor()
            if monitor is not None:
                setter = getattr(monitor, "set_intermission_state", None)
                if callable(setter):
                    setter(summary=self.store.fetch_summary(), items=self.store.fetch_items())
        except Exception as exc:
            logger.warning("Failed to persist private intermission screening result %s: %s", item.item_id, exc)

    def _speak(self, text: str, source: str = "intermission", expects_response: bool = False) -> None:
        chunk = str(text or "").strip()
        if not chunk:
            return
        self._publish_prompt(chunk, source=source, expects_response=expects_response)
        set_playback_status_hook = getattr(self.tts, "set_playback_status_hook", None)
        playback_status_hook_installed = callable(set_playback_status_hook)
        led_active = False

        def set_tts_active(active: bool) -> None:
            nonlocal led_active
            led_active = bool(active)
            self._set_tts_active(led_active)

        self._suspend_music_for_spoken_audio()
        try:
            if playback_status_hook_installed:
                set_playback_status_hook(set_tts_active)
            else:
                set_tts_active(True)
            self.tts.speak_stream(chunk)
        finally:
            if playback_status_hook_installed:
                set_playback_status_hook(None)
            if led_active:
                self._set_tts_active(False)
            self._resume_music_after_spoken_audio()

    def _music_is_background(self) -> bool:
        method = getattr(self.music, "is_background", None)
        if not callable(method):
            return False
        try:
            return bool(method())
        except Exception as exc:
            logger.warning("Intermission music background-state check failed: %s", exc)
            return False

    def _suspend_music_for_spoken_audio(self) -> None:
        if self.music is None:
            return
        if self._music_is_background():
            stop_method = getattr(self.music, "stop", None)
            if callable(stop_method):
                stop_method()
                time.sleep(AUDIO_RELEASE_SETTLE_SEC)
                return
        self._duck_music()

    def _resume_music_after_spoken_audio(self) -> None:
        if self.music is None:
            return
        if self._music_is_background():
            start_method = getattr(self.music, "start", None)
            if callable(start_method):
                start_method()
            self._restore_music()
            return
        self._restore_music()

    def _duck_music(self) -> None:
        method = getattr(self.music, "duck", None)
        if callable(method):
            method()

    def _restore_music(self) -> None:
        method = getattr(self.music, "restore_volume", None)
        if callable(method):
            method()

    def _set_tts_active(self, active: bool) -> None:
        method = getattr(self.status_leds, "set_tts_active", None)
        if not callable(method):
            return
        try:
            method(bool(active))
        except Exception as exc:
            logger.warning("Intermission TTS LED hook failed: %s", exc)

    def _set_stt_active(self, active: bool) -> None:
        method = getattr(self.status_leds, "set_stt_active", None)
        if not callable(method):
            return
        try:
            method(bool(active))
        except Exception as exc:
            logger.warning("Intermission STT LED hook failed: %s", exc)

    def _publish_prompt(self, text: str, *, source: str, expects_response: bool) -> None:
        monitor = get_active_status_monitor()
        if monitor is None:
            return
        setter = getattr(monitor, "set_prompt", None)
        if not callable(setter):
            return
        try:
            setter(
                text=str(text or ""),
                source=source,
                expects_response=bool(expects_response),
            )
        except Exception:
            return

    def _sleep_until_ready(
        self,
        deadline: float,
        is_ready: Callable[[], bool],
        should_stop: Callable[[], bool] | None,
    ) -> None:
        while time.monotonic() < deadline and not is_ready() and not self._should_stop(should_stop):
            time.sleep(max(0.01, self.settings.poll_interval_sec))

    def _sleep_before_bridge(self, should_stop: Callable[[], bool] | None) -> None:
        delay = max(0.0, float(self.settings.transition_delay_sec))
        if delay <= 0:
            return
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline and not self._should_stop(should_stop):
            time.sleep(min(max(0.01, self.settings.poll_interval_sec), deadline - time.monotonic()))

    def _sleep_between_scripted_steps(
        self,
        delay: float,
        deadline: float,
        is_ready: Callable[[], bool],
        should_stop: Callable[[], bool] | None,
    ) -> None:
        remaining = max(0.0, min(float(delay), max(0.0, deadline - time.monotonic())))
        while remaining > 0 and not is_ready() and not self._should_stop(should_stop):
            interval = min(
                remaining,
                max(0.01, self.settings.poll_interval_sec),
                max(0.0, deadline - time.monotonic()),
            )
            if interval <= 0:
                return
            time.sleep(interval)
            remaining = max(0.0, remaining - interval)

    @staticmethod
    def _should_stop(should_stop: Callable[[], bool] | None) -> bool:
        return bool(callable(should_stop) and should_stop())


def build_intermission_settings() -> IntermissionSettings:
    return IntermissionSettings(
        enabled=config_loader.INTERMISSION_ENABLED,
        tts_backend=config_loader.INTERMISSION_TTS_BACKEND,
        tts_command=config_loader.INTERMISSION_TTS_COMMAND,
        fallback_to_primary_tts=config_loader.INTERMISSION_FALLBACK_TO_PRIMARY_TTS,
        screening_enabled=config_loader.INTERMISSION_SCREENING_ENABLED,
        breathing_enabled=config_loader.INTERMISSION_BREATHING_ENABLED,
        mindfulness_enabled=config_loader.INTERMISSION_MINDFULNESS_ENABLED,
        max_seconds=config_loader.INTERMISSION_MAX_SECONDS,
        poll_interval_sec=config_loader.INTERMISSION_POLL_INTERVAL_SEC,
        max_screening_items_per_turn=config_loader.INTERMISSION_MAX_SCREENING_ITEMS_PER_TURN,
        trigger_min_user_speech_sec=config_loader.INTERMISSION_TRIGGER_MIN_USER_SPEECH_SEC,
        trigger_min_interval_turns=config_loader.INTERMISSION_TRIGGER_MIN_INTERVAL_TURNS,
        trigger_probability=config_loader.INTERMISSION_TRIGGER_PROBABILITY,
        cooldown_turns=config_loader.INTERMISSION_COOLDOWN_TURNS,
        persist_results=config_loader.INTERMISSION_PERSIST_RESULTS,
        db_path=config_loader.INTERMISSION_DB_PATH,
        results_json_path=config_loader.INTERMISSION_RESULTS_JSON_PATH,
        lead_in_text=config_loader.INTERMISSION_LEAD_IN_TEXT,
        bridge_text=config_loader.INTERMISSION_BRIDGE_TEXT,
        transition_delay_sec=config_loader.INTERMISSION_TRANSITION_DELAY_SEC,
    )


def build_intermission_runner(
    stt=None,
    primary_tts=None,
    music=None,
    status_leds=None,
) -> IntermissionRunner | NullIntermissionRunner:
    settings = build_intermission_settings()
    if not settings.enabled:
        return NullIntermissionRunner()
    tts = build_tts_from_settings(
        TTSRouteSettings(
            role="intermission",
            backend=settings.tts_backend,
            command=settings.tts_command,
            timeout_sec=config_loader.VOICE_TTS_TIMEOUT_SEC,
            fallback_to_primary=settings.fallback_to_primary_tts,
        ),
        primary_tts=primary_tts,
    )
    if tts is None:
        return NullIntermissionRunner()
    store = (
        build_intermission_store(settings.db_path, settings.results_json_path)
        if settings.persist_results
        else None
    )
    return IntermissionRunner(
        settings=settings,
        tts=tts,
        stt=stt,
        music=music,
        status_leds=status_leds,
        store=store,
    )
