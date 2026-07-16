import time
from typing import Dict, Any

import numpy as np
import os
import pandas as pd

from src.questioner import (
    QuestionTurnOutcome,
    ask_question,
    append_pending_next_question_intro,
    pop_pending_validation_for_workflow_transition,
)
from src.CBT import run_cbt
from src.session.control import NullSessionControl
from src.utils.config_loader import (
    ITEM_N_STATES,
    GAMMA,
    ALPHA,
    QUESTION_LIB_FILENAME,
    SUBJECT_ID,
    DATA_DIR,
    RESULT_DIR,
    REPORT_FILE,
    NOTES_FILE,
    STAGED_SCREENING_ENABLED,
    STAGED_SCREENING_STAGES,
)
from src.utils.config_loader import RECORD_CSV
from src.utils.io_question_lib import load_question_lib, save_question_lib, generate_results
from src.utils.io_record import RECORD_LOCK, init_record, log_question, log_system_message, set_question_prefix
from src.utils.rl_qtables import (
    initialize_q_table,
    choose_action,
    get_env_feedback,
)
from src.utils.session_records import (
    build_rl_trace_path,
    build_session_summary_path,
    cbt_candidates_from_question_lib,
    make_run_id,
    now_iso,
    write_rl_trace,
    write_session_summary,
)
# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.llm_client import llm_complete
logger = get_logger("HandlerRL")

OPENING_GREETING = (
    "Hello, I am Caiti, your AI therapist. Thank you for joining me today."
)

class HandlerRL:
    """
    Top-level RL workflow coordinator.
    Handles the main reinforcement learning loop for question selection and evaluation.
    All file I/O is performed via utility modules.
    """

    def __init__(self, session_control=None):
        # Stores the last question asked to the user
        self.last_question: str = " "
        # Stores all user responses for later result generation
        self.new_response: list = []
        # The main question library loaded from file
        self.question_lib: Dict[str, Any] = {}
        # Q-table for item selection (top-level RL)
        self.item_q_table = None
        # Action id -> label mapping for logging readability
        self.item_action_labels = {}
        self.screening_stages: list[dict[str, Any]] = []
        self.session_control = session_control or NullSessionControl()
        self.run_id = make_run_id(SUBJECT_ID)
        self.rl_trace_records = []
        self.rl_trace_file = build_rl_trace_path(RESULT_DIR, SUBJECT_ID, self.run_id)
        self.session_summary_file = build_session_summary_path(RESULT_DIR, SUBJECT_ID, self.run_id)
        self.q_table_file = os.path.join(DATA_DIR, "q_tables", f"item_qtable_{SUBJECT_ID}.csv")

    def setup(self):
        """
        Initialize records, load question library, and set up Q-tables and masks.
        """
        logger.info("Initializing RL handler setup: loading records and question library.")
        init_record()
        self.question_lib = load_question_lib(QUESTION_LIB_FILENAME)
        dimension_count = len(self.question_lib)
        expected_n_states = dimension_count + 2  # START + 37 dimensions + END.
        if ITEM_N_STATES != expected_n_states:
            logger.warning(
                "Configured ITEM_N_STATES=%s, but question library has %s dimensions; "
                "paper-consistent state count should be %s.",
                ITEM_N_STATES,
                dimension_count,
                expected_n_states,
            )
        # Define possible actions for item selection (as string indices)
        item_actions = ['{0}'.format(e) for e in np.arange(0, ITEM_N_STATES)]
        # # Initialize masks and question-level Q-tables are deprecated; single-question per item is used
        # self.all_question_mask = {}
        # self.all_question_q_table = {}
        self.item_q_table = initialize_q_table(ITEM_N_STATES, item_actions)
        self.item_actions = item_actions

        # Build action id -> label mapping for logging readability
        # Action "0" is a synthetic start/index action and not part of the question lib
        # Action ITEM_N_STATES - 1 is the paper's explicit END state and is masked out.
        self.item_action_labels = {"0": "START", str(ITEM_N_STATES - 1): "END"}
        for i in range(1, dimension_count + 1):
            self.item_action_labels[str(i)] = self.question_lib[str(i)]["1"]["label"]
        self.screening_stages = self._resolve_screening_stages()
  
        # Load persistent Q tables (if exist)
        qdir = os.path.join(DATA_DIR, "q_tables")
        qfile = self.q_table_file
        if os.path.exists(qfile):
            loaded_q_table = pd.read_csv(qfile, index_col=0)
            expected_columns = list(self.item_q_table.columns)
            if loaded_q_table.shape == self.item_q_table.shape and list(loaded_q_table.columns) == expected_columns:
                self.item_q_table = loaded_q_table
                logger.info(f"Loaded item Q table for subject {SUBJECT_ID} from {qfile}.")
            else:
                logger.warning(
                    "Ignoring incompatible item Q table at %s. Expected shape=%s columns=%s, "
                    "got shape=%s columns=%s. A paper-consistent table will be initialized.",
                    qfile,
                    self.item_q_table.shape,
                    expected_columns,
                    loaded_q_table.shape,
                    list(loaded_q_table.columns),
                )
        else:
            logger.info(f"Item Q table for subject {SUBJECT_ID} not found at {qfile}. ")
        
        logger.info("RL handler setup complete.")

    def _resolve_screening_stages(self) -> list[dict[str, Any]]:
        """Resolve configured dimension labels to item ids and validate full coverage."""
        if not STAGED_SCREENING_ENABLED:
            logger.info("Staged screening is disabled; RL may select from all remaining dimensions.")
            return []
        if not isinstance(STAGED_SCREENING_STAGES, list) or not STAGED_SCREENING_STAGES:
            raise ValueError(
                "rl.staged_screening.enabled is true, but no screening stages are configured"
            )

        label_to_item: dict[str, int] = {}
        for item_index in range(1, len(self.question_lib) + 1):
            label = str(
                self.question_lib.get(str(item_index), {}).get("1", {}).get("label", "")
            ).strip()
            if not label:
                raise ValueError(f"Question item {item_index} has no dimension label")
            if label in label_to_item:
                raise ValueError(f"Duplicate question-library dimension label: {label}")
            label_to_item[label] = item_index

        resolved: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        seen_names: set[str] = set()
        for stage_position, raw_stage in enumerate(STAGED_SCREENING_STAGES, start=1):
            if not isinstance(raw_stage, dict):
                raise ValueError(f"Screening stage {stage_position} must be a mapping")
            stage_name = str(raw_stage.get("name", "")).strip()
            stage_intro = str(raw_stage.get("intro", "")).strip()
            dimensions = raw_stage.get("dimensions", [])
            if not stage_name:
                raise ValueError(f"Screening stage {stage_position} has no name")
            if not stage_intro:
                raise ValueError(f"Screening stage {stage_name} has no intro")
            if stage_name in seen_names:
                raise ValueError(f"Duplicate screening stage name: {stage_name}")
            if not isinstance(dimensions, list) or not dimensions:
                raise ValueError(f"Screening stage {stage_name} has no dimensions")

            item_ids: list[int] = []
            normalized_labels: list[str] = []
            for raw_label in dimensions:
                label = str(raw_label).strip()
                if label not in label_to_item:
                    raise ValueError(
                        f"Unknown dimension {label!r} in screening stage {stage_name}"
                    )
                if label in seen_labels:
                    raise ValueError(
                        f"Dimension {label!r} appears in more than one screening stage"
                    )
                seen_labels.add(label)
                normalized_labels.append(label)
                item_ids.append(label_to_item[label])

            seen_names.add(stage_name)
            resolved.append(
                {
                    "name": stage_name,
                    "labels": normalized_labels,
                    "intro": stage_intro,
                    "item_ids": item_ids,
                }
            )

        missing_labels = sorted(set(label_to_item) - seen_labels)
        if missing_labels:
            raise ValueError(
                "Staged screening does not cover every question-library dimension: "
                + ", ".join(missing_labels)
            )

        logger.info(
            "Resolved %s screening stages covering %s dimensions.",
            len(resolved),
            len(seen_labels),
        )
        return resolved

    def _active_stage_mask(
        self,
        item_mask: list[int],
    ) -> tuple[list[int], int | None, str]:
        """Limit available actions to the earliest stage with unanswered dimensions."""
        if not self.screening_stages:
            return list(item_mask), None, ""

        for stage_index, stage in enumerate(self.screening_stages):
            active_item_ids = [
                item_id
                for item_id in stage["item_ids"]
                if 0 < item_id < len(item_mask) - 1 and item_mask[item_id] == 1
            ]
            if active_item_ids:
                stage_mask = [0] * len(item_mask)
                for item_id in active_item_ids:
                    stage_mask[item_id] = 1
                return stage_mask, stage_index, str(stage["name"])

        if sum(item_mask):
            raise RuntimeError("Unanswered dimensions exist outside the configured screening stages")
        return list(item_mask), None, ""

    def _item_is_answered(self, item_index: int) -> bool:
        entry = self.question_lib.get(str(item_index), {}).get("1", {})
        scores = entry.get("score", [])
        return bool(scores)

    def _build_item_mask(self, dimension_count: int) -> list[int]:
        # START and END are never askable screening actions.
        mask = [0]
        answered = 0
        remaining = 0
        for item_index in range(1, dimension_count + 1):
            if self._item_is_answered(item_index):
                mask.append(0)
                answered += 1
            else:
                mask.append(1)
                remaining += 1
        mask.append(0)
        logger.info(
            "Prepared screening restart mask from saved question library: answered=%s remaining=%s",
            answered,
            remaining,
        )
        return mask

    def _sync_answered_item_mask(self, item_mask: list[int]) -> int:
        """Remove dimensions scored incidentally by a multi-segment answer."""
        cleared = 0
        for item_index in range(1, min(len(self.question_lib) + 1, len(item_mask) - 1)):
            if item_mask[item_index] and self._item_is_answered(item_index):
                item_mask[item_index] = 0
                cleared += 1
        if cleared:
            logger.info("Removed %s cross-scored dimension(s) from the active item mask.", cleared)
        return cleared

    def _apply_question_outcome_to_mask(
        self,
        item_mask: list[int],
        current_item_index: int,
        outcome: QuestionTurnOutcome,
    ) -> set[int]:
        """Mask only dimensions with committed scores; keep an unanswered current item available."""
        masked_item_ids: set[int] = set()
        for item_id in outcome.covered_item_ids:
            item_index = int(item_id)
            if 0 < item_index < len(item_mask) - 1:
                item_mask[item_index] = 0
                masked_item_ids.add(item_index)

        current_item_index = int(current_item_index)
        if 0 < current_item_index < len(item_mask) - 1:
            item_mask[current_item_index] = (
                0 if outcome.current_answered else 1
            )
            if outcome.current_answered:
                masked_item_ids.add(current_item_index)

        logger.info(
            "Updated screening mask from question outcome: covered=%s current=%s current_answered=%s",
            sorted(masked_item_ids),
            current_item_index,
            outcome.current_answered,
        )
        return masked_item_ids

    def _persist_runtime_question_lib(self) -> None:
        save_question_lib(QUESTION_LIB_FILENAME, self.question_lib)
        logger.info("Persisted runtime question library to %s.", QUESTION_LIB_FILENAME)

    def _queue_pending_screening_validation_for_cbt(self) -> str:
        """Move terminal screening validation onto the first CBT output."""
        validation = pop_pending_validation_for_workflow_transition()
        if validation:
            set_question_prefix(validation)
            logger.info(
                "Queued terminal screening validation for delivery before the first CBT output."
            )
        return validation

    def _update_active_item_q_table(
        self,
        q_table,
        state,
        action,
        next_state,
        reward,
    ) -> tuple[float, float, float]:
        """Apply one Q update in place so the next selection sees it."""
        q_before = q_table.loc[state, action]
        if next_state == "terminal":
            q_target = reward
        else:
            q_target = reward + GAMMA * q_table.iloc[int(next_state), :].max()
        q_table.loc[state, action] = q_before + ALPHA * (q_target - q_before)
        return q_before, q_table.loc[state, action], q_target

    def run(self):
        """
        Main RL loop for the entire screening process.
        Iteratively selects items and asks questions using RL, updating Q-tables and saving results.
        """
        logger.info("Starting main RL screening process.")
        self.setup()

        # Opening greeting is deterministic so it cannot be rewritten into
        # "Hello CaiTI" or delay the first real screening question.
        set_question_prefix(OPENING_GREETING)
        time.sleep(0.5)
        active_q_table = self.item_q_table.copy()
        S = 0  # Start state for item RL
        is_terminated = False
        dimension_count = len(self.question_lib)
        # Mask for available actions: START and END are states, not screening dimensions to ask.
        # On restart/resume, any dimension with a saved score is treated as already completed.
        item_mask = self._build_item_mask(dimension_count)
        active_stage_index: int | None = None
        while not is_terminated:
            control_action = self.session_control.checkpoint("screening")
            if control_action == "skip_to_cbt":
                is_terminated = True
                logger.info("Session button requested skipping screening and proceeding to CBT.")
                break
            # If all items have been asked, exit to CBT directly
            if sum(item_mask) == 0:
                is_terminated = True
                logger.info("All items have been asked. Proceeding to CBT.")
                break
            # Staging changes only the eligible actions. The existing Q-values,
            # epsilon-greedy policy, rewards, and update rule remain untouched.
            selection_mask, stage_index, stage_name = self._active_stage_mask(item_mask)
            if stage_index is not None and stage_index != active_stage_index:
                logger.info(
                    "Entering screening stage %s/%s: %s",
                    stage_index + 1,
                    len(self.screening_stages),
                    stage_name,
                )
                active_stage_index = stage_index
                stage_intro = str(self.screening_stages[stage_index].get("intro", "")).strip()
                if stage_intro:
                    append_pending_next_question_intro(stage_intro)
                    logger.info("Queued screening stage intro: %s", stage_name)
            A = choose_action(
                S,
                active_q_table,
                selection_mask,
                ITEM_N_STATES,
                self.item_actions,
                self.item_action_labels,
            )
            if int(A) < 1 or int(A) > dimension_count:
                is_terminated = True
                logger.info("RL selected non-screening state %s. Proceeding to CBT.", A)
                break
            # Ask questions for the selected item
            turn_start = len(self.new_response)
            question_outcome = ask_question(
                self.question_lib,
                int(A),
                turn_records=self.new_response,
                session_control=self.session_control,
            )
            openai_res = question_outcome.reward
            DLA_terminate = question_outcome.terminate
            last_question_updated = question_outcome.previous_question
            self._apply_question_outcome_to_mask(
                item_mask,
                int(A),
                question_outcome,
            )
            self._persist_runtime_question_lib()
            action_turn_records = self.new_response[turn_start:]
            primary_turn = action_turn_records[-1] if action_turn_records else {}
            self.last_question = last_question_updated
            # Get next state and reward for item RL
            S_, R = get_env_feedback(S, A, openai_res, DLA_terminate, item_mask)
            # Update the same table used by the next choose_action call.
            q_predict, q_after, q_target = self._update_active_item_q_table(
                active_q_table,
                S,
                A,
                S_,
                R,
            )
            if S_ == "terminal":
                is_terminated = True
            trace_record = {
                "RunID": self.run_id,
                "SubjectID": SUBJECT_ID,
                "Step": len(self.rl_trace_records) + 1,
                "Timestamp": now_iso(),
                "State": S,
                "Action": A,
                "NextState": S_,
                "ScreeningStageIndex": stage_index + 1 if stage_index is not None else "",
                "ScreeningStage": stage_name,
                "Dimension": self.item_action_labels.get(str(A), str(A)),
                "Question": primary_turn.get("Original_question", ""),
                "UserResponse": primary_turn.get("User_input", ""),
                "Classification": primary_turn.get("DLA_result", ""),
                "Score": primary_turn.get("Score", ""),
                "Reward": R,
                "QBefore": q_predict,
                "QAfter": q_after,
                "Terminate": DLA_terminate,
                "AttemptCount": len(action_turn_records),
                "SegmentCount": primary_turn.get("SegmentCount", 0),
                "AnalyzerCallCount": primary_turn.get("AnalyzerCallCount", 0),
                "AnalyzerLatencyMs": primary_turn.get("AnalyzerLatencyMs", 0.0),
                "RVLatencyMs": primary_turn.get("RVLatencyMs", 0.0),
                "TotalTurnLatencyMs": primary_turn.get("TotalTurnLatencyMs", 0.0),
                "BatchFallback": primary_turn.get("BatchFallback", 0),
            }
            self.rl_trace_records.append(trace_record)
            for turn_record in action_turn_records:
                turn_record["Reward"] = R
                turn_record["QState"] = S
                turn_record["QAction"] = A
                turn_record["QBefore"] = q_predict
                turn_record["QAfter"] = q_after
                turn_record["NextState"] = S_
                turn_record["ScreeningStageIndex"] = stage_index + 1 if stage_index is not None else ""
                turn_record["ScreeningStage"] = stage_name
            logger.debug(
                f"Q update applied at action: Q(S={S},A={A}) {q_predict} -> {q_after} (target={q_target})"
            )
            S = S_
            # If the DLA process signals termination, end the loop and save results
            if DLA_terminate == 1:
                # DLA process signaled termination; proceed to save artifacts
                is_terminated = True
                save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
                save_question_lib(save_filename, self.question_lib)
                logger.info(f"Saved question library to {save_filename} after DLA termination.")
                # log_question("Goodbye. We will do the screening in another time. 886")
                logger.info("Goodbye. We will do the screening in another time. 886")        # Save results if terminated
        if is_terminated:
            # Persist question library snapshot upon termination
            save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
            save_question_lib(save_filename, self.question_lib)
            logger.info(f"Saved question library to {save_filename} after session termination.")
            
            # Save Q tables (in parallel with existing results)
            qdir = os.path.join(DATA_DIR, "q_tables")
            qfile = self.q_table_file
            self.item_q_table = active_q_table
            dir_preexisted = os.path.exists(qdir)
            if not dir_preexisted:
                os.makedirs(qdir, exist_ok=True)
                logger.info(f"Created q_tables directory at {qdir}.")
            file_preexisted = os.path.exists(qfile)
            self.item_q_table.to_csv(qfile)
            if file_preexisted:
                logger.info(f"Updated item Q table for subject {SUBJECT_ID} at {qfile}.")
            else:
                logger.info(f"Created new item Q table for subject {SUBJECT_ID} at {qfile}.")
            write_rl_trace(self.rl_trace_file, self.rl_trace_records)
            logger.info(f"Saved RL trace for subject {SUBJECT_ID} at {self.rl_trace_file}.")

            write_session_summary(
                self.session_summary_file,
                {
                    "run_id": self.run_id,
                    "subject_id": SUBJECT_ID,
                    "timestamp": now_iso(),
                    "screening_turn_count": len(self.rl_trace_records),
                    "rl_trace_file": self.rl_trace_file,
                    "q_table_file": qfile,
                    "cbt_candidates": cbt_candidates_from_question_lib(self.question_lib),
                },
            )
            logger.info(f"Saved session summary at {self.session_summary_file}.")

        # Run CBT after the screening loop concludes
        self.session_control.mark_cbt()
        self._queue_pending_screening_validation_for_cbt()
        run_cbt(self.question_lib, session_control=self.session_control)
        logger.info("Completed CBT flow.")
        self._persist_runtime_question_lib()
        # Persist question_lib again to capture CBT notes
        save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
        save_question_lib(save_filename, self.question_lib)
        logger.info(f"Saved question library with CBT notes to {save_filename}.")

        # Generate final results for this session
        generate_results(self.question_lib, self.new_response)
        logger.info("Generated final results for this session.")

        try:
            cbt_used, cbt_summary = self._detect_cbt_summary()
            write_session_summary(
                self.session_summary_file,
                {
                    "run_id": self.run_id,
                    "subject_id": SUBJECT_ID,
                    "timestamp": now_iso(),
                    "screening_turn_count": len(self.rl_trace_records),
                    "rl_trace_file": self.rl_trace_file,
                    "q_table_file": self.q_table_file,
                    "report_file": REPORT_FILE,
                    "notes_file": NOTES_FILE,
                    "cbt_used": cbt_used,
                    "cbt_summary": cbt_summary,
                    "cbt_candidates": cbt_candidates_from_question_lib(self.question_lib),
                },
            )
            logger.info(f"Updated session summary after CBT at {self.session_summary_file}.")
        except Exception as e:
            logger.warning(f"Failed to update session summary after CBT: {e}")

        # Deliver concluding message (LLM-generated) only if CBT was NOT used
        # If CBT ran, its own final message is the user-visible conclusion. Avoid double messages due to lock semantics.
        try:
            cbt_used, cbt_summary = self._detect_cbt_summary()
            if not cbt_used:
                sys_prompt = (
                    "You are a warm, concise, and professional therapist-assistant.\n\n"
                    "Background: This message appears at the end of a brief screening/CBT session.\n"
                    "Goal: Generate a short closing message for the user.\n\n"
                    "Inputs you may receive:\n"
                    "- cbt_used: whether CBT was conducted in this session (true/false).\n"
                    "- session_summary: brief bullet/lines from the session (if available).\n\n"
                    "Instructions:\n"
                    "- If cbt_used is true: Congratulate the user for working on CBT today, acknowledge their effort, and say goodbye.\n"
                    "- If cbt_used is false: Indicate there is no area of concern identified today and say goodbye.\n"
                    "- 1–2 sentences only.\n"
                    "- Friendly, non-judgmental tone.\n"
                    "- No headers or labels; output the final message directly.\n"
                )
                user_payload = (
                    f"cbt_used: {str(cbt_used).lower()}\n" + (f"session_summary:\n{cbt_summary}" if cbt_summary else "")
                )
                closing = llm_complete(sys_prompt, user_payload).strip()
                time.sleep(0.5)
                log_system_message(closing)
            else:
                logger.info("CBT delivered its own closing; skipping RL-level closing to avoid double message.")
        except Exception as e:
            logger.warning(f"Concluding message generation failed: {e}")
            # Only attempt fallback if CBT was not used
            cbt_used, _ = self._detect_cbt_summary()
            if not cbt_used:
                fallback = "Thank you for your time today. Take care, and goodbye."
                time.sleep(0.5)
                log_system_message(fallback)

    def _detect_cbt_summary(self) -> tuple:
        """Return (cbt_used, summary_str) by scanning question_lib notes for CBT markers."""
        try:
            lines = []
            cbt_used = False
            for i in range(1, len(self.question_lib) + 1):
                for j in range(1, len(self.question_lib[str(i)]) + 1):
                    entry = self.question_lib[str(i)][str(j)]
                    notes = entry.get("notes", [])
                    for note in notes:
                        if isinstance(note, list) and any((isinstance(x, str) and x.startswith("CBT_")) for x in note):
                            cbt_used = True
                            for x in note:
                                if isinstance(x, str) and (
                                    x.startswith("CBT_dimension:") or
                                    x.startswith("CBT_statement:") or
                                    x.startswith("CBT_unhelpful_thoughts:") or
                                    x.startswith("CBT_challenge:") or
                                    x.startswith("CBT_reframe:") or
                                    x.startswith("CBT_stage:")
                                ):
                                    lines.append(x)
            summary = "\n".join(lines[-8:]) if lines else ""
            return cbt_used, summary
        except Exception:
            return False, ""

    def _unlock_question_if_stuck(self) -> None:
        """If Question_Lock remains set after a system message, clear it to avoid blocking."""
        try:
            with RECORD_LOCK:
                df = pd.read_csv(RECORD_CSV)
                if int(df.loc[0, "Question_Lock"]) == 1:
                    df.loc[0, "Question_Lock"] = 0
                    tmp_path = f"{RECORD_CSV}.{os.getpid()}.{time.time_ns()}.tmp"
                    df.to_csv(tmp_path, index=False)
                    os.replace(tmp_path, RECORD_CSV)
                    logger.info("Force-unlocked Question_Lock after system message.")
        except Exception as e:
            logger.warning(f"Failed to force-unlock Question_Lock: {e}")
