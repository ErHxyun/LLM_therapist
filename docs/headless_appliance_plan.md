# Headless Appliance Plan

## Goal

Turn CaiTI into a button-driven Jetson appliance that:

1. boots without needing keyboard, mouse, or screen
2. starts its services automatically
3. preloads the model stack before the first user session
4. begins a session only when the user presses the session button
5. returns to an idle-ready state after the session ends

## Current Behavior

Today the voice app already waits for a session-button start, but it still:

- preloads the LLM after the button press
- warms up STT/TTS after the button press
- exits after the session finishes

That means the user still experiences startup cost on the first press, and the
process does not remain in memory for the next user.

## Target Runtime Lifecycle

```text
boot
-> start llm service
-> start emotion service
-> start voice app
-> cleanup stale audio
-> preload llm
-> warm up stt
-> warm up tts
-> warm up intermission tts
-> phase = ready_idle
-> wait for session button
-> run one session
-> cleanup session resources
-> phase = ready_idle
-> wait for next session
```

## Target Button Semantics

### Idle

- short press: start a new session
- long press: ignore

### Active Session

- short press: pause
- short press again: resume
- long press during screening: skip to CBT
- long press later: shutdown confirmation

### Busy States

Ignore button actions while in:

- `preloading`
- `loading`
- `cleanup`

This avoids accidental double-taps turning into pause requests immediately
after session start.

## Proposed New Phase Names

The status monitor currently exposes only a small phase set. Add:

- `preloading`
- `ready_idle`
- `cleanup`

These are useful for both the local monitor and any future remote dashboard.

## Proposed Code Changes

### 1. `LLM_therapist_Voice_Application.py`

Split startup into two layers:

#### Process-level startup, once

- create record bridge
- build monitor, LEDs, STT, TTS, music, intermission runner, session control
- start button listeners
- start voice I/O loop
- cleanup stale audio
- preload LLM
- warm up STT/TTS
- set phase to `ready_idle`

#### Session loop, many times

```python
while True:
    session_control.reset_for_next_session()
    set_phase("ready_idle")
    wait_for_start_button()
    set_phase("loading")
    prepare_new_session_context()
    music.start()
    mark_screening()
    HandlerRL(session_control=session_control).run()
    cleanup_after_session()
```

### 2. `src/session/control.py`

Add a reset method:

```python
def reset_for_next_session(self) -> None:
    ...
```

Reset at least:

- `_started`
- `_started_event`
- `_paused`
- `_pause_requested`
- `_skip_to_cbt_requested`
- `_shutdown_confirm_requested`
- `_awaiting_shutdown_confirmation`
- `_shutdown_event`
- `_phase`

Also add logic to ignore presses in `preloading`, `loading`, and `cleanup`.

### 3. `src/utils/session_event_logger.py`

The current session id defaults to a process-level value. That is fine for a
single-shot process, but not for a long-running appliance.

Add a mutable session id entry point:

```python
def set_session_id(session_id: str) -> None:
    ...
```

Then generate a new `session_id` at the start of each session.

### 4. `src/utils/io_record.py`

Add a reset helper so the record bridge returns to a clean state before the
next session begins.

Possible helper:

```python
def reset_record_state() -> None:
    ...
```

That helper should clear:

- `Question`
- `Resp`
- `Question_Lock`
- `Resp_Lock`
- pending prefix buffers

### 5. `src/voice/music.py`

Keep the existing `stop()` behavior, but add a public cleanup helper for
startup and shutdown safety:

```python
def force_cleanup_audio_processes() -> None:
    ...
```

This helper should:

- kill orphan `mpv` processes that match the CaiTI IPC socket
- kill music-only `aplay` processes for `assets/audio/*`
- remove stale IPC socket files

## Authentication and Boot

Do not rely on disabling the Jetson password.

Recommended model:

- keep the system password
- run the services with `systemd`
- optionally use SSH key-based admin access
- avoid depending on desktop auto-login

This gives you automatic startup without weakening the device account.

## Music Cleanup Strategy

Use three cleanup layers:

1. startup cleanup before the voice app begins
2. normal cleanup when a session ends
3. systemd `ExecStopPost` cleanup if the process crashes or is killed

That should eliminate most cases where background music survives beyond the
session.

## Suggested Build Order

1. add systemd deployment files and cleanup script
2. add `ready_idle` / `preloading` / `cleanup` monitor phases
3. make the app session id resettable
4. add session-control reset
5. convert the main app to a persistent session loop
6. test repeated sessions without restarting the process

## Success Criteria

- power on Jetson, wait for preload
- no login required for CaiTI to become available
- first button press starts the session immediately
- session end returns to `ready_idle`
- second button press starts a fresh session
- no orphan waiting-music process remains after stop or crash
