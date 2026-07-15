#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CAITI_DEVICE_MAP="${CAITI_DEVICE_MAP:-cuda:0}"

USER_ID="$(id -u)"
USER_NAME="$(id -un)"
USER_HOME="$(getent passwd "$USER_ID" | cut -d: -f6)"

export HOME="${HOME:-$USER_HOME}"
export USER="${USER:-$USER_NAME}"
export LOGNAME="${LOGNAME:-$USER_NAME}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$USER_ID}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "$XDG_RUNTIME_DIR/bus" ]]; then
  export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
fi

if [[ -z "${PULSE_SERVER:-}" && -S "$XDG_RUNTIME_DIR/pulse/native" ]]; then
  export PULSE_SERVER="unix:$XDG_RUNTIME_DIR/pulse/native"
fi

printf 'Voice app env: HOME=%s XDG_RUNTIME_DIR=%s DISPLAY=%s XAUTHORITY=%s PULSE_SERVER=%s DBUS_SESSION_BUS_ADDRESS=%s\n' \
  "$HOME" \
  "$XDG_RUNTIME_DIR" \
  "$DISPLAY" \
  "$XAUTHORITY" \
  "${PULSE_SERVER:-}" \
  "${DBUS_SESSION_BUS_ADDRESS:-}"

exec python LLM_therapist_Voice_Application.py "$@"
