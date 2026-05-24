#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CAITI_DEVICE_MAP="${CAITI_DEVICE_MAP:-cuda:0}"

exec python LLM_therapist_Voice_Application.py "$@"
