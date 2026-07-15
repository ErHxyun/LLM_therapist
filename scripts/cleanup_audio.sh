#!/usr/bin/env bash
set -euo pipefail

IPC_PATH="${CAITI_MUSIC_IPC_PATH:-/tmp/caiti_mpv_music.sock}"

# Stop orphan mpv background music processes that belong to CaiTI.
pkill -f -- "--input-ipc-server=${IPC_PATH}" || true

# Stop command-backend waiting music players that are reading from the repo's
# background audio assets. This intentionally avoids broad `pkill aplay`.
pkill -f -- "aplay -q .*assets/audio/" || true

# Remove stale mpv IPC socket if it still exists.
rm -f "${IPC_PATH}" || true
