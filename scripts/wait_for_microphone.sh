#!/usr/bin/env bash
set -euo pipefail

device="${CAITI_STT_AUDIO_DEVICE:-plughw:CARD=Device,DEV=0}"
timeout_sec="${CAITI_MIC_WAIT_TIMEOUT_SEC:-30}"

if ! [[ "$timeout_sec" =~ ^[0-9]+$ ]] || (( timeout_sec < 1 )); then
  printf 'CAITI_MIC_WAIT_TIMEOUT_SEC must be a positive integer, got %q\n' "$timeout_sec" >&2
  exit 2
fi

deadline=$((SECONDS + timeout_sec))
while (( SECONDS < deadline )); do
  if grep -Fxq "$device" < <(arecord -L 2>/dev/null); then
    printf 'CaiTI microphone ready: %s\n' "$device"
    exit 0
  fi
  sleep 1
done

printf 'CaiTI microphone not found after %ss: %s\n' "$timeout_sec" "$device" >&2
printf 'Available ALSA capture devices:\n' >&2
arecord -L >&2 || true
exit 1
