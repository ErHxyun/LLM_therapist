#!/usr/bin/env bash
set -euo pipefail

if ! command -v busybox >/dev/null 2>&1; then
  echo "busybox is required for devmem pinmux setup" >&2
  exit 1
fi

busybox devmem 0x2440020 w 0x5
busybox devmem 0x243D020 w 0x5
busybox devmem 0x243D010 w 0x5
busybox devmem 0x243D000 w 0x5
