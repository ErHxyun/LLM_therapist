#!/usr/bin/env python
"""Run a persistent CaiTI local LLM server.

Start this in one terminal, then point the voice app at it with
CAITI_LLM_SERVER_URL=http://127.0.0.1:8890.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CAITI_DEVICE_MAP", "cuda:0")

from src.local_llm.server import LLMServerSettings, serve_forever  # noqa: E402
from src.utils import config_loader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the persistent CaiTI local LLM server.")
    parser.add_argument("--host", default=config_loader.LOCAL_LLM_SERVER_HOST)
    parser.add_argument("--port", type=int, default=config_loader.LOCAL_LLM_SERVER_PORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    serve_forever(LLMServerSettings(host=args.host, port=args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
