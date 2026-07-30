#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def request_start(base_url: str) -> int:
    url = f"{base_url.rstrip('/')}/api/session/start"
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        print(payload.get("message", "Session start was rejected."), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unable to contact CaiTI monitor: {exc}", file=sys.stderr)
        return 2
    print(payload.get("message", "Session start requested."))
    return 0 if payload.get("accepted") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Control the local CaiTI voice application.")
    parser.add_argument("command", choices=["start"])
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765",
        help="CaiTI monitor base URL",
    )
    args = parser.parse_args()
    if args.command == "start":
        return request_start(args.url)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
