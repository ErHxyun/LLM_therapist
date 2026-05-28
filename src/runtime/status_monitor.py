"""Small local HTTP/SSE monitor for CaiTI runtime state."""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from src.utils import config_loader
from src.utils.log_util import get_logger

logger = get_logger("StatusMonitor")


@dataclass(frozen=True)
class StatusMonitorSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class NullStatusMonitor:
    def start(self) -> bool:
        return False

    def stop(self) -> None:
        return

    @property
    def url(self) -> str:
        return ""

    def snapshot(self) -> dict[str, Any]:
        return {}

    def set_phase(self, phase: str) -> None:
        return

    def set_light(self, color: str, active: bool) -> None:
        return

    def set_button_event(self, event: str) -> None:
        return


@dataclass
class StatusMonitor:
    settings: StatusMonitorSettings
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _condition: threading.Condition = field(init=False)
    _server: ThreadingHTTPServer | None = field(default=None, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _version: int = field(default=0, init=False)
    _state: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)
        now = _now_iso()
        self._state = {
            "app": "CaiTI",
            "phase": "booting",
            "lights": {
                "white": False,
                "yellow": False,
                "blue": False,
                "green": False,
            },
            "button": {
                "last_event": "",
                "updated_at": "",
            },
            "started_at": now,
            "updated_at": now,
            "version": 0,
        }

    @property
    def url(self) -> str:
        if self._server is None:
            return f"http://{self.settings.host}:{self.settings.port}"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> bool:
        if not self.settings.enabled:
            logger.info("Status monitor disabled.")
            return False
        with self._lock:
            if self._started:
                return True
            try:
                server = _StatusThreadingHTTPServer((self.settings.host, self.settings.port), _StatusRequestHandler)
                server.status_monitor = self
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                self._server = server
                self._thread = thread
                self._started = True
            except Exception as exc:
                self._server = None
                self._thread = None
                self._started = False
                logger.warning("Status monitor failed to start: %s", exc)
                return False
        logger.info("Status monitor started at %s", self.url)
        return True

    def stop(self) -> None:
        server = None
        thread = None
        with self._lock:
            if not self._started:
                return
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
            self._started = False
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=1.0)
        logger.info("Status monitor stopped.")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def wait_for_update(self, last_version: int, timeout_sec: float = 15.0) -> dict[str, Any]:
        with self._condition:
            if self._version <= last_version:
                self._condition.wait(timeout=timeout_sec)
            return self.snapshot()

    def set_phase(self, phase: str) -> None:
        phase = str(phase or "").strip() or "unknown"

        def mutate(state: dict[str, Any]) -> None:
            state["phase"] = phase

        self._update(mutate)

    def set_light(self, color: str, active: bool) -> None:
        color = str(color or "").strip().lower()
        if color == "red":
            color = "blue"
        if color not in {"white", "yellow", "blue", "green"}:
            return

        def mutate(state: dict[str, Any]) -> None:
            state["lights"][color] = bool(active)

        self._update(mutate)

    def set_button_event(self, event: str) -> None:
        event = str(event or "").strip()

        def mutate(state: dict[str, Any]) -> None:
            state["button"]["last_event"] = event
            state["button"]["updated_at"] = _now_iso()

        self._update(mutate)

    def _update(self, mutate) -> None:
        with self._condition:
            mutate(self._state)
            self._version += 1
            self._state["version"] = self._version
            self._state["updated_at"] = _now_iso()
            self._condition.notify_all()


class _StatusRequestHandler(BaseHTTPRequestHandler):
    server_version = "CaiTIStatusMonitor/1.0"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html()
            return
        if path == "/status":
            self._send_json(self._monitor().snapshot())
            return
        if path == "/events":
            self._send_events()
            return
        self.send_error(404, "Not found")

    def log_message(self, format: str, *args) -> None:
        logger.debug("HTTP %s", format % args)

    def _monitor(self) -> StatusMonitor:
        return self.server.status_monitor

    def _send_json(self, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_body(body)

    def _send_html(self) -> None:
        body = _HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_body(body)

    def _write_body(self, body: bytes) -> None:
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def _send_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        last_version = -1
        try:
            while True:
                snapshot = self._monitor().wait_for_update(last_version)
                last_version = int(snapshot.get("version", last_version))
                payload = json.dumps(snapshot, ensure_ascii=False)
                self.wfile.write(f"event: status\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return


class _StatusThreadingHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        _, exc, _ = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError, OSError)):
            logger.debug("Status monitor client disconnected: %s", exc)
            return
        super().handle_error(request, client_address)

    def shutdown_request(self, request) -> None:
        try:
            super().shutdown_request(request)
        except OSError:
            return


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_status_monitor_settings() -> StatusMonitorSettings:
    return StatusMonitorSettings(
        enabled=config_loader.MONITOR_ENABLED,
        host=config_loader.MONITOR_HOST,
        port=config_loader.MONITOR_PORT,
    )


def build_status_monitor() -> StatusMonitor | NullStatusMonitor:
    settings = build_status_monitor_settings()
    if not settings.enabled:
        return NullStatusMonitor()
    return StatusMonitor(settings)


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CaiTI Monitor</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7f8;
      color: #111827;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      box-sizing: border-box;
    }
    main {
      width: min(760px, 100%);
    }
    h1 {
      font-size: 28px;
      margin: 0 0 8px;
      font-weight: 700;
    }
    .meta {
      color: #4b5563;
      margin-bottom: 24px;
      font-size: 15px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .light {
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 18px 14px;
      background: #ffffff;
      min-height: 110px;
      box-sizing: border-box;
    }
    .dot {
      width: 34px;
      height: 34px;
      border-radius: 999px;
      margin-bottom: 16px;
      background: #9ca3af;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,.12);
    }
    .light.on .dot {
      box-shadow: 0 0 22px currentColor, inset 0 0 0 1px rgba(0,0,0,.12);
    }
    .white { color: #6b7280; }
    .white.on .dot { background: #f9fafb; }
    .yellow { color: #ca8a04; }
    .yellow.on .dot { background: #facc15; }
    .blue { color: #2563eb; }
    .blue.on .dot { background: #3b82f6; }
    .green { color: #16a34a; }
    .green.on .dot { background: #22c55e; }
    .label {
      font-size: 18px;
      font-weight: 700;
      text-transform: capitalize;
    }
    .meaning {
      color: #4b5563;
      font-size: 13px;
      margin-top: 6px;
      line-height: 1.35;
    }
    .panel {
      margin-top: 18px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      background: #ffffff;
      padding: 16px;
    }
    .row {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 8px 0;
      border-bottom: 1px solid #e5e7eb;
      font-size: 15px;
    }
    .row:last-child { border-bottom: 0; }
    .key { color: #4b5563; }
    .value { font-weight: 650; text-align: right; }
    @media (max-width: 640px) {
      body { padding: 16px; place-items: start; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <main>
    <h1>CaiTI Monitor</h1>
    <div class="meta" id="connection">Connecting...</div>
    <section class="grid" id="lights"></section>
    <section class="panel">
      <div class="row"><span class="key">Phase</span><span class="value" id="phase">-</span></div>
      <div class="row"><span class="key">Button</span><span class="value" id="button">-</span></div>
      <div class="row"><span class="key">Updated</span><span class="value" id="updated">-</span></div>
    </section>
  </main>
  <script>
    const meanings = {
      white: "Project process is running",
      yellow: "Session has begun",
      blue: "Therapist is speaking",
      green: "Client is speaking"
    };
    const lights = document.getElementById("lights");
    for (const color of ["white", "yellow", "blue", "green"]) {
      const card = document.createElement("article");
      card.className = `light ${color}`;
      card.id = `light-${color}`;
      card.innerHTML = `<div class="dot"></div><div class="label">${color}</div><div class="meaning">${meanings[color]}</div>`;
      lights.appendChild(card);
    }
    function render(state) {
      for (const [color, active] of Object.entries(state.lights || {})) {
        const el = document.getElementById(`light-${color}`);
        if (el) el.classList.toggle("on", Boolean(active));
      }
      document.getElementById("phase").textContent = state.phase || "-";
      document.getElementById("button").textContent = (state.button && state.button.last_event) || "-";
      document.getElementById("updated").textContent = state.updated_at || "-";
      document.getElementById("connection").textContent = `Live · version ${state.version || 0}`;
    }
    async function poll() {
      const res = await fetch("/status", { cache: "no-store" });
      render(await res.json());
    }
    if ("EventSource" in window) {
      const events = new EventSource("/events");
      events.addEventListener("status", event => render(JSON.parse(event.data)));
      events.onerror = () => {
        document.getElementById("connection").textContent = "Reconnecting...";
      };
    } else {
      poll();
      setInterval(poll, 1000);
    }
  </script>
</body>
</html>
"""
