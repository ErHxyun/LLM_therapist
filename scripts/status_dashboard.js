#!/usr/bin/env node
"use strict";

const http = require("http");
const { URL } = require("url");

const HOST = process.env.CAITI_NODE_MONITOR_HOST || "127.0.0.1";
const PORT = Number.parseInt(process.env.CAITI_NODE_MONITOR_PORT || "8787", 10);
const UPSTREAM_URL = process.env.CAITI_MONITOR_URL || "http://127.0.0.1:8765";
const POLL_MS = Number.parseInt(process.env.CAITI_NODE_MONITOR_POLL_MS || "500", 10);

let cachedState = offlineState("booting");
let lastPayload = "";
const clients = new Set();

function offlineState(reason) {
  const now = new Date().toISOString();
  return {
    app: "CaiTI",
    phase: "offline",
    lights: {
      white: false,
      yellow: false,
      blue: false,
      green: false,
    },
    button: {
      last_event: "",
      updated_at: "",
    },
    started_at: now,
    updated_at: now,
    version: 0,
    source: {
      online: false,
      url: UPSTREAM_URL,
      reason,
    },
  };
}

function withSource(state, online, reason = "") {
  return {
    ...state,
    source: {
      online,
      url: UPSTREAM_URL,
      reason,
    },
  };
}

function getJson(urlString, timeoutMs = 1500) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlString);
    const transport = url.protocol === "https:" ? require("https") : http;
    const req = transport.get(
      {
        hostname: url.hostname,
        port: url.port,
        path: `${url.pathname}${url.search}`,
        timeout: timeoutMs,
      },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          body += chunk;
        });
        res.on("end", () => {
          if (res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`upstream status ${res.statusCode}`));
            return;
          }
          try {
            resolve(JSON.parse(body));
          } catch (err) {
            reject(err);
          }
        });
      }
    );
    req.on("timeout", () => {
      req.destroy(new Error("upstream timeout"));
    });
    req.on("error", reject);
  });
}

async function refreshState() {
  try {
    const state = await getJson(`${UPSTREAM_URL.replace(/\/$/, "")}/status`);
    cachedState = withSource(state, true);
  } catch (err) {
    cachedState = offlineState(err.message || "upstream unavailable");
  }

  const payload = JSON.stringify(cachedState);
  if (payload !== lastPayload) {
    lastPayload = payload;
    broadcast(payload);
  }
}

function broadcast(payload) {
  for (const res of clients) {
    res.write(`event: status\ndata: ${payload}\n\n`);
  }
}

function sendJson(res, code, value) {
  const body = JSON.stringify(value, null, 2);
  res.writeHead(code, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Access-Control-Allow-Origin": "*",
  });
  res.end(body);
}

function sendHtml(res) {
  res.writeHead(200, {
    "Content-Type": "text/html; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(HTML);
}

function handleEvents(req, res) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin": "*",
  });
  clients.add(res);
  res.write(`event: status\ndata: ${JSON.stringify(cachedState)}\n\n`);
  req.on("close", () => {
    clients.delete(res);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || `${HOST}:${PORT}`}`);
  if (req.method === "GET" && url.pathname === "/") {
    sendHtml(res);
    return;
  }
  if (req.method === "GET" && url.pathname === "/api/status") {
    sendJson(res, 200, cachedState);
    return;
  }
  if (req.method === "GET" && url.pathname === "/events") {
    handleEvents(req, res);
    return;
  }
  if (req.method === "GET" && url.pathname === "/health") {
    sendJson(res, 200, {
      ok: true,
      upstream_online: Boolean(cachedState.source && cachedState.source.online),
      upstream_url: UPSTREAM_URL,
    });
    return;
  }
  sendJson(res, 404, { error: "not found" });
});

server.listen(PORT, HOST, () => {
  console.log(`CaiTI Node monitor: http://${HOST}:${PORT}`);
  console.log(`Reading CaiTI Python monitor from ${UPSTREAM_URL}`);
});

setInterval(refreshState, Math.max(100, POLL_MS));
refreshState();

function shutdown() {
  for (const res of clients) {
    res.end();
  }
  server.close(() => process.exit(0));
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

const HTML = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Caiti Hardware Monitor</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #111827;
    }
    body {
      margin: 0;
      min-height: 100vh;
      padding: 24px;
      box-sizing: border-box;
    }
    main {
      width: min(980px, 100%);
      margin: 0 auto;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 26px;
      line-height: 1.15;
      font-weight: 750;
    }
    .connection {
      color: #4b5563;
      font-size: 14px;
      text-align: right;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .light {
      border: 1px solid #d5dae1;
      border-radius: 8px;
      background: #ffffff;
      min-height: 136px;
      padding: 16px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .dot {
      width: 36px;
      height: 36px;
      border-radius: 999px;
      background: #a8b0bc;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,.14);
    }
    .light.on .dot {
      box-shadow: 0 0 22px currentColor, inset 0 0 0 1px rgba(0,0,0,.14);
    }
    .white { color: #6b7280; }
    .white.on .dot { background: #f9fafb; }
    .yellow { color: #b7791f; }
    .yellow.on .dot { background: #facc15; }
    .blue { color: #2563eb; }
    .blue.on .dot { background: #3b82f6; }
    .green { color: #15803d; }
    .green.on .dot { background: #22c55e; }
    .label {
      font-size: 18px;
      font-weight: 750;
      text-transform: capitalize;
      margin-top: 16px;
    }
    .meaning {
      color: #4b5563;
      font-size: 13px;
      line-height: 1.35;
      margin-top: 4px;
    }
    .panel {
      margin-top: 14px;
      border: 1px solid #d5dae1;
      border-radius: 8px;
      background: #ffffff;
      padding: 8px 16px;
    }
    .row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 42px;
      border-bottom: 1px solid #e5e7eb;
      font-size: 15px;
    }
    .row:last-child { border-bottom: 0; }
    .key { color: #4b5563; }
    .value {
      font-weight: 650;
      text-align: right;
      overflow-wrap: anywhere;
    }
    .offline {
      color: #b91c1c;
    }
    @media (max-width: 720px) {
      body { padding: 16px; }
      header { align-items: flex-start; flex-direction: column; }
      .connection { text-align: left; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Caiti Hardware Monitor</h1>
      <div class="connection" id="connection">Connecting...</div>
    </header>
    <section class="grid" id="lights"></section>
    <section class="panel">
      <div class="row"><span class="key">Phase</span><span class="value" id="phase">-</span></div>
      <div class="row"><span class="key">Button</span><span class="value" id="button">-</span></div>
      <div class="row"><span class="key">Updated</span><span class="value" id="updated">-</span></div>
      <div class="row"><span class="key">Python source</span><span class="value" id="source">-</span></div>
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
      card.className = "light " + color;
      card.id = "light-" + color;
      card.innerHTML = '<div><div class="dot"></div><div class="label">' + color + '</div></div><div class="meaning">' + meanings[color] + '</div>';
      lights.appendChild(card);
    }
    function render(state) {
      for (const color of ["white", "yellow", "blue", "green"]) {
        const active = Boolean(state.lights && state.lights[color]);
        const el = document.getElementById("light-" + color);
        if (el) el.classList.toggle("on", active);
      }
      const source = state.source || {};
      document.getElementById("phase").textContent = state.phase || "-";
      document.getElementById("button").textContent = (state.button && state.button.last_event) || "-";
      document.getElementById("updated").textContent = state.updated_at || "-";
      document.getElementById("source").textContent = (source.online ? "online · " : "offline · ") + (source.url || "-") + (source.reason ? " · " + source.reason : "");
      document.getElementById("source").classList.toggle("offline", !source.online);
      document.getElementById("connection").textContent = source.online ? "Live through Node.js" : "Waiting for Python monitor";
    }
    async function poll() {
      const response = await fetch("/api/status", { cache: "no-store" });
      render(await response.json());
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
</html>`;
