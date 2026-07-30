# CaiTI Headless Systemd Draft

These unit files are a deployment draft for turning CaiTI into a headless,
button-driven appliance on the Jetson.

They assume the current local paths:

- Repo: `/home/xiyun/Desktop/Projects/LLM_therapist`
- Emotion service: `/home/xiyun/Desktop/Projects/emo_module`
- LLM server: `127.0.0.1:8890`
- Emotion server: `127.0.0.1:8000`

## What These Units Do

- `caiti-llm.service`
  Starts the persistent CaiTI LLM server.
- `caiti-pinmux.service`
  Applies the required Jetson `busybox devmem` pinmux writes at boot.
- `caiti-emotion.service`
  Starts the external FastAPI emotion service.
- `caiti-app.service`
  Starts the voice app as a system service. This is kept as a fallback, but on
  Jetson audio is often more reliable when the app runs inside the real user
  session.
- `deploy/systemd-user/caiti-app.service`
  Starts the voice app as a `systemd --user` service, which is the recommended
  mode when TTS or background audio is silent under the system service even
  though the logs show playback completed.

## Important Notes

1. These units expect `python` to resolve correctly for the service user.
   If your Jetson uses a conda env or a venv, replace `python` in `ExecStart`
   with the absolute interpreter path.

2. The voice app remains resident after a session and returns to
   `ready_idle`. `Restart=always` is retained so a confirmed full shutdown or
   unexpected process failure restarts the appliance controller.

3. The monitor remains local by default at `http://127.0.0.1:8765`.
4. To publish the monitor on a real domain with login protection, use the
   Cloudflare Tunnel deployment in `deploy/cloudflared/README.md`.

## Recommended Split

Keep infrastructure services as system units:

- `caiti-pinmux.service`
- `caiti-llm.service`
- `caiti-emotion.service`

Run the voice app as a user unit:

- `deploy/systemd-user/caiti-app.service`

This split keeps GPIO pinmux and local servers available at boot, while the
actual speaking/listening app runs inside the logged-in user's audio session.

## Install System Services

```bash
cd /home/xiyun/Desktop/Projects/LLM_therapist
sudo cp deploy/systemd/caiti-llm.service /etc/systemd/system/
sudo cp deploy/systemd/caiti-pinmux.service /etc/systemd/system/
sudo cp deploy/systemd/caiti-emotion.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo chmod +x scripts/cleanup_audio.sh scripts/setup_jetson_pinmux.sh
sudo systemctl enable caiti-pinmux.service caiti-llm.service caiti-emotion.service
sudo systemctl start caiti-pinmux.service caiti-llm.service caiti-emotion.service
```

## Install App As User Service

```bash
cd /home/xiyun/Desktop/Projects/LLM_therapist
mkdir -p ~/.config/systemd/user
cp deploy/systemd-user/caiti-app.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable caiti-app.service
systemctl --user start caiti-app.service
```

If you previously enabled the system-level app service, disable it first so two
copies do not fight each other:

```bash
sudo systemctl disable --now caiti-app.service
```

## Check Status

```bash
sudo systemctl status caiti-pinmux.service
sudo systemctl status caiti-llm.service
sudo systemctl status caiti-emotion.service
systemctl --user status caiti-app.service
curl http://127.0.0.1:8890/health
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8765/status
```

## Manual Pinmux Check

```bash
sudo systemctl start caiti-pinmux.service
sudo systemctl status caiti-pinmux.service --no-pager -l
sudo busybox devmem 0x2440020
sudo busybox devmem 0x243D020
sudo busybox devmem 0x243D010
sudo busybox devmem 0x243D000
```

## Logs

```bash
tail -f data/logs/systemd-caiti-llm.log
tail -f data/logs/systemd-caiti-emotion.log
tail -f data/logs/systemd-caiti-app.log
```

For remote monitor publishing:

```bash
tail -f data/logs/systemd-caiti-cloudflared.log
```

## Notes

- `caiti-pinmux.service` runs as root because `devmem` needs elevated access.
- `caiti-app.service` requires `caiti-pinmux.service`; only the pinmux unit
  performs the `devmem` writes, so normal boot does not execute them twice.
- `caiti-app.service` also requires `caiti-emotion.service` and keeps the
  emotion health check as a hard start gate.
- If your Jetson image does not have `busybox`, install it first.
- If the app is silent under `sudo systemctl start caiti-app.service` but the
  same TTS command is audible from a normal terminal, move the app to the user
  service above. Exporting `HOME`, `DISPLAY`, and `PULSE_SERVER` alone is often
  not enough because the process is still outside the real user audio session.

## Current Session Lifecycle

The deployed app preloads once, enters `ready_idle`, accepts a start request
from Pin 37, the local monitor, or `scripts/caiti_control.py start`, runs one
session, clears participant-specific live state, and returns to `ready_idle`.
Participant sessions created at or after the configured cutover are isolated
under `data/users/<participant>/sessions/<session_id>/`.
