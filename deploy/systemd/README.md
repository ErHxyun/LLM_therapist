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

2. The current voice app still exits after a session finishes. With these
   units, systemd will restart it automatically, but that is still different
   from a true in-process `ready_idle -> session -> ready_idle` loop.

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
- `caiti-app.service` also runs the same pinmux script in `ExecStartPre`, so an
  app restart without a reboot still reapplies the required writes.
- If your Jetson image does not have `busybox`, install it first.
- If the app is silent under `sudo systemctl start caiti-app.service` but the
  same TTS command is audible from a normal terminal, move the app to the user
  service above. Exporting `HOME`, `DISPLAY`, and `PULSE_SERVER` alone is often
  not enough because the process is still outside the real user audio session.

## Recommended Next Code Change

Move the voice app from a one-session process into a persistent loop:

1. boot and preload once
2. enter `ready_idle`
3. wait for button start
4. run one session
5. reset session state
6. return to `ready_idle`

That design is described in `docs/headless_appliance_plan.md`.
