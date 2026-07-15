# CaiTI Remote Monitor via Cloudflare Tunnel

This deployment path publishes the CaiTI research monitor on a real domain such
as `monitor.example.com` without exposing Jetson port `8765` directly to the
public Internet.

Recommended architecture:

1. `caiti-app.service` keeps the monitor on `127.0.0.1:8765`
2. `cloudflared` publishes that local origin
3. Cloudflare Access protects the domain with login policies

This is the safest default for a monitor that can display participant IDs,
responses, scores, emotion summaries, and session history.

## Prerequisites

- A domain already managed in Cloudflare DNS
- A Cloudflare Zero Trust account
- `cloudflared` installed on the Jetson
- The CaiTI app already working locally

Verify local health first:

```bash
curl http://127.0.0.1:8765/status
```

## 1. Keep the monitor local-only

If you previously changed the monitor to `0.0.0.0`, install the provided
systemd drop-in so the published origin is only reachable through
`cloudflared`:

```bash
sudo mkdir -p /etc/systemd/system/caiti-app.service.d
sudo cp deploy/systemd/caiti-app-monitor-local.conf \
  /etc/systemd/system/caiti-app.service.d/monitor-local.conf
sudo systemctl daemon-reload
sudo systemctl restart caiti-app.service
```

Then confirm:

```bash
curl http://127.0.0.1:8765/status
```

## 2. Create the Cloudflare tunnel

Authenticate once:

```bash
cloudflared tunnel login
```

Create a named tunnel:

```bash
cloudflared tunnel create caiti-monitor
```

That command returns a tunnel UUID and writes credentials under
`~/.cloudflared/`.

## 3. Install the tunnel config

Copy the example config and fill in:

- `YOUR_TUNNEL_UUID`
- your real hostname, for example `monitor.example.com`

```bash
sudo mkdir -p /etc/cloudflared
sudo cp deploy/cloudflared/caiti-monitor.yml.example /etc/cloudflared/caiti-monitor.yml
sudo nano /etc/cloudflared/caiti-monitor.yml
```

Example final config:

```yaml
tunnel: 11111111-2222-3333-4444-555555555555
credentials-file: /etc/cloudflared/11111111-2222-3333-4444-555555555555.json

ingress:
  - hostname: monitor.example.com
    service: http://127.0.0.1:8765
  - service: http_status:404
```

Copy the generated credentials file into `/etc/cloudflared/`:

```bash
sudo cp ~/.cloudflared/YOUR_TUNNEL_UUID.json /etc/cloudflared/
sudo chmod 600 /etc/cloudflared/YOUR_TUNNEL_UUID.json
```

## 4. Route the DNS record

Create the DNS route from Cloudflare to the tunnel:

```bash
cloudflared tunnel route dns caiti-monitor monitor.example.com
```

## 5. Install the systemd service

```bash
cd /home/xiyun/Desktop/Projects/LLM_therapist
sudo cp deploy/systemd/caiti-cloudflared.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable caiti-cloudflared.service
sudo systemctl start caiti-cloudflared.service
```

Check status:

```bash
sudo systemctl status caiti-cloudflared.service --no-pager -l
tail -f data/logs/systemd-caiti-cloudflared.log
```

## 6. Protect the domain with Cloudflare Access

In the Cloudflare Zero Trust dashboard:

1. Go to `Access` -> `Applications`
2. Add a `Self-hosted` application
3. Set the application domain to `monitor.example.com`
4. Add an `Allow` policy for your team emails
5. Enable your preferred login method

Recommended first policy:

- Allow only specific email addresses or your organization email domain

Do not leave the hostname public unless you explicitly want anonymous access.

## 7. Validate end to end

From a browser:

```text
https://monitor.example.com
```

Expected behavior:

- browser hits Cloudflare
- user authenticates through Access
- Cloudflare forwards to the local Jetson monitor
- CaiTI dashboard loads and live updates continue through `/events`

## Troubleshooting

If the page opens but does not update live:

- confirm `/events` is reachable through the same hostname
- check `cloudflared` logs
- confirm the app still serves locally at `127.0.0.1:8765`

If the page is public without login:

- verify the Access application exists for the exact hostname
- verify the policy is `Allow` only for your approved users

If the tunnel is up but page returns error:

- re-check `service: http://127.0.0.1:8765`
- re-check the tunnel UUID and credentials file path
- check `sudo systemctl status caiti-app.service`
