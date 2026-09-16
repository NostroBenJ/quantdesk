# Deploying the recorder and dashboard

Two processes, one purpose: **never lose another session of options positioning
history.** Nothing here trades, and nothing here needs an API key.

## What runs, and why only this

| unit | schedule | job |
|---|---|---|
| `quantdesk-recorder.timer` | Mon–Fri 12:45, 17:00, 21:15 UTC | fetch the CBOE chain, store it, write `health.json` |
| `quantdesk-dash.service` | always | read-only research review UI on `127.0.0.1:8010` |

There is deliberately **no trading process**. No signal module exists yet, and
the project's own rule is that nothing reaches live execution without a
validated out-of-sample record plus a human sign-off. A recorder running 24/7
does not touch that rule.

## The failure this deployment is designed against

The previous capture pipeline died on **2026-08-12** and nobody noticed for nine
days. Three scheduled tasks were terminated mid-run (`0xC000013A` — the machine
slept), then silently disabled. Open interest snapshots simply stopped.

Three things here address that directly:

1. **`Persistent=true`** on the timer — if the box was down at the scheduled
   time, the run happens on the next boot rather than being skipped.
2. **`health.json` is written on failure too**, so silence is never mistaken for
   success.
3. **`missing_sessions`** in the health file lists weekdays with no snapshot.
   A dead recorder shows up as a growing list on the dashboard, not as an
   absence of news.

## Where to host it

The account behind this is small, so treat the box as a **research expense, not
a trading cost** — at $500–1,000 of capital, $10/month is 1–2% of the account
per month, which no realistic return covers. Options in order of what I'd pick:

| option | cost | notes |
|---|---|---|
| **Oracle Cloud Always Free** | $0 | 4 ARM cores / 24 GB, genuinely free indefinitely. Best fit. |
| Existing AWS Lightsail | $3.50–5/mo | Simplest if the instance is already running and paid for. |
| GitHub Actions cron | $0 | Elegant for the recorder alone (it is a 1-second job), but SQLite in git ages badly and there is nowhere to host the dashboard. |
| Home box / Raspberry Pi | $0 | Fine, but it is the sleeping-laptop failure again unless it genuinely stays on. |

CPU is nothing — 1 second per run. Disk is not nothing, and the measured figure
is bigger than it sounds: one SPY snapshot is ~11,000 stored rows at 226 bytes,
so **2.5 MB per snapshot**.

| schedule | disk/year |
|---|---|
| 1×/day, post-close only | **0.6 GB** |
| 3×/day (the shipped timer) | **1.9 GB** |

Worth deciding deliberately, because **open interest only updates once daily,
after the close.** The extra two runs buy intraday quote and IV detail, not more
OI history. If the GEX work is the priority, drop to the 21:15 UTC run alone and
save two thirds of the disk. If you want intraday IV term structure later, keep
all three — but know that is what you are paying for.

On a free Oracle tier (roughly 50–200 GB) either is fine for years.

## Remote access — do not expose the dashboard

The dashboard has **no authentication of any kind**. It is a research tool, not
a product, and `run_dash.py` refuses to bind anything except loopback or a
tailnet address.

Use **Tailscale** (free, private, has a phone app):

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# then bind the dash to the tailnet address the CLI reports (100.x.y.z)
```

Or an SSH tunnel, if you only ever look from a laptop:

```bash
ssh -L 8010:127.0.0.1:8010 user@host
```

Either way nothing is published to the internet, and there is no auth code to
get wrong.

## Install

```bash
# on the server, as root
adduser --system --group --home /opt/quantdesk quantdesk
apt-get update && apt-get install -y python3-venv rsync

# from your laptop
rsync -av --exclude .git --exclude runs --exclude data \
      ~/Downloads/quantdesk/ user@host:/opt/quantdesk/

# back on the server
cd /opt/quantdesk
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn jinja2      # recorder core needs nothing
mkdir -p data runs && chown -R quantdesk:quantdesk /opt/quantdesk

cp deploy/*.service deploy/*.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now quantdesk-recorder.timer
systemctl enable --now quantdesk-dash.service
```

## Verify

```bash
systemctl list-timers quantdesk-recorder.timer   # next run time
.venv/bin/python -m quantdesk.data.recorder SPY  # force one now
cat data/health.json | python3 -m json.tool
journalctl -u quantdesk-recorder --since today
```

A healthy first run looks like:

```
SPY: session 2026-08-20 spot 762.60 stored 10,944/14,100 OI 22,074,847
```

## Check it weekly

The one thing that matters:

```bash
python3 -c "import json;print(json.load(open('/opt/quantdesk/data/health.json'))['symbols']['SPY']['missing_sessions'])"
```

Empty list means the recorder is doing its job. Anything else means sessions
were lost, and they cannot be recovered — only recorded going forward.
