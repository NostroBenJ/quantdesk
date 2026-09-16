"""
screen_orb.py -- opening range breakout on ES, and whether gamma gates it.

PRE-REGISTERED. Written before any ORB statistic was computed and before
the extended ES history was ordered.

Holdout: the last 30% of sessions is SEALED. This file never reads it.

WHY ORB IS WORTH RE-TESTING RATHER THAN ASSUMED DEAD
----------------------------------------------------
The inherited repo wrote ORB off as "worse than random". That test ran on
GOLD, on the old harness, sized off a retired 100K account, with no costs
and no control. ORB is an equity-index strategy and it has never been
tested on ES here.

It is also a contested PUBLISHED result rather than folklore: Zarattini &
Aziz (2023) reported strong ORB numbers on QQQ and were criticised
specifically on cost assumptions. A specific rule, a known objection, and
a costed framework is exactly what this harness is for. So the primary
test uses the PUBLISHED specification -- a 5-minute opening range -- not a
variant of my own, because testing my variant would be testing my idea
rather than the claim that carries the prior.

A CORRECTION TO THE PLAN THIS FILE WAS WRITTEN FROM
---------------------------------------------------
The interaction with gamma regime was going to be the primary. Writing it
down exposed why it cannot be: the gamma series covers 251 sessions and
BUYING MORE ES CANNOT EXTEND IT. Sixteen years of ES gives roughly 4,000
ORB trades for O1 and O3, and still only ~250 for the interaction, split
two ways.

So the primary is O1, the unconditional claim, which is where the data and
the published prior both are. The interaction stays as O2 and is labelled
sample-limited in advance, so a null there is read as "could not resolve"
rather than "refuted" -- the distinction that the VRP screen turned on.

THE HYPOTHESES

O1  PRIMARY. Unconditional ORB on ES, net of costs. Two-sided: the
    published prior says positive, the crowding prior says the edge is
    gone. Both are defensible ex ante and neither is assumed.

O2  Gamma interaction, SAMPLE-LIMITED to the 251 gamma sessions. The
    mechanism, already measured at t=+6.53: short gamma means dealers
    hedge WITH the move, so breakouts should RUN; long gamma means they
    hedge against it, so breakouts should FAIL. Directional prior.

O3  THE CONTROL, and the test that decides whether the OPENING range is
    special. Identical rules applied to an arbitrary mid-session window.
    If breaking a random 5 minutes at 12:00 pays as well as breaking the
    first 5 minutes of the day, this is a study of intraday momentum with
    an opening-bell story attached. This is the test that killed the
    gamma walls.

O4  Context, not a hypothesis: long breaks versus short breaks. Equities
    drift up, so asymmetry is expected and is reported rather than tested.

SPECIFICATION, fixed in advance
-------------------------------
* Opening range: 09:30-09:35 ET (5 minutes). Secondaries 15 and 30, both
  counted against the correction.
* Entry: the first bar whose high exceeds the range high (long) or whose
  low breaks the range low (short). Direction is the direction of the
  break. One trade per session, first break only -- no re-entry, because
  re-entry rules are a parameter and this is a test, not a fit.
* Exit: the 16:00 ET close. No stop in the primary: a stop is a free
  parameter and the cleanest test of "does the break carry information"
  does not include one. A secondary uses the opposite side of the opening
  range as a stop.
* Fills: at the break level, then costs charged. Not at the bar close,
  which would hand the strategy the rest of that minute for free.
* Costs: MES round trip, ~1.3bp of notional -- roughly $1.50 commission
  plus a tick of slippage each way on a ~$32,500 contract. Charged to
  every trade, winners included.
* Sizing: one contract throughout. Results are reported per trade in basis
  points so nothing depends on an account size.

WHAT WOULD MAKE THIS TRADEABLE, stated before seeing a number
-------------------------------------------------------------
A per-trade edge above the 1.3bp cost floor is necessary and nowhere near
sufficient. The Flex 50K box ends at -$2,000, so the question that decides
deployment is not the mean but the drawdown: what is the worst run of
losses, and does a 0.5% risk fraction survive it? That is computed in the
same Monte Carlo already used for the eval odds, and it happens AFTER the
holdout, not before.

Run: python scripts/screen_orb.py
"""

from __future__ import annotations

import csv
import datetime as dt
import glob
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.research_stats import Screen, describe_split  # noqa: E402
from quantdesk.data.sources.databento_glbx import (  # noqa: E402
    read_bars, session_bars,
)

GEX_CSV = ROOT / "data" / "spx_gex_daily.csv"
ES_GLOB = (str(Path.home()) + r"\Downloads\databento_spx\es_full\*.zst")

HOLDOUT_FRACTION = 0.30
COST_BP = 1.3
PRIMARY_RANGE_MIN = 5
RANGE_MINUTES = (5, 15, 30)
PLACEBO_START = dt.time(12, 0)     # arbitrary, fixed in advance

RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)
ET = dt.timezone(dt.timedelta(hours=-4))


def rth(bars):
    return [b for b in bars
            if RTH_OPEN <= b.ts.astimezone(ET).time() < RTH_CLOSE]


def orb_trade(bars, range_minutes: int, start: dt.time = RTH_OPEN,
              use_stop: bool = False):
    """One session's first break. Returns net basis points, or None.

    Fills at the break LEVEL, not the breaking bar's close -- filling at
    the close would hand the strategy whatever else happened in that
    minute, which is a lookahead of up to 59 seconds and is exactly the
    sort of thing that makes a backtest glow.
    """
    session = [b for b in bars if b.ts.astimezone(ET).time() >= start]
    if len(session) < range_minutes + 30:
        return None
    opening = session[:range_minutes]
    hi = max(b.high for b in opening)
    lo = min(b.low for b in opening)
    if hi <= lo:
        return None

    rest = session[range_minutes:]
    for i, b in enumerate(rest):
        long_break = b.high >= hi
        short_break = b.low <= lo
        if not (long_break or short_break):
            continue
        # A bar touching both sides is ambiguous about which came first;
        # minute bars cannot say, so the session is skipped rather than
        # resolved by a coin flip that would flatter whichever we chose.
        if long_break and short_break:
            return None
        direction = 1 if long_break else -1
        entry = hi if long_break else lo
        exit_px = rest[-1].close
        if use_stop:
            stop = lo if long_break else hi
            for later in rest[i:]:
                hit = (later.low <= stop) if long_break else (later.high >= stop)
                if hit:
                    exit_px = stop
                    break
        gross = 10000.0 * direction * (exit_px / entry - 1.0)
        return {"net_bp": gross - COST_BP, "gross_bp": gross,
                "direction": "long" if long_break else "short"}
    return None


def load_regimes() -> dict[dt.date, bool]:
    """file_date -> long_gamma, keyed at the last moment it is known."""
    out: dict[dt.date, bool] = {}
    if not GEX_CSV.exists():
        return out
    with GEX_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("flip_7") and r["spot"]:
                out[dt.date.fromisoformat(r["file_date"])] = (
                    float(r["spot"]) > float(r["flip_7"]))
    return out


def main() -> int:
    paths = glob.glob(ES_GLOB)
    if not paths:
        raise SystemExit(f"no ES files at {ES_GLOB}")
    sessions = {d: rth(b) for d, b in session_bars(read_bars(paths[0])).items()}
    sessions = {d: b for d, b in sessions.items() if len(b) >= 300}
    days = sorted(sessions)
    cut = int(len(days) * (1 - HOLDOUT_FRACTION))
    ins = days[:cut]
    regimes = load_regimes()

    print("{0:,} ES sessions; in-sample {1:,} ({2} .. {3}); holdout {4:,} "
          "SEALED\n".format(len(days), len(ins), ins[0], ins[-1],
                            len(days) - cut))

    screen = Screen(cost_bp=COST_BP)

    # ---------------------------------------------------------------- O1
    print("O1  PRIMARY: unconditional ORB, net of costs (two-sided prior)")
    primary_trades = {}
    for rng in RANGE_MINUTES:
        trades = [t for t in (orb_trade(sessions[d], rng) for d in ins) if t]
        if rng == PRIMARY_RANGE_MIN:
            primary_trades = {d: t for d in ins
                              if (t := orb_trade(sessions[d], rng))}
        tag = " (primary)" if rng == PRIMARY_RANGE_MIN else ""
        screen.record("{0}min range{1}".format(rng, tag),
                      [t["net_bp"] for t in trades])
    print()

    # ---------------------------------------------------------------- O3
    print("O3  CONTROL: same rules on an arbitrary {0} window".format(
        PLACEBO_START.strftime("%H:%M")))
    print("    if this pays too, the OPENING range is not what matters")
    placebo = [t for t in (orb_trade(sessions[d], PRIMARY_RANGE_MIN,
                                     start=PLACEBO_START) for d in ins) if t]
    real = [t for t in primary_trades.values()]
    buckets = {"opening range": [t["net_bp"] for t in real],
               "midday range": [t["net_bp"] for t in placebo]}
    for k in ("opening range", "midday range"):
        screen.record(k, buckets[k], descriptive=True)
    describe_split(buckets, "opening range", "midday range", screen=screen,
                   label="O3 opening vs placebo")
    print()

    # ---------------------------------------------------------------- O2
    print("O2  gamma interaction -- SAMPLE-LIMITED to the gamma series")
    print("    prior: SHORT gamma breakouts run, LONG gamma breakouts fail")
    inter: dict[str, list[float]] = {}
    for d, t in primary_trades.items():
        if d in regimes:
            key = "long gamma" if regimes[d] else "short gamma"
            inter.setdefault(key, []).append(t["net_bp"])
    if min((len(v) for v in inter.values()), default=0) < 20:
        print("    too few overlapping sessions ({0}) -- not testable yet"
              .format({k: len(v) for k, v in inter.items()}))
    else:
        for k in ("short gamma", "long gamma"):
            screen.record(k + " ORB", inter[k], descriptive=True)
        describe_split(inter, "short gamma", "long gamma", screen=screen,
                       label="O2 gamma interaction")
    print()

    # ---------------------------------------------------------------- O4
    print("O4  context, not a hypothesis: break direction")
    for side in ("long", "short"):
        rows = [t["net_bp"] for t in real if t["direction"] == side]
        if len(rows) > 2:
            print("  {0:<7} n={1:<5} mean {2:+.2f}bp  gross {3:+.2f}bp".format(
                side, len(rows), st.mean(rows), st.mean(rows) + COST_BP))
    print("  trades fired on {0:.0%} of in-sample sessions".format(
        len(real) / len(ins)))

    print()
    screen.summary()
    print("\nSpecification and every hypothesis were fixed before the first "
          "ORB statistic was computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
