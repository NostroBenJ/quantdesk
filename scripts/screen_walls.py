"""
screen_walls.py -- do gamma walls act as levels, and can you trade against them?

PRE-REGISTERED. Written before any wall statistic was computed. The GEX
regime screen answered a different question (does the flip predict
volatility) and this one is separate: do the large gamma STRIKES behave
like support and resistance?

It matters because unlike the regime signal, a level-based effect is
DIRECTIONALLY tradeable. "Short gamma regime" is not an entry. "ES reached
the call wall, fade it" is.

Holdout: the last 30% of sessions is SEALED. This file never reads it.

THE MECHANISM, AND WHY THE INTERACTION IS THE REAL HYPOTHESIS
-------------------------------------------------------------
A dealer long gamma hedges against the move: sells into a rally toward a
strike they are long, buying it back on the dip. That pins price -- the
wall acts as a barrier and touches should REVERT.

A dealer short gamma hedges with the move: buys as price rises. That
accelerates a break -- the wall acts as a trigger and touches should
CONTINUE.

So "do walls revert?" is the wrong question on its own; the regime should
decide the sign. W2 is therefore the hypothesis this screen is really
about, and W1 is the unconditional version tested for completeness with a
two-sided prior.

CONVERTING SPX STRIKES TO ES
---------------------------
Walls are SPX strikes; ES is a future trading at a basis to the index. The
walls are converted to a PERCENTAGE distance from the parity-implied spot,
then applied to that session's ES open. Parity gives the FORWARD and ES is
a forward, so the two are on the same footing -- using the raw index level
would put the whole basis into the level and pin every wall in the wrong
place.

CAUSALITY
---------
Same as the regime screen: a file dated D holds open interest for D-1 and
prices for D, so the signal is fully known by D's close and is applied to
session D+1. Nothing is used before it exists.

THE HYPOTHESES

W1  Unconditional. On first touch of a wall, does the rest of the session
    revert or continue? Two-sided -- the mechanism does not predict a sign
    without knowing the regime.

W2  PRIMARY. THE INTERACTION. Reversion in the long-gamma regime,
    continuation in the short-gamma regime. Directional prior, stated in
    advance, and the only prediction the mechanism actually makes.

W3  Are walls magnets? Do they get touched more often than a level the
    same distance away has any right to be?

W4  THE CONTROL, and the test that decides whether "wall" means anything.
    Every wall statistic is recomputed against a PLACEBO level: the wall
    distance from a DIFFERENT session, applied to this one. Same geometry,
    same distance distribution, no gamma information. If the real walls do
    not beat the shuffled ones, this is a study of round numbers.

Costs: an MES round trip is ~1.3bp. A reversion smaller than that is not a
trade no matter how significant it is.

Run: python scripts/screen_walls.py
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
ES_GLOB = r"C:\Users\jontr\Downloads\databento_spx\es\*.zst"

HOLDOUT_FRACTION = 0.30
PRIMARY_DTE = 7
COST_BP = 1.3
PLACEBO_SHIFT = 37          # sessions; coprime-ish with any weekly cycle

RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)
ET = dt.timezone(dt.timedelta(hours=-4))


def load_signal() -> dict[dt.date, dict]:
    """Keyed by file_date -- the last moment every input is known."""
    out: dict[dt.date, dict] = {}
    with GEX_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            flip = r.get(f"flip_{PRIMARY_DTE}")
            cw = r.get(f"call_wall_{PRIMARY_DTE}")
            pw = r.get(f"put_wall_{PRIMARY_DTE}")
            if not (flip and cw and pw and r["spot"]):
                continue
            spot = float(r["spot"])
            out[dt.date.fromisoformat(r["file_date"])] = {
                "spot": spot,
                # Distances as fractions of spot, so they transfer to ES.
                "call_pct": float(cw) / spot - 1.0,
                "put_pct": float(pw) / spot - 1.0,
                "long_gamma": spot > float(flip),
            }
    return out


def es_minutes() -> dict[dt.date, list]:
    paths = glob.glob(ES_GLOB)
    if not paths:
        raise SystemExit(f"no ES files at {ES_GLOB}")
    out = {}
    for day, bars in session_bars(read_bars(paths[0])).items():
        rth = [b for b in bars
               if RTH_OPEN <= b.ts.astimezone(ET).time() < RTH_CLOSE]
        if len(rth) >= 60:
            out[day] = rth
    return out


def touch(bars, level: float, from_above: bool):
    """First bar reaching `level`, and the move from there to the close.

    Returns (index, reversion_bp) where reversion_bp is signed so POSITIVE
    always means the level held -- price came back toward where it started
    -- regardless of which side the level sits on. Negative means the level
    broke and price kept going.
    """
    for i, b in enumerate(bars):
        hit = (b.high >= level) if from_above else (b.low <= level)
        if not hit:
            continue
        close = bars[-1].close
        move_bp = 10000.0 * (close / level - 1.0)
        # Above spot: holding means falling back, so flip the sign.
        return i, (-move_bp if from_above else move_bp)
    return None, None


def collect(signal, es, days, shift: int = 0):
    """Wall outcomes. `shift` uses another session's wall distances,
    which is the placebo: same geometry, no gamma information."""
    rows = []
    for k, d in enumerate(days):
        nxt = es_day_after(es, d)
        if nxt is None:
            continue
        src = signal[days[(k + shift) % len(days)]] if shift else signal[d]
        bars = es[nxt]
        op = bars[0].open
        for side, from_above in (("call", True), ("put", False)):
            level = op * (1.0 + src[f"{side}_pct"])
            idx, rev = touch(bars, level, from_above)
            rows.append({
                "day": d, "side": side,
                "long_gamma": signal[d]["long_gamma"],
                "touched": idx is not None,
                "reversion_bp": rev,
                "distance_bp": abs(10000.0 * src[f"{side}_pct"]),
            })
    return rows


def es_day_after(es, d):
    later = [x for x in es if x > d]
    return min(later) if later else None


def main() -> int:
    signal, es = load_signal(), es_minutes()
    days = sorted(set(signal) & {d for d in signal if es_day_after(es, d)})
    cut = int(len(days) * (1 - HOLDOUT_FRACTION))
    ins = days[:cut]
    print("{0:,} usable signal days; in-sample {1:,} ({2} .. {3}); "
          "holdout {4:,} SEALED\n".format(
              len(days), len(ins), ins[0], ins[-1], len(days) - cut))

    real = collect(signal, es, ins)
    placebo = collect(signal, es, ins, shift=PLACEBO_SHIFT)
    screen = Screen(cost_bp=COST_BP)

    def touched(rows, **filt):
        return [r for r in rows if r["touched"]
                and all(r[k] == v for k, v in filt.items())]

    # ---------------------------------------------------------------- W3
    print("W3  are walls reached at all?")
    for label, rows in (("real walls", real), ("placebo walls", placebo)):
        hit = sum(1 for r in rows if r["touched"])
        dist = st.median([r["distance_bp"] for r in rows])
        print("  {0:<16} touched {1:,} of {2:,} ({3:.1%})   "
              "median distance {4:.0f}bp".format(
                  label, hit, len(rows), hit / len(rows), dist))
    print()

    # ---------------------------------------------------------------- W1
    print("W1  unconditional: does a touched wall hold? (two-sided)")
    print("    positive = level held (price came back); negative = broke")
    buckets = {"real": [r["reversion_bp"] for r in touched(real)],
               "placebo": [r["reversion_bp"] for r in touched(placebo)]}
    for name in ("real", "placebo"):
        screen.record(name + " wall", buckets[name], descriptive=True)
    describe_split(buckets, "real", "placebo", screen=screen,
                   label="W1 real vs placebo")
    print()

    # ---------------------------------------------------------------- W2
    print("W2  PRIMARY -- the interaction")
    print("    prior: LONG gamma reverts (positive), SHORT gamma continues")
    inter = {
        "long gamma": [r["reversion_bp"]
                       for r in touched(real, long_gamma=True)],
        "short gamma": [r["reversion_bp"]
                        for r in touched(real, long_gamma=False)],
    }
    for name in ("long gamma", "short gamma"):
        screen.record(name + " touch", inter[name], descriptive=True)
    describe_split(inter, "long gamma", "short gamma", screen=screen,
                   label="W2 interaction")
    print()

    # ---------------------------------------------------------------- W4
    print("W4  CONTROL -- same split on PLACEBO walls")
    print("    if this looks like W2, the gamma information is doing nothing")
    pinter = {
        "long gamma": [r["reversion_bp"]
                       for r in touched(placebo, long_gamma=True)],
        "short gamma": [r["reversion_bp"]
                        for r in touched(placebo, long_gamma=False)],
    }
    for name in ("long gamma", "short gamma"):
        screen.record("placebo " + name, pinter[name], descriptive=True)
    describe_split(pinter, "long gamma", "short gamma", screen=screen,
                   label="W4 placebo interaction")
    print()

    print("by side, real walls only (context, not a hypothesis):")
    for side in ("call", "put"):
        rows = touched(real, side=side)
        if len(rows) > 2:
            s = st.mean(r["reversion_bp"] for r in rows)
            print("  {0:<6} n={1:<4} mean reversion {2:+.2f}bp".format(
                side, len(rows), s))

    print()
    screen.summary()
    print("\nEvery hypothesis above was written before any wall statistic "
          "was computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
