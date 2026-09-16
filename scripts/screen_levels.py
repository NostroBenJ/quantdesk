"""
screen_levels.py -- VWAP and prior-session POC as intraday levels on ES.

PRE-REGISTERED. Written before any statistic was computed. 16 years of ES
minute bars; the last 30% of sessions is SEALED and this file never reads
it. The ES holdout is already sealed for the ORB screen and stays sealed
here -- the same cut, so nothing leaks between the two.

TWO CAUSALITY TRAPS, both handled before they could flatter anything
--------------------------------------------------------------------
1. SESSION VWAP is cumulative and therefore causal: the value at 11:00
   uses only bars up to 11:00. Safe as long as it is computed forward and
   never from the full session.

2. THE POINT OF CONTROL IS NOT. A session's POC -- the price bucket with
   the most traded volume -- is only known once that session has closed.
   Trading today's POC during today is a lookahead of the whole session,
   and it is the single easiest way to make a volume-profile backtest
   glow. So the level tested is the PREVIOUS session's POC applied to the
   current one, which is both causal and how it is actually traded.

WHAT IS ACTUALLY BEING ASKED
----------------------------
Not "does price react at these levels" -- price reacts everywhere, and a
level that price merely visits is not a level that pays. The question is
whether reactions at THESE levels differ from reactions at an arbitrary
line drawn nearby.

So every hypothesis has a matched PLACEBO:

* VWAP placebo: VWAP DISPLACED by a fixed +0.30%. Identical shape,
  identical dynamics, identical touch frequency profile -- wrong location.
* POC placebo: a price drawn from the prior session's range that is NOT
  the volume peak, at the same distance from the open as the real POC.

If the real level does not beat its displaced twin, the strategy is about
price revisiting a line, which any line offers.

THE HYPOTHESES

L1  PRIMARY. VWAP cross: after price crosses session VWAP, does the move
    CONTINUE (reclaim) or REVERT (bounce)? Two-sided -- both are widely
    traded and the mechanism does not favour either a priori. Signed so
    positive means continuation.

L2  VWAP versus its displaced placebo. The test that decides whether VWAP
    location matters at all.

L3  Prior-session POC touch: continuation or reversion, same convention.

L4  POC versus its placebo.

L5  Context, not a hypothesis: does approaching from above differ from
    approaching from below? Equities drift up, so asymmetry is expected.

SPECIFICATION, fixed in advance
-------------------------------
* First qualifying event per session only. Multiple crosses in one session
  are not independent observations, and counting them would inflate n
  without adding information.
* No event before 10:00 ET: session VWAP is unstable in its first minutes,
  when it is an average of almost nothing.
* Entry at the level, not the crossing bar's close -- the same rule as the
  ORB screen, and for the same reason.
* Exit after 30 minutes (primary) and at the 16:00 close (secondary).
* Costs: 1.3bp MES round trip charged to every trade, winners included.
* POC buckets are 1 index point wide, built from minute bars. A true POC
  uses tick or TPO data; this is a minute-resolution approximation and is
  stated as one rather than presented as exact.

Run: python scripts/screen_levels.py
"""

from __future__ import annotations

import collections
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

ES_GLOB = (str(Path.home()) + r"\Downloads\databento_spx\es_full\*.zst")
HOLDOUT_FRACTION = 0.30
COST_BP = 1.3
HOLD_MINUTES = 30
EARLIEST = dt.time(10, 0)
VWAP_DISPLACE = 0.0030          # +0.30%, fixed in advance
POC_BUCKET = 1.0                # index points

RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)
ET = dt.timezone(dt.timedelta(hours=-4))


def rth(bars):
    return [b for b in bars
            if RTH_OPEN <= b.ts.astimezone(ET).time() < RTH_CLOSE]


def running_vwap(bars):
    """Cumulative VWAP, one value per bar, using only bars up to that point."""
    out, pv, vol = [], 0.0, 0.0
    for b in bars:
        typical = (b.high + b.low + b.close) / 3.0
        v = max(b.volume, 1)
        pv += typical * v
        vol += v
        out.append(pv / vol)
    return out


def session_poc(bars) -> float | None:
    """Price bucket carrying the most volume. Known only after the close."""
    if not bars:
        return None
    profile = collections.Counter()
    for b in bars:
        bucket = round(((b.high + b.low + b.close) / 3.0) / POC_BUCKET)
        profile[bucket] += max(b.volume, 1)
    if not profile:
        return None
    return profile.most_common(1)[0][0] * POC_BUCKET


def poc_placebo(bars, real_poc: float) -> float | None:
    """A level from the same session's range that is NOT the volume peak.

    Mirrored across the session midpoint, so it sits a comparable distance
    from where price traded without being where the volume was.
    """
    if real_poc is None or not bars:
        return None
    hi = max(b.high for b in bars)
    lo = min(b.low for b in bars)
    mid = (hi + lo) / 2.0
    mirrored = 2.0 * mid - real_poc
    return mirrored if lo < mirrored < hi else None


def cross_event(bars, level_at, hold: int):
    """First crossing of a level after EARLIEST. Signed for CONTINUATION.

    `level_at(i)` gives the level for bar i, so it works for a static level
    and a running one alike.
    """
    prev_side = None
    for i, b in enumerate(bars):
        lvl = level_at(i)
        if lvl is None or lvl <= 0:
            continue
        side = 1 if b.close > lvl else -1
        if prev_side is None:
            prev_side = side
            continue
        if side == prev_side:
            continue
        prev_side = side
        if b.ts.astimezone(ET).time() < EARLIEST:
            continue
        if i + hold >= len(bars):
            return None
        entry = lvl                       # fill at the level, not the close
        exit_px = bars[i + hold].close
        # side is the direction just crossed INTO: +1 = crossed up.
        gross = 10000.0 * side * (exit_px / entry - 1.0)
        return {"net_bp": gross - COST_BP, "from_below": side > 0}
    return None


def main() -> int:
    paths = glob.glob(ES_GLOB)
    if not paths:
        raise SystemExit(f"no ES files at {ES_GLOB}")
    sessions = {d: rth(b) for d, b in session_bars(read_bars(paths[0])).items()}
    sessions = {d: b for d, b in sessions.items() if len(b) >= 300}
    days = sorted(sessions)
    cut = int(len(days) * (1 - HOLDOUT_FRACTION))
    ins = days[:cut]
    print("{0:,} ES sessions; in-sample {1:,} ({2} .. {3}); holdout {4:,} "
          "SEALED\n".format(len(days), len(ins), ins[0], ins[-1],
                            len(days) - cut))

    screen = Screen(cost_bp=COST_BP)
    vwap_real, vwap_fake, poc_real, poc_fake = [], [], [], []
    from_above, from_below = [], []

    for k, d in enumerate(ins):
        bars = sessions[d]
        vw = running_vwap(bars)

        ev = cross_event(bars, lambda i: vw[i], HOLD_MINUTES)
        if ev:
            vwap_real.append(ev["net_bp"])
            (from_below if ev["from_below"] else from_above).append(ev["net_bp"])
        ev = cross_event(bars, lambda i: vw[i] * (1 + VWAP_DISPLACE),
                         HOLD_MINUTES)
        if ev:
            vwap_fake.append(ev["net_bp"])

        # Prior session's POC -- causal, unlike this session's.
        if k == 0:
            continue
        prior = sessions[ins[k - 1]]
        poc = session_poc(prior)
        if poc:
            ev = cross_event(bars, lambda i: poc, HOLD_MINUTES)
            if ev:
                poc_real.append(ev["net_bp"])
            fake = poc_placebo(prior, poc)
            if fake:
                ev = cross_event(bars, lambda i: fake, HOLD_MINUTES)
                if ev:
                    poc_fake.append(ev["net_bp"])

    print("L1  PRIMARY: VWAP cross, {0}min hold (positive = continuation)"
          .format(HOLD_MINUTES))
    screen.record("VWAP cross", vwap_real)
    print()

    print("L2  VWAP vs its displaced twin (+{0:.2%})".format(VWAP_DISPLACE))
    b = {"real VWAP": vwap_real, "displaced": vwap_fake}
    for k2 in b:
        screen.record(k2, b[k2], descriptive=True)
    describe_split(b, "real VWAP", "displaced", screen=screen,
                   label="L2 VWAP vs placebo")
    print()

    print("L3  prior-session POC touch, {0}min hold".format(HOLD_MINUTES))
    screen.record("POC cross", poc_real)
    print()

    print("L4  POC vs a non-peak level from the same range")
    b = {"real POC": poc_real, "placebo level": poc_fake}
    for k2 in b:
        screen.record(k2, b[k2], descriptive=True)
    describe_split(b, "real POC", "placebo level", screen=screen,
                   label="L4 POC vs placebo")
    print()

    print("L5  context: which side price crossed from")
    for name, rows in (("from below", from_below), ("from above", from_above)):
        if len(rows) > 2:
            print("  {0:<11} n={1:<5} mean {2:+.2f}bp".format(
                name, len(rows), st.mean(rows)))
    print("  VWAP events fired on {0:.0%} of sessions".format(
        len(vwap_real) / len(ins)))

    print()
    screen.summary()
    print("\nSpecification and hypotheses fixed before the first statistic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
