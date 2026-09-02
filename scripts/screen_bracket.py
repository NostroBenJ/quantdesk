"""
screen_bracket.py -- simulate the ACTUAL trade, not the forward return.

PRE-REGISTERED. Written before any statistic below was computed.

THE OBJECTION THIS EXISTS TO TEST
---------------------------------
Every screen so far measured signed FORWARD RETURN over a fixed horizon
and found nothing. But the bot does not hold for a fixed horizon -- it
exits at whichever of stop or target is touched first. Those are different
measurements, and a signal can have exactly zero mean drift while still
paying a 2:1 bracket, if the PATH is asymmetric: if price tends to reach
+2R before it reaches -1R more often than 1-in-3, the bracket wins on a
signal with no drift at all.

Forward return measures drift. A bracket measures path. Concluding "no
drift, therefore no edge" skips a step, and this screen takes it.

HOW THE TRADE IS SIMULATED
--------------------------
Entry, stop and target are computed EXACTLY as strategy_ema_pullback does:

    sl = min(prev_low, last_low) - (close - ema) * 0.5      [buy]
    tp = entry + 2.0 * |entry - sl|

then bars are walked forward and the first level touched wins, using each
bar's HIGH and LOW rather than its close -- a close-only test would miss
every intrabar touch and systematically overstate the hold time.

THREE PLACES THIS COULD FLATTER THE STRATEGY, ALL CLOSED
--------------------------------------------------------
1. BOTH LEVELS IN ONE BAR. 5-minute bars are coarse enough that a single
   bar often spans both. Without tick data the order is unknowable, and
   the stop is the NEARER level (1R vs 2R), so it is more likely to have
   printed first. Booked as a LOSS. This is the same tie-break the patched
   bot_eval.py now uses, and the old code booked it as a win.
2. STOP SLIPPAGE. A stop triggers to market and fills at or past the
   level. The one real Lucid fill slipped 0.81 points. Charged to every
   loss. Targets are resting limits and fill at the limit, so they are
   charged nothing beyond commission.
3. COMMISSION AND SPREAD. 1.3bp round trip on every trade regardless of
   outcome.

THE CONTROL IS THE WHOLE POINT
------------------------------
A bracket on ANY entry produces a win rate near 33%, because that is what
the geometry does. So the pullback's win rate alone proves nothing. The
comparison that matters is against RANDOM bars in the same kill zones,
signed by EMA slope, with the bracket built by the SAME formula from their
own bars. If the pullback's path is special, it beats that. If it does
not, the pattern is decoration on a coin flip.

THE HYPOTHESES

B1  PRIMARY. Pullback bracket expectancy in R, net of costs. Two-sided.
B2  Win rate against the break-even the bracket actually imposes once
    slippage is charged (~35.8% at the median stop, not 33.3%).
B3  DECISIVE CONTROL. Pullback expectancy minus random-bar expectancy.
B4  Does the ADX filter change the bracket outcome? ADX>=25 vs ADX<25.

ASSUMPTION (microstructure): 0.81 points of stop slippage, measured on a
single live fill. The DIRECTION is certain from the order types; the
magnitude is one observation. Results are reported at 0.0, 0.81 and 1.62
points so the conclusion can be read against that uncertainty rather than
resting on it.

Run: python scripts/screen_bracket.py
"""

from __future__ import annotations

import math
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.sources.lse_data import load  # noqa: E402
from scripts.screen_gold_5m import (  # noqa: E402
    ADX_MIN, ADX_PERIOD, EMA_PERIOD, adx, ema, in_kill_zone,
)

HOLDOUT_FRACTION = 0.30
RISK_REWARD = 2.0
MAX_HOLD_BARS = 288          # 24 hours of 5-minute bars
COST_BP = 1.3
SLIPPAGE_POINTS = (0.0, 0.81, 1.62)


def bracket(direction, prev, last, ema_now):
    """Entry, stop and target exactly as strategy_ema_pullback builds them."""
    entry = last["close"]
    if direction == 1:
        sl = min(prev["low"], last["low"]) - (last["close"] - ema_now) * 0.5
    else:
        sl = max(prev["high"], last["high"]) + (ema_now - last["close"]) * 0.5
    dist = abs(entry - sl)
    if dist <= 0:
        return None
    tp = entry + dist * RISK_REWARD if direction == 1 else entry - dist * RISK_REWARD
    return entry, sl, tp, dist


def resolve(bars, i, direction, entry, sl, tp, slip):
    """Walk forward; first level touched wins. Returns R multiple or None
    if the trade never resolved inside MAX_HOLD_BARS."""
    dist = abs(entry - sl)
    for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, len(bars))):
        b = bars[j]
        if direction == 1:
            hit_sl, hit_tp = b["low"] <= sl, b["high"] >= tp
        else:
            hit_sl, hit_tp = b["high"] >= sl, b["low"] <= tp
        if hit_sl:
            # Nearer level; without tick data it is the honest tie-break.
            return -(dist + slip) / dist
        if hit_tp:
            return RISK_REWARD
    return None


def run(bars, events, slip):
    out = []
    for e in events:
        i = e["i"]
        br = bracket(e["direction"], bars[i - 1], bars[i], e["ema"])
        if br is None:
            continue
        entry, sl, tp, dist = br
        r = resolve(bars, i, e["direction"], entry, sl, tp, slip)
        if r is None:
            continue
        # Costs in R: a fixed bp cost is a larger fraction of a tight stop.
        cost_r = (entry * COST_BP / 10000.0) / dist
        out.append(r - cost_r)
    return out


def stats(rs):
    n = len(rs)
    m = st.mean(rs)
    se = st.stdev(rs) / math.sqrt(n) if n > 1 else float("nan")
    wins = sum(1 for r in rs if r > 0)
    return {"n": n, "mean": m, "se": se, "t": m / se if se else float("nan"),
            "win": wins / n if n else float("nan")}


def main() -> int:
    bars = load("GC.F", "5m")
    if not bars:
        raise SystemExit("no cached data -- fetch GC.F 5m first")
    cut = int(len(bars) * (1 - HOLDOUT_FRACTION))
    ins = bars[:cut]

    closes = [b["close"] for b in ins]
    e = ema(closes, EMA_PERIOD)
    a = adx(ins, ADX_PERIOD)

    pulls, randoms = [], []
    warm = EMA_PERIOD + ADX_PERIOD + 5
    for i in range(warm, len(ins) - 1):
        if a[i] is None or not in_kill_zone(ins[i]["ts"]):
            continue
        up, dn = e[i] > e[i - 1], e[i] < e[i - 1]
        if not (up or dn):
            continue
        rec = {"i": i, "adx": a[i], "ema": e[i]}
        if up and ins[i - 1]["low"] <= e[i - 1] and ins[i]["close"] > e[i]:
            pulls.append({**rec, "direction": 1})
        elif dn and ins[i - 1]["high"] >= e[i - 1] and ins[i]["close"] < e[i]:
            pulls.append({**rec, "direction": -1})
        else:
            randoms.append({**rec, "direction": 1 if up else -1})

    random.seed(0)
    sample = random.sample(randoms, min(len(pulls), len(randoms)))

    print("in-sample {0:,} bars ({1} .. {2}); holdout SEALED".format(
        len(ins), ins[0]["ts"].date(), ins[-1]["ts"].date()))
    print("{0:,} kill-zone pullbacks, {1:,} matched random controls\n".format(
        len(pulls), len(sample)))
    print("Simulating the REAL trade: first touch of stop or target, using")
    print("intrabar highs and lows. Both-in-one-bar books the LOSS.\n")

    for slip in SLIPPAGE_POINTS:
        p = stats(run(ins, pulls, slip))
        c = stats(run(ins, sample, slip))
        # Break-even win rate this bracket actually imposes at this slip.
        print("=" * 70)
        print("STOP SLIPPAGE {0:.2f} points{1}".format(
            slip, "   <-- measured on the live fill" if slip == 0.81 else ""))
        print("=" * 70)
        print("B1/B2 pullback   n={0:,}  expectancy {1:+.4f}R  t={2:+.2f}  "
              "win rate {3:.1%}".format(p["n"], p["mean"], p["t"], p["win"]))
        print("B3    random     n={0:,}  expectancy {1:+.4f}R  t={2:+.2f}  "
              "win rate {3:.1%}".format(c["n"], c["mean"], c["t"], c["win"]))
        diff = p["mean"] - c["mean"]
        sed = math.sqrt(p["se"] ** 2 + c["se"] ** 2)
        print("      -> pullback minus random: {0:+.4f}R  t={1:+.2f}".format(
            diff, diff / sed if sed else float("nan")))

        hi = stats(run(ins, [x for x in pulls if x["adx"] >= ADX_MIN], slip))
        lo = stats(run(ins, [x for x in pulls if x["adx"] < ADX_MIN], slip))
        d2 = hi["mean"] - lo["mean"]
        se2 = math.sqrt(hi["se"] ** 2 + lo["se"] ** 2)
        print("B4    ADX>=25 {0:+.4f}R (n={1:,})  ADX<25 {2:+.4f}R (n={3:,})"
              .format(hi["mean"], hi["n"], lo["mean"], lo["n"]))
        print("      -> ADX adds: {0:+.4f}R  t={1:+.2f}".format(
            d2, d2 / se2 if se2 else float("nan")))
        print()

    print("An expectancy of 0.0000R is break-even. The 2:1 geometry gives")
    print("~33% wins to ANY entry, so the pullback's win rate alone means")
    print("nothing -- B3, against random bars in the same windows, is the")
    print("test that can actually fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
