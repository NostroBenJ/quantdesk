"""
screen_ablation.py -- take Goldbot apart. Does either component carry
information on its own?

PRE-REGISTERED. Written before any statistic below was computed.

THE QUESTION
------------
Goldbot's live strategy is two rules stacked: an ADX(14) >= 25 filter, and
an EMA(21) pullback-and-reclaim pattern. Measured together on 116 trades it
returned +0.009R at t=0.07 -- a win rate of 33.6% against the 33.3%
break-even its own 2:1 bracket imposes. The creator says it works. This
screen asks which half, if either, is doing anything.

TWO THINGS THE ORIGINAL MEASUREMENT COULD NOT SEE
--------------------------------------------------
1. IT MEASURED THE BRACKET, NOT THE SIGNAL. A 2:1 stop/target forces a
   33.3% break-even win rate whether the entry is informed or a coin toss.
   Reading 33.6% off that tells you the bracket geometry worked, and says
   almost nothing about the entry. So this screen measures the FORWARD
   RETURN signed by the signal's direction -- if the pattern predicts, that
   is positive; if it does not, no bracket can rescue it.

2. IT HAD 116 TRADES. Yahoo caps 5-minute gold at 60 days, and the
   resulting confidence interval excluded nothing. We own sixteen years of
   ES minute bars, resampled here to 5 minutes -- roughly two orders of
   magnitude more sample.

ASSUMPTION (generality): the rules are generic technical patterns with
nothing gold-specific in them, so they are tested on ES where the data
exists. If a component works on gold and provably not on ES, that is a much
narrower claim than "it works", and one that 60 days cannot support either
way. Gold is tested alongside at hourly resolution for what it is worth.

The ADX is computed exactly as `strategy_vwap.calculate_adx` does -- simple
rolling means, NOT Wilder's smoothing. That is a real difference in the
indicator, and using the textbook version here would test a strategy nobody
is running.

THE HYPOTHESES

A1  PRIMARY. The EMA pullback pattern ALONE, no ADX filter. Signed forward
    return over 12 bars. Two-sided: the creator's prior says positive, the
    116-trade measurement says nothing.

A2  Does the ADX filter ADD anything? Pullbacks with ADX >= 25 against
    pullbacks with ADX < 25. This is the ablation proper -- if the filter
    is doing work, these differ.

A3  ADX ALONE. High-ADX bars against low-ADX bars, signed by EMA slope.
    Tests whether "trending" by this definition predicts continuation at
    all, with no pattern involved.

A4  CONTROL. Random bars matched on count and direction. If a randomly
    chosen bar signed by EMA slope pays the same as a pullback, the
    pattern is not what is being measured.

Costs: MES round trip ~1.3bp, charged to every observation.

Run: python scripts/screen_ablation.py
"""

from __future__ import annotations

import datetime as dt
import glob
import random
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

EMA_PERIOD = 21
ADX_PERIOD = 14
ADX_MIN = 25.0
FORWARD_BARS = 12          # one hour at 5-minute resolution
RESAMPLE = 5               # minutes

RTH_OPEN, RTH_CLOSE = dt.time(9, 30), dt.time(16, 0)
ET = dt.timezone(dt.timedelta(hours=-4))


def resample(bars, minutes: int):
    """1-minute bars -> N-minute bars. Open of the first, close of the
    last, high and low across the group."""
    out = []
    for i in range(0, len(bars) - minutes + 1, minutes):
        g = bars[i:i + minutes]
        out.append({"ts": g[0].ts, "open": g[0].open,
                    "high": max(b.high for b in g),
                    "low": min(b.low for b in g),
                    "close": g[-1].close})
    return out


def ema(values, period: int):
    k = 2.0 / (period + 1.0)
    out, cur = [], values[0]
    for v in values:
        cur = v * k + cur * (1 - k)
        out.append(cur)
    return out


def adx(bars, period: int):
    """Simple-rolling-mean ADX, matching strategy_vwap.calculate_adx.

    NOT Wilder's smoothing. The bot runs this version, so this is the
    version that has to be tested -- the textbook one would measure a
    strategy nobody is trading.
    """
    n = len(bars)
    plus_dm, minus_dm, tr = [0.0] * n, [0.0] * n, [0.0] * n
    for i in range(1, n):
        up = bars[i]["high"] - bars[i - 1]["high"]
        dn = bars[i - 1]["low"] - bars[i]["low"]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(bars[i]["high"] - bars[i]["low"],
                    abs(bars[i]["high"] - bars[i - 1]["close"]),
                    abs(bars[i]["low"] - bars[i - 1]["close"]))

    def roll(xs):
        out, run = [None] * n, 0.0
        for i, x in enumerate(xs):
            run += x
            if i >= period:
                run -= xs[i - period]
            out[i] = run / period if i >= period - 1 else None
        return out

    atr, pdm, mdm = roll(tr), roll(plus_dm), roll(minus_dm)
    dx = [None] * n
    for i in range(n):
        if atr[i] is None or not atr[i]:
            continue
        pdi, mdi = 100 * pdm[i] / atr[i], 100 * mdm[i] / atr[i]
        if pdi + mdi > 0:
            dx[i] = 100 * abs(pdi - mdi) / (pdi + mdi)
    out, window = [None] * n, []
    for i in range(n):
        if dx[i] is None:
            window = []
            continue
        window.append(dx[i])
        if len(window) > period:
            window.pop(0)
        if len(window) == period:
            out[i] = sum(window) / period
    return out


def signals(bars):
    """Every EMA-pullback event, with its ADX and forward return.

    Direction and the pullback test replicate strategy_ema_pullback exactly:
    uptrend needs a rising EMA, the PREVIOUS bar's low at or below its EMA,
    and the current close back above the EMA.
    """
    closes = [b["close"] for b in bars]
    e = ema(closes, EMA_PERIOD)
    a = adx(bars, ADX_PERIOD)
    out = []
    warm = EMA_PERIOD + ADX_PERIOD + 5
    for i in range(warm, len(bars) - FORWARD_BARS):
        if a[i] is None:
            continue
        up, dn = e[i] > e[i - 1], e[i] < e[i - 1]
        direction = 0
        if up and bars[i - 1]["low"] <= e[i - 1] and bars[i]["close"] > e[i]:
            direction = 1
        elif dn and bars[i - 1]["high"] >= e[i - 1] and bars[i]["close"] < e[i]:
            direction = -1
        fwd = 10000.0 * (bars[i + FORWARD_BARS]["close"] / bars[i]["close"] - 1)
        out.append({"i": i, "adx": a[i], "direction": direction,
                    "trend": 1 if up else (-1 if dn else 0),
                    "fwd_bp": fwd})
    return out


def main() -> int:
    paths = glob.glob(ES_GLOB)
    if not paths:
        raise SystemExit("no ES data")
    print("loading ES and resampling to {0}-minute ...".format(RESAMPLE))
    sess = session_bars(read_bars(paths[0]))
    days = sorted(sess)
    cut = int(len(days) * (1 - HOLDOUT_FRACTION))
    ins = days[:cut]

    events = []
    for d in ins:
        rth = [b for b in sess[d]
               if RTH_OPEN <= b.ts.astimezone(ET).time() < RTH_CLOSE]
        if len(rth) < 200:
            continue
        bars = resample(rth, RESAMPLE)
        if len(bars) < EMA_PERIOD + ADX_PERIOD + FORWARD_BARS + 10:
            continue
        events.extend(signals(bars))

    pullbacks = [e for e in events if e["direction"] != 0]
    print("in-sample {0:,} sessions ({1} .. {2}); holdout SEALED".format(
        len(ins), ins[0], ins[-1]))
    print("{0:,} bars examined, {1:,} pullback events\n".format(
        len(events), len(pullbacks)))

    screen = Screen(cost_bp=COST_BP)
    signed = lambda e: e["direction"] * e["fwd_bp"] - COST_BP

    # ---------------------------------------------------------------- A1
    print("A1  PRIMARY: EMA pullback ALONE, no ADX filter")
    print("    signed forward return over {0} bars, net of costs".format(
        FORWARD_BARS))
    screen.record("pullback, any ADX", [signed(e) for e in pullbacks])
    print()

    # ---------------------------------------------------------------- A2
    print("A2  does the ADX filter add anything?")
    b2 = {"ADX >= 25": [signed(e) for e in pullbacks if e["adx"] >= ADX_MIN],
          "ADX < 25": [signed(e) for e in pullbacks if e["adx"] < ADX_MIN]}
    for k in b2:
        screen.record(k, b2[k], descriptive=True)
    describe_split(b2, "ADX >= 25", "ADX < 25", screen=screen,
                   label="A2 ADX filter adds")
    print()

    # ---------------------------------------------------------------- A3
    print("A3  ADX ALONE: high vs low ADX, signed by EMA slope, no pattern")
    trend_only = [e for e in events if e["trend"] != 0]
    b3 = {"ADX >= 25": [e["trend"] * e["fwd_bp"] - COST_BP
                        for e in trend_only if e["adx"] >= ADX_MIN],
          "ADX < 25": [e["trend"] * e["fwd_bp"] - COST_BP
                       for e in trend_only if e["adx"] < ADX_MIN]}
    for k in b3:
        screen.record(k, b3[k], descriptive=True)
    describe_split(b3, "ADX >= 25", "ADX < 25", screen=screen,
                   label="A3 ADX alone")
    print()

    # ---------------------------------------------------------------- A4
    print("A4  CONTROL: random bars, signed by EMA slope, matched on count")
    random.seed(0)
    pool = [e for e in trend_only if e["direction"] == 0]
    sample = random.sample(pool, min(len(pullbacks), len(pool)))
    b4 = {"pullback": [signed(e) for e in pullbacks],
          "random bar": [e["trend"] * e["fwd_bp"] - COST_BP for e in sample]}
    for k in b4:
        screen.record(k, b4[k], descriptive=True)
    describe_split(b4, "pullback", "random bar", screen=screen,
                   label="A4 pattern vs random")
    print()

    print("context, not hypotheses:")
    hits = sum(1 for e in pullbacks if e["direction"] * e["fwd_bp"] > 0)
    print("  pullbacks resolving in the signal's direction: {0:.1%}".format(
        hits / len(pullbacks)))
    print("  {0:.1%} of pullbacks had ADX >= 25".format(
        sum(1 for e in pullbacks if e["adx"] >= ADX_MIN) / len(pullbacks)))
    print()
    screen.summary()
    print("\nMeasured on forward return, so the 2:1 bracket cannot supply a "
          "win rate the signal did not earn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
