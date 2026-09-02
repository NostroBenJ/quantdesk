"""
screen_gold_5m.py -- Goldbot's strategy at the exact configuration it trades.

PRE-REGISTERED. Written before any statistic below was computed.

WHY THIS IS THE ONE THAT COUNTS
-------------------------------
Two screens have already failed to answer the author's claim cleanly, each
for a stated reason:

    screen_ablation.py   16 years, 5-minute, but ES -- wrong instrument
    screen_gold.py       gold, but hourly       -- wrong resolution

Every null so far has carried a caveat the author could fairly point at.
This one carries neither. `bot_eval.py` reads 5-minute GC candles and
enters only inside four ET kill zones; this screen reads 5-minute GC
candles from 2016-05-01 and applies the same four windows. 723,721 bars,
free from London Strategic Edge, verified by `lse_data.verify_bars()`
before use rather than trusted.

If the pullback carries information anywhere, it is here.

WHAT IS HELD FIXED TO THE LIVE BOT
----------------------------------
* EMA(21) and the pullback-and-reclaim test, replicating
  strategy_ema_pullback exactly (previous bar's low at or below its EMA,
  current close back above it, EMA rising).
* ADX(14) with SIMPLE ROLLING MEANS, matching strategy_vwap.calculate_adx.
  NOT Wilder's smoothing -- the textbook version would test a strategy
  nobody is running.
* Entries only inside KILL_ZONES from config.py, in EASTERN time with real
  DST handling via zoneinfo. A fixed UTC-4 offset (as an earlier screen
  used) silently shifts every winter bar by an hour, which moves entries
  into and out of the windows being tested.

HORIZONS, both pre-registered so neither is chosen after the fact:
    6 bars  = 30 minutes, the MEDIAN hold of the 18 live trades
    12 bars = 1 hour, matching the ES ablation so the two are comparable

THE HYPOTHESES -- five, at two horizons, corrected as one family of ten

P1  PRIMARY. EMA pullback alone, inside kill zones, no ADX filter.
P2  Does the ADX filter ADD anything? ADX >= 25 against ADX < 25. This is
    the author's specific claim: he says the ADX is the part that works.
P3  ADX ALONE, signed by EMA slope, no pattern.
P4  CONTROL. Random bars in the same windows, signed by EMA slope,
    matched on count. If a random bar pays the same, the pattern is not
    what is being measured.
P5  Do the KILL ZONES themselves help? Pullbacks inside the windows
    against pullbacks outside them. The windows were chosen by someone;
    this asks whether the choice survives out of sample.

COST: MGC round trip. ASSUMPTION (microstructure): 1.3bp, roughly a tick
of spread plus commission on a ~$4,400 contract. Charged to every
observation. The live fills show 0.81 POINTS of slip past the stop, but
that is a stop-fill cost rather than a round-trip cost and applying it to
every bar would overcharge.

Measured on FORWARD RETURN signed by the signal's direction, never on
bracket outcome. A 2:1 bracket manufactures a ~33% win rate from a coin
toss, so a win rate proves nothing about the entry; a signed forward
return cannot be rescued by geometry.

Run: python scripts/screen_gold_5m.py
"""

from __future__ import annotations

import datetime as dt
import random
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.research_stats import Screen, describe_split  # noqa: E402
from quantdesk.data.sources.lse_data import load, verify_bars  # noqa: E402

SYMBOL, RESOLUTION = "GC.F", "5m"
HOLDOUT_FRACTION = 0.30
COST_BP = 1.3

EMA_PERIOD = 21
ADX_PERIOD = 14
ADX_MIN = 25.0
HORIZONS = (6, 12)

ET = ZoneInfo("America/New_York")

#: KILL_ZONES from shadow-trade/config.py, Eastern time. London Open was
#: removed by the author on 2026-07-30 after 3 trades and 0 wins, so it is
#: absent here too -- this tests the bot as it runs today, not as it ran.
KILL_ZONES = (
    ((8, 30), (9, 0)),      # Macro Window
    ((9, 30), (10, 0)),     # NY Open
    ((10, 0), (11, 0)),     # NY AM
    ((14, 0), (15, 0)),     # NY PM
)


def in_kill_zone(ts_utc: dt.datetime) -> bool:
    local = ts_utc.astimezone(ET)
    if local.weekday() >= 5:
        return False
    mins = local.hour * 60 + local.minute
    return any(a[0] * 60 + a[1] <= mins < b[0] * 60 + b[1]
               for a, b in KILL_ZONES)


def ema(values, period: int):
    k = 2.0 / (period + 1.0)
    out, cur = [], values[0]
    for v in values:
        cur = v * k + cur * (1 - k)
        out.append(cur)
    return out


def adx(bars, period: int):
    """Simple-rolling-mean ADX, matching strategy_vwap.calculate_adx."""
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


def signals(bars, forward: int):
    closes = [b["close"] for b in bars]
    e = ema(closes, EMA_PERIOD)
    a = adx(bars, ADX_PERIOD)
    out = []
    warm = EMA_PERIOD + ADX_PERIOD + 5
    for i in range(warm, len(bars) - forward):
        if a[i] is None:
            continue
        up, dn = e[i] > e[i - 1], e[i] < e[i - 1]
        direction = 0
        if up and bars[i - 1]["low"] <= e[i - 1] and bars[i]["close"] > e[i]:
            direction = 1
        elif dn and bars[i - 1]["high"] >= e[i - 1] and bars[i]["close"] < e[i]:
            direction = -1
        fwd = 10000.0 * (bars[i + forward]["close"] / bars[i]["close"] - 1)
        out.append({"adx": a[i], "direction": direction,
                    "trend": 1 if up else (-1 if dn else 0),
                    "fwd_bp": fwd, "kz": in_kill_zone(bars[i]["ts"])})
    return out


def main() -> int:
    bars = load(SYMBOL, RESOLUTION)
    if not bars:
        raise SystemExit(
            "no cached data -- run:\n"
            "  python -m quantdesk.data.sources.lse_data --fetch GC.F 5m")

    print("verifying the tape before trusting it")
    if not verify_bars(bars, RESOLUTION):
        print("\nTape failed verification -- stopping. Every number below "
              "would inherit the defect.")
        return 1

    cut = int(len(bars) * (1 - HOLDOUT_FRACTION))
    ins = bars[:cut]
    print("\nin-sample {0:,} bars ({1} .. {2}); holdout {3:,} SEALED".format(
        len(ins), ins[0]["ts"].date(), ins[-1]["ts"].date(), len(bars) - cut))
    print("GOLD, 5-MINUTE, KILL ZONES -- the configuration the bot runs.")
    print("No instrument caveat, no resolution caveat.\n")

    screen = Screen(cost_bp=COST_BP)

    def signed(e):
        return e["direction"] * e["fwd_bp"] - COST_BP

    for H in HORIZONS:
        events = signals(ins, H)
        kz = [e for e in events if e["kz"]]
        pulls = [e for e in kz if e["direction"] != 0]
        trend = [e for e in kz if e["trend"] != 0]
        tag = "{0}b".format(H)
        primary = H == HORIZONS[0]

        print("=" * 70)
        print("FORWARD {0} BARS = {1} MINUTES{2}".format(
            H, H * 5, "   <-- PRIMARY (median live hold)" if primary else ""))
        print("{0:,} in-zone pullbacks of {1:,} in-zone bars".format(
            len(pulls), len(kz)))
        print("=" * 70)

        print("P1  PRIMARY: pullback alone, in kill zones, no ADX filter")
        screen.record("P1 pullback, " + tag, [signed(e) for e in pulls])

        print("\nP2  does the ADX filter add anything?  <-- the author's claim")
        b2 = {"ADX >= 25": [signed(e) for e in pulls if e["adx"] >= ADX_MIN],
              "ADX < 25": [signed(e) for e in pulls if e["adx"] < ADX_MIN]}
        for k in b2:
            screen.record(k, b2[k], descriptive=True)
        describe_split(b2, "ADX >= 25", "ADX < 25", screen=screen,
                       label="P2 ADX filter adds, " + tag)

        print("\nP3  ADX alone, signed by EMA slope, no pattern")
        b3 = {"ADX >= 25": [e["trend"] * e["fwd_bp"] - COST_BP
                            for e in trend if e["adx"] >= ADX_MIN],
              "ADX < 25": [e["trend"] * e["fwd_bp"] - COST_BP
                           for e in trend if e["adx"] < ADX_MIN]}
        for k in b3:
            screen.record(k, b3[k], descriptive=True)
        describe_split(b3, "ADX >= 25", "ADX < 25", screen=screen,
                       label="P3 ADX alone, " + tag)

        print("\nP4  CONTROL: random in-zone bars signed by EMA slope")
        random.seed(0)
        pool = [e for e in trend if e["direction"] == 0]
        sample = random.sample(pool, min(len(pulls), len(pool)))
        b4 = {"pullback": [signed(e) for e in pulls],
              "random bar": [e["trend"] * e["fwd_bp"] - COST_BP
                             for e in sample]}
        for k in b4:
            screen.record(k, b4[k], descriptive=True)
        describe_split(b4, "pullback", "random bar", screen=screen,
                       label="P4 pattern vs random, " + tag)

        print("\nP5  do the KILL ZONES help? in-zone vs out-of-zone pullbacks")
        out_pulls = [e for e in events if not e["kz"] and e["direction"] != 0]
        b5 = {"in zone": [signed(e) for e in pulls],
              "out of zone": [signed(e) for e in out_pulls]}
        for k in b5:
            screen.record(k, b5[k], descriptive=True)
        describe_split(b5, "in zone", "out of zone", screen=screen,
                       label="P5 kill zones help, " + tag)

        hits = sum(1 for e in pulls if e["direction"] * e["fwd_bp"] > 0)
        hi = sum(1 for e in pulls if e["adx"] >= ADX_MIN)
        print("\ncontext: {0:.1%} resolved in the signal's direction; "
              "{1:.1%} had ADX >= 25".format(hits / len(pulls),
                                             hi / len(pulls)))
        print()

    screen.summary()
    print("\nRight instrument, right resolution, right session windows.")
    print("Measured on forward return, so a 2:1 bracket cannot supply a win")
    print("rate the signal did not earn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
