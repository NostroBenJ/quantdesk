"""
screen_gold.py -- the Goldbot ablation, run on GOLD.

PRE-REGISTERED. Written before any statistic below was computed.

WHY THIS EXISTS
---------------
`screen_ablation.py` tested the EMA-pullback and ADX rules on sixteen years
of ES and found nothing, under an explicit assumption that the rules are
generic and the instrument should not matter. That assumption is exactly
what the bot's author disputes: he says it works on gold, and the ADX
filter in particular. This screen removes the assumption.

WHAT IS AND IS NOT FIXED HERE
-----------------------------
FIXED: the instrument. This is GC, the thing actually traded, 13,740 bars
from 2024-04-07 to 2026-08-28.

NOT FIXED: the timeframe. The live bot reads 5-minute candles
(`bot.py:256`, unit=2 unit_number=5) and the only gold history on hand is
hourly. An hourly bar is twelve of the bot's bars, so a pullback here is a
coarser event than the one the bot trades. This CANNOT be waved away, so
it is stated as the headline caveat: a null here is weaker evidence than a
null on ES was, and a positive here would need 5-minute confirmation
before it meant anything.

HOLDING PERIOD, matched to the live record rather than guessed. The 18
live trades resolved in a median of about 30 minutes, with a long tail
(one ran 4.4 hours, one overnight). At hourly resolution the closest
honest brackets are 1 and 2 bars, and BOTH are reported -- picking the
better-looking one afterwards is the error this project exists to avoid.

THE HYPOTHESES -- the same four as the ES run, so the two are comparable

G1  PRIMARY. EMA pullback ALONE, no ADX filter. Signed forward return.
G2  Does the ADX filter ADD anything? Pullbacks with ADX >= 25 vs < 25.
    This is the author's specific claim and the reason for the screen.
G3  ADX ALONE. High vs low ADX signed by EMA slope, no pattern.
G4  CONTROL. Random bars matched on count and direction.

Corrected across both horizons as one family: eight tests, not four. The
horizon was not chosen after seeing the answer, so it costs correction.

COST: MGC round trip. ASSUMPTION (microstructure): 1.3bp, the same figure
the ES screen charged, which is roughly a tick of spread plus commission
on a ~$4,400 contract. The live fills show 0.81 points of slip past the
stop, which is larger -- but that is a stop-fill cost, not a round-trip
cost, and applying it to every observation would overcharge.

Run: python scripts/screen_gold.py
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.research_stats import Screen, describe_split  # noqa: E402

GC_CSV = Path(r"C:\Users\jontr\dev\shadow-trade\gc_1h.csv")
HOLDOUT_FRACTION = 0.30
COST_BP = 1.3

EMA_PERIOD = 21
ADX_PERIOD = 14
ADX_MIN = 25.0
HORIZONS = (1, 2)          # hourly bars; the live median hold was ~30 min


def load_gc(path: Path) -> list[dict]:
    """Hourly GC bars. Rows with a zero or missing OHLC are dropped rather
    than carried forward -- a synthesised bar would create a pullback that
    never happened."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                o, h, l, c = (float(r["Open"]), float(r["High"]),
                              float(r["Low"]), float(r["Close"]))
            except (ValueError, KeyError, TypeError):
                continue
            if min(o, h, l, c) <= 0:
                continue
            out.append({"ts": r["ts"], "open": o, "high": h,
                        "low": l, "close": c})
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
    NOT Wilder's smoothing -- the bot runs this version."""
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
    """Every EMA-pullback event, replicating strategy_ema_pullback exactly:
    uptrend needs a rising EMA, the PREVIOUS bar's low at or below its EMA,
    and the current close back above the EMA."""
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
        out.append({"i": i, "adx": a[i], "direction": direction,
                    "trend": 1 if up else (-1 if dn else 0), "fwd_bp": fwd})
    return out


def main() -> int:
    if not GC_CSV.exists():
        raise SystemExit("no gold data at {0}".format(GC_CSV))
    bars = load_gc(GC_CSV)
    cut = int(len(bars) * (1 - HOLDOUT_FRACTION))
    ins = bars[:cut]

    print("GOLD (GC), hourly. {0:,} bars {1} .. {2}".format(
        len(bars), bars[0]["ts"][:10], bars[-1]["ts"][:10]))
    print("in-sample {0:,}; holdout {1:,} SEALED".format(cut, len(bars) - cut))
    print("CAVEAT: the bot trades 5-MINUTE bars. This is hourly -- the right")
    print("instrument at the wrong resolution. Stated up front, not buried.\n")

    screen = Screen(cost_bp=COST_BP)

    def signed(e):
        return e["direction"] * e["fwd_bp"] - COST_BP

    for H in HORIZONS:
        events = signals(ins, H)
        pullbacks = [e for e in events if e["direction"] != 0]
        trend_only = [e for e in events if e["trend"] != 0]
        tag = "{0}h".format(H)
        print("=" * 68)
        print("FORWARD {0} BAR{1}  ({2:,} pullback events)".format(
            H, "" if H == 1 else "S", len(pullbacks)))
        print("=" * 68)

        print("G1  PRIMARY: EMA pullback alone, no ADX filter")
        screen.record("G1 pullback, " + tag, [signed(e) for e in pullbacks])

        print("\nG2  does the ADX filter add anything?  <-- the author's claim")
        b2 = {"ADX >= 25": [signed(e) for e in pullbacks
                            if e["adx"] >= ADX_MIN],
              "ADX < 25": [signed(e) for e in pullbacks if e["adx"] < ADX_MIN]}
        for k in b2:
            screen.record(k, b2[k], descriptive=True)
        describe_split(b2, "ADX >= 25", "ADX < 25", screen=screen,
                       label="G2 ADX filter adds, " + tag)

        print("\nG3  ADX alone, signed by EMA slope, no pattern")
        b3 = {"ADX >= 25": [e["trend"] * e["fwd_bp"] - COST_BP
                            for e in trend_only if e["adx"] >= ADX_MIN],
              "ADX < 25": [e["trend"] * e["fwd_bp"] - COST_BP
                           for e in trend_only if e["adx"] < ADX_MIN]}
        for k in b3:
            screen.record(k, b3[k], descriptive=True)
        describe_split(b3, "ADX >= 25", "ADX < 25", screen=screen,
                       label="G3 ADX alone, " + tag)

        print("\nG4  CONTROL: random bars signed by EMA slope")
        random.seed(0)
        pool = [e for e in trend_only if e["direction"] == 0]
        sample = random.sample(pool, min(len(pullbacks), len(pool)))
        b4 = {"pullback": [signed(e) for e in pullbacks],
              "random bar": [e["trend"] * e["fwd_bp"] - COST_BP
                             for e in sample]}
        for k in b4:
            screen.record(k, b4[k], descriptive=True)
        describe_split(b4, "pullback", "random bar", screen=screen,
                       label="G4 pattern vs random, " + tag)

        hits = sum(1 for e in pullbacks if e["direction"] * e["fwd_bp"] > 0)
        hi = sum(1 for e in pullbacks if e["adx"] >= ADX_MIN)
        print("\ncontext: {0:.1%} resolved in the signal's direction; "
              "{1:.1%} had ADX >= 25".format(hits / len(pullbacks),
                                             hi / len(pullbacks)))
        print()

    screen.summary()
    print("\nSame four hypotheses as the ES run, on the actual instrument.")
    print("Both horizons pre-registered, so neither was chosen after the fact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
