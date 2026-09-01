"""
screen_vrp_timing.py -- when is the option market WRONG about volatility?

PRE-REGISTERED. Written before any statistic below was computed. Holdout:
the last 30% of sessions is SEALED and this file never reads it.

WHY THIS IS THE STRATEGY QUESTION AND THE OTHERS WERE NOT
---------------------------------------------------------
Twelve screens looked for direction and found nothing. The one survivor was
a volatility-regime forecast, which was filed as a consolation prize
because a futures account cannot express it. In an options account it is
not a consolation prize -- it is the entire trade.

But "vol will be high" is not tradeable either, and this is the part that
matters. The variance risk premium means implied vol sits ABOVE realised on
average: buying options is a losing bet on average, and selling them is a
winning one. So a forecast of high volatility does not tell you to buy
options. It tells you volatility will be high, which the option market has
probably already priced.

The tradeable question is narrower:

    WHEN DOES REALISED VOLATILITY EXCEED IMPLIED?

That is the conditional state where the premium inverts -- where the market
underprices what is coming. Identify it and both sides become actionable:

    forecast VRP large and positive  ->  sell premium (Level 3 spreads)
    forecast VRP negative            ->  buy premium, or stand aside

At Level 2 only the second is available, which is why the sign of the
prediction matters more here than its magnitude.

THE MEASURE
-----------
    VRP(t) = VIX(t) - realised vol over the NEXT 21 sessions

Positive means the option market overcharged and selling won. Negative
means it undercharged and buying won. VIX(t) is known at t; the realised
leg is strictly forward. No lookahead in the signal, by construction.

OVERLAP: forward 21-session windows sampled daily share 20 of 21 days.
Adjacent observations are nearly the same fact. Every standard error here
is widened by sqrt(21) -- the correction that turned a t of +15.13 into
+1.91 on the earlier VRP screen, and the reason that screen did not report
a discovery.

THE HYPOTHESES

R1  POSITIVE CONTROL. Is VRP positive on average? It must be -- this is one
    of the most replicated facts in finance. If it is not, the measurement
    is broken and nothing below can be read.

R2  PRIMARY. Does the VIX term-structure slope predict VRP? Mechanism:
    backwardation (VIX above VIX3M) is the market paying up for IMMEDIATE
    protection, which happens in stress -- and stress is where realised
    most often overshoots what was priced. Prior: flatter or inverted slope
    means LOWER (more often negative) VRP.

R3  Does the slope add anything beyond the VIX LEVEL? High VIX and inverted
    slope co-occur, so if level explains it the slope is redundant. This is
    the same control that H5 was for gamma, and it decides whether R2 is a
    finding or a restatement.

R4  Gamma regime, SAMPLE-LIMITED to the 251 sessions with SPX positioning.
    Labelled limited in advance so a null reads as "could not resolve".

COST, stated before any number
------------------------------
For an options trade the cost is the spread crossed, not a futures tick.
A liquid SPY ~30-day option runs roughly 0.5 vol points wide. An edge
smaller than that is not a trade, and it is reported against that floor.

ASSUMPTION (microstructure): 0.5 vol points is a stated placeholder, not a
measured quantity. It should be replaced with the real spread from the
option chain before anything is sized on it.

Run: python scripts/screen_vrp_timing.py
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.research_stats import (  # noqa: E402
    Screen, describe_split, overlap_corrected, welch_overlap,
)

VOL_CSV = ROOT / "spy_vix_term.csv"
GEX_CSV = ROOT / "data" / "spx_gex_daily.csv"

HOLDOUT_FRACTION = 0.30
HORIZON = 21                 # sessions; matches VIX's 30-calendar-day window
COST_VOL_PTS = 0.5


def load_rows() -> list[dict]:
    with VOL_CSV.open(encoding="utf-8") as fh:
        return [{"date": dt.date.fromisoformat(r["date"]),
                 "spy": float(r["spy"]), "vix": float(r["vix"]),
                 "vix3m": float(r["vix3m"])}
                for r in csv.DictReader(fh)]


def forward_rv(rows: list[dict], i: int, horizon: int) -> float | None:
    """Annualised realised vol over the NEXT `horizon` sessions, vol points."""
    if i + horizon >= len(rows):
        return None
    rets = [rows[j + 1]["spy"] / rows[j]["spy"] - 1.0
            for j in range(i, i + horizon)]
    if len(rets) < 2:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return 100.0 * math.sqrt(var) * math.sqrt(252.0)


def load_gamma_regimes() -> dict[dt.date, bool]:
    out: dict[dt.date, bool] = {}
    if not GEX_CSV.exists():
        return out
    with GEX_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("flip_7") and r["spot"]:
                out[dt.date.fromisoformat(r["file_date"])] = (
                    float(r["spot"]) > float(r["flip_7"]))
    return out


def report(label: str, xs: list[float], screen: Screen, unit="vol pts"):
    s = overlap_corrected(xs, HORIZON)
    if s is None:
        print("  {0:<26} too few".format(label))
        return None
    print("  {0:<26} n={1:<5} ({2:>4.0f} indep)  mean {3:>+6.2f} {4}  "
          "t={5:>+5.2f}  (naive t {6:>+6.2f})".format(
              label, s["n"], s["n_indep"], s["mean"], unit, s["t"],
              s["naive_t"]))
    screen.record(label, xs, unit=unit, costed=False, descriptive=True)
    return s


def main() -> int:
    rows = load_rows()
    cut = int(len(rows) * (1 - HOLDOUT_FRACTION))
    ins = rows[:cut]
    print("{0:,} sessions; in-sample {1:,} ({2} .. {3}); holdout {4:,} SEALED"
          .format(len(rows), len(ins), ins[0]["date"], ins[-1]["date"],
                  len(rows) - cut))
    print("VRP = VIX - realised vol over the NEXT {0} sessions. Positive "
          "means selling won.\n".format(HORIZON))

    vrp: list[tuple[int, float]] = []
    for i in range(len(ins)):
        rv = forward_rv(ins, i, HORIZON)
        if rv is not None:
            vrp.append((i, ins[i]["vix"] - rv))

    screen = Screen(cost_bp=0.0)

    # ---------------------------------------------------------------- R1
    print("R1  POSITIVE CONTROL: is VRP positive on average?")
    print("    (it must be -- if not, the measurement is broken)")
    values = [v for _, v in vrp]
    s = report("VRP, all sessions", values, screen)
    if s:
        neg = sum(1 for v in values if v < 0) / len(values)
        print("    {0:.1%} of sessions had realised ABOVE implied -- those "
              "are the ones worth finding".format(neg))
    print()

    # ---------------------------------------------------------------- R2
    print("R2  PRIMARY: does the term-structure slope predict VRP?")
    print("    (prior: flatter/inverted slope -> LOWER, more often negative)")
    slopes = [ins[i]["vix3m"] - ins[i]["vix"] for i, _ in vrp]
    terciles = st.quantiles(slopes, n=3)

    def slope_bucket(x: float) -> str:
        if x < terciles[0]:
            return "flat/inverted"
        if x > terciles[1]:
            return "steep contango"
        return "middle"

    buckets: dict[str, list[float]] = {}
    for (i, v), sl in zip(vrp, slopes):
        buckets.setdefault(slope_bucket(sl), []).append(v)
    for name in ("flat/inverted", "middle", "steep contango"):
        report(name, buckets[name], screen)
    w = welch_overlap(buckets["flat/inverted"], buckets["steep contango"],
                      HORIZON)
    print("    -> flat/inverted minus steep: {0:+.2f} vol pts  t={1:+.2f}  "
          "95% CI [{2:+.2f}, {3:+.2f}]".format(
              w["diff"], w["t"], w["lo"], w["hi"]))
    screen.register_difference("R2 slope: flat minus steep", w, "vol pts")
    print()

    # ---------------------------------------------------------------- R3
    print("R3  CONTROL: does the slope add anything beyond the VIX LEVEL?")
    print("    high VIX and inversion co-occur; if level explains it, the")
    print("    slope is a restatement rather than a signal")
    levels = [ins[i]["vix"] for i, _ in vrp]
    lvl_terciles = st.quantiles(levels, n=3)
    for lab, lo, hi in (("low VIX", -1e9, lvl_terciles[0]),
                        ("mid VIX", lvl_terciles[0], lvl_terciles[1]),
                        ("high VIX", lvl_terciles[1], 1e9)):
        sub: dict[str, list[float]] = {}
        for (i, v), sl, lv in zip(vrp, slopes, levels):
            if lo <= lv < hi:
                sub.setdefault(slope_bucket(sl), []).append(v)
        a = sub.get("flat/inverted", [])
        b = sub.get("steep contango", [])
        if min(len(a), len(b)) < HORIZON * 2:
            print("  {0:<10} too few in one bucket".format(lab))
            continue
        w = welch_overlap(a, b, HORIZON)
        print("  {0:<10} flat {1:>+6.2f}  steep {2:>+6.2f}  ->  diff "
              "{3:+.2f} vol pts  t={4:+.2f}".format(
                  lab, st.mean(a), st.mean(b), w["diff"], w["t"]))
        screen.register_difference("R3 within " + lab, w, "vol pts")
    print()

    # ---------------------------------------------------------------- R4
    print("R4  gamma regime -- SAMPLE-LIMITED to the SPX positioning series")
    regimes = load_gamma_regimes()
    g: dict[str, list[float]] = {}
    for i, v in vrp:
        d = ins[i]["date"]
        if d in regimes:
            g.setdefault("long gamma" if regimes[d] else "short gamma",
                         []).append(v)
    if min((len(x) for x in g.values()), default=0) < HORIZON * 2:
        print("    only {0} overlapping sessions -- not testable"
              .format({k: len(v) for k, v in g.items()}))
    else:
        for name in ("short gamma", "long gamma"):
            report(name, g[name], screen)
        w = welch_overlap(g["short gamma"], g["long gamma"], HORIZON)
        print("    -> short minus long: {0:+.2f} vol pts  t={1:+.2f}".format(
            w["diff"], w["t"]))
        screen.register_difference("R4 gamma regime", w, "vol pts")

    print()
    screen.summary()
    print("\nAn edge below the {0:.1f} vol-point spread floor is not a trade."
          .format(COST_VOL_PTS))
    print("Hypotheses fixed before the first statistic was computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
