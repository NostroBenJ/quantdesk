"""
screen_gex.py -- does SPX dealer gamma predict how ES trades?

PRE-REGISTERED. Written and committed BEFORE the gamma series finished
building, so none of the hypotheses, horizons or cuts below could be chosen
after seeing a result. 7DTE was registered as the primary horizon earlier
still, before the build was launched at all.

Holdout: the last 30% of sessions is SEALED. This file never reads it.

WHAT THE MECHANISM ACTUALLY PREDICTS
------------------------------------
A dealer who is long gamma hedges by selling into rallies and buying dips,
which DAMPENS realised volatility. Short gamma hedges the other way and
AMPLIFIES it. So the honest primary prediction is about volatility, not
direction -- the same shape of claim the VIX term structure made, and it
should be tested the same way rather than dressed up as an alpha signal.

Spot below the flip is the short-gamma regime; above it is long gamma.

WHY DISTANCE FROM THE FLIP AND NOT NET GEX
------------------------------------------
Two findings from earlier in this project, both measured:

* The flip LEVEL is invariant to the dealer sign convention -- negating a
  function does not move its zero -- while the magnitude flips sign with it.
  Which side is long gamma is an assumption; where the line sits is not.
* The flip survives replacing per-strike IV with one flat vol (five of six
  test sessions moved it under 0.11% of spot) while net GEX does not
  (ratios ran 0.44 to 1.10).

A signal built on the magnitude inherits both fragilities. One built on
distance from the flip inherits neither.

THE HYPOTHESES

H1  PRIMARY. Forward realised volatility by regime. Short gamma (spot
    below flip) should realise MORE vol than long gamma. One-sided,
    directional, stated in advance.

H2  Intraday range, same mechanism measured a different way. If H1 holds
    and H2 does not, the effect is in close-to-close noise rather than in
    how the session actually traded.

H3  Trend versus chop. Long gamma should produce choppier sessions --
    dealers fading moves -- and short gamma cleaner trends. Measured by
    efficiency ratio, |close-open| / (high-low): low is chop, high is trend.

H4  Direction. Two-sided, and an expected NULL. Included because it is the
    first thing anyone asks and because a null is the informative answer.

H5  THE CONTROL, and the test that decides whether any of this is new.
    VIX already forecasts realised vol emphatically in this project
    (t=+9.58, measured, free). If the gamma regime only works because it
    correlates with VIX, it is a rediscovery with an options-data bill
    attached. H5 asks whether gamma separates high-vol from low-vol
    sessions WITHIN a VIX bucket -- that is, whether it adds anything.

COSTS: an MES round trip is ~1.3bp of a ~$38,600 contract. Effects are
reported against that floor, because a statistically real effect smaller
than the cost of trading it is not a trade.

Run: python scripts/screen_gex.py
"""

from __future__ import annotations

import csv
import datetime as dt
import glob
import math
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
VOL_CSV = ROOT / "spy_vix_term.csv"

HOLDOUT_FRACTION = 0.30
PRIMARY_DTE = 7
COST_BP = 1.3

#: Regular trading hours in ET. The gamma hedging that the mechanism
#: describes happens while the cash index is open, so the session measured
#: is RTH rather than the full 23-hour ES day.
RTH_OPEN = dt.time(9, 30)
RTH_CLOSE = dt.time(16, 0)
ET = dt.timezone(dt.timedelta(hours=-4))


# --------------------------------------------------------------- assembly

def load_gex() -> dict[dt.date, dict]:
    rows: dict[dt.date, dict] = {}
    with GEX_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            flip = r.get(f"flip_{PRIMARY_DTE}")
            if not flip or not r["spot"]:
                continue
            spot = float(r["spot"])
            rows[dt.date.fromisoformat(r["session"])] = {
                "spot": spot,
                "flip": float(flip),
                # POSITIVE means spot is ABOVE the flip: the long-gamma,
                # vol-suppressed regime.
                "distance_pct": 100.0 * (spot - float(flip)) / spot,
                "net_gex": float(r[f"net_gex_{PRIMARY_DTE}"] or 0.0),
                "used": int(r[f"used_{PRIMARY_DTE}"] or 0),
            }
    return rows


def es_sessions() -> dict[dt.date, dict]:
    """Per-session RTH statistics from ES front-month minute bars."""
    paths = glob.glob(ES_GLOB)
    if not paths:
        raise SystemExit(f"no ES files at {ES_GLOB}")
    bars = read_bars(paths[0])
    out: dict[dt.date, dict] = {}
    for day, day_bars in session_bars(bars).items():
        rth = [b for b in day_bars
               if RTH_OPEN <= b.ts.astimezone(ET).time() < RTH_CLOSE]
        if len(rth) < 60:                 # holiday or partial session
            continue
        closes = [b.close for b in rth]
        rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        if len(rets) < 30:
            continue
        hi = max(b.high for b in rth)
        lo = min(b.low for b in rth)
        op, cl = rth[0].open, rth[-1].close
        # Annualised from minute returns: 252 sessions x 390 RTH minutes.
        rv = st.stdev(rets) * math.sqrt(252 * 390) * 100.0
        out[day] = {
            "open": op, "close": cl, "high": hi, "low": lo,
            "rv": rv,
            "range_bp": 10000.0 * (hi - lo) / op,
            "ret_bp": 10000.0 * (cl / op - 1.0),
            # Efficiency: how much of the day's travel ended up as net
            # movement. Near 0 is chop, near 1 is a clean trend.
            "efficiency": abs(cl - op) / (hi - lo) if hi > lo else 0.0,
        }
    return out


def load_vix() -> dict[dt.date, float]:
    if not VOL_CSV.exists():
        return {}
    with VOL_CSV.open(encoding="utf-8") as fh:
        return {dt.date.fromisoformat(r["date"]): float(r["vix"])
                for r in csv.DictReader(fh) if r["vix"]}


# ------------------------------------------------------------------- main

def main() -> int:
    gex, es, vix = load_gex(), es_sessions(), load_vix()
    days = sorted(set(gex) & set(es))
    if len(days) < 40:
        print(f"only {len(days)} joined sessions -- has the build finished?")
        return 1

    cut = int(len(days) * (1 - HOLDOUT_FRACTION))
    ins = days[:cut]
    print("joined {0:,} sessions; in-sample {1:,} ({2} .. {3}); "
          "holdout {4:,} SEALED\n".format(
              len(days), len(ins), ins[0], ins[-1], len(days) - cut))

    dist = [gex[d]["distance_pct"] for d in ins]
    below = sum(1 for x in dist if x < 0)
    print("distance from spot to the {0}DTE flip, in % of spot:".format(
        PRIMARY_DTE))
    print("  median {0:+.3f}%   10th {1:+.3f}%   90th {2:+.3f}%".format(
        st.median(dist), st.quantiles(dist, n=10)[0],
        st.quantiles(dist, n=10)[8]))
    print("  SHORT gamma (spot below flip) on {0:,} of {1:,} sessions "
          "({2:.1%})\n".format(below, len(ins), below / len(ins)))

    def regime(day) -> str:
        return "short gamma" if gex[day]["distance_pct"] < 0 else "long gamma"

    screen = Screen(cost_bp=COST_BP)

    # ---------------------------------------------------------------- H1
    print("H1  PRIMARY: same-session realised vol by regime")
    print("    (prior: SHORT gamma realises MORE vol)")
    buckets: dict[str, list[float]] = {}
    for d in ins:
        buckets.setdefault(regime(d), []).append(es[d]["rv"])
    for name in ("short gamma", "long gamma"):
        screen.record(name, buckets.get(name, []), unit="vol pts", costed=False)
    describe_split(buckets, "short gamma", "long gamma", unit="vol pts")
    print()

    # ---------------------------------------------------------------- H2
    print("H2  intraday range by regime")
    buckets = {}
    for d in ins:
        buckets.setdefault(regime(d), []).append(es[d]["range_bp"])
    for name in ("short gamma", "long gamma"):
        screen.record(name + " range", buckets.get(name, []), costed=False)
    describe_split(buckets, "short gamma", "long gamma")
    print()

    # ---------------------------------------------------------------- H3
    print("H3  trend vs chop (efficiency = |close-open| / (high-low))")
    print("    (prior: LONG gamma is choppier, so LOWER efficiency)")
    buckets = {}
    for d in ins:
        buckets.setdefault(regime(d), []).append(100.0 * es[d]["efficiency"])
    for name in ("short gamma", "long gamma"):
        screen.record(name + " eff", buckets.get(name, []), unit="%",
                      costed=False)
    describe_split(buckets, "short gamma", "long gamma", unit="%")
    print()

    # ---------------------------------------------------------------- H4
    print("H4  direction (two-sided prior; a null is the expected answer)")
    buckets = {}
    for d in ins:
        buckets.setdefault(regime(d), []).append(es[d]["ret_bp"])
    for name in ("short gamma", "long gamma"):
        screen.record(name + " return", buckets.get(name, []))
    describe_split(buckets, "short gamma", "long gamma")
    print()

    # ---------------------------------------------------------------- H5
    print("H5  CONTROL: does gamma add anything BEYOND VIX?")
    print("    VIX already forecasts vol at t=+9.58 in this project. If the")
    print("    regimes do not separate WITHIN a VIX bucket, this is a")
    print("    rediscovery with a data bill attached.")
    have_vix = [d for d in ins if d in vix]
    if len(have_vix) < 40:
        print("    only {0} sessions have VIX -- skipped".format(len(have_vix)))
    else:
        levels = [vix[d] for d in have_vix]
        terciles = st.quantiles(levels, n=3)
        for label, lo, hi in (("low VIX", -1e9, terciles[0]),
                              ("mid VIX", terciles[0], terciles[1]),
                              ("high VIX", terciles[1], 1e9)):
            sub: dict[str, list[float]] = {}
            for d in have_vix:
                if lo <= vix[d] < hi:
                    sub.setdefault(regime(d), []).append(es[d]["rv"])
            if min(len(v) for v in sub.values()) < 5 if sub else True:
                print("    {0}: too few in one regime".format(label))
                continue
            for name in ("short gamma", "long gamma"):
                screen.record("{0}, {1}".format(label, name), sub[name],
                              unit="vol pts", costed=False)
            print("   ", label)
            describe_split(sub, "short gamma", "long gamma", unit="vol pts")

    print()
    screen.summary()
    print("\nPrimary horizon {0}DTE and every hypothesis above were "
          "registered before the series existed.".format(PRIMARY_DTE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
