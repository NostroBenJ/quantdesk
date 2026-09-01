"""
screen_slope.py -- Johnson's VIX term-structure Slope, tested out of sample.

PRE-REGISTERED. Written before any statistic below was computed.

THE CLAIM BEING TESTED
----------------------
Johnson (2017), "Risk Premia and the VIX Term Structure", JFQA: a single
principal component of the VIX term structure -- Slope -- predicts the
excess returns of S&P 500 variance swaps, VIX futures and S&P 500
straddles, incrementally to other proxies for the conditional variance risk
premium. A relatively LOW slope predicts relatively HIGH returns to long
variance positions.

Restated in the quantity this project can measure: LOW SLOPE SHOULD PREDICT
LOW VRP, where VRP is implied minus subsequently realised volatility.
Low VRP means the option market undercharged and the long side won.

WHY THIS IS WORTH RUNNING RATHER THAN ASSUMING
----------------------------------------------
An earlier screen here used a two-point slope, VIX3M minus VIX, and found
exactly this direction at t=-1.09 -- underpowered, not significant, and at
the time indistinguishable from noise. Two things change now:

* The signal is the principal component of five tenors rather than one
  noisy contrast, which is Johnson's actual construction. The factor
  reproduces: 92.6% level, 6.7% slope, loadings running -0.640 at the
  9-day end to +0.608 at one year.
* The short end lets the horizon shrink. VIX9D against a 7-session forward
  window yields roughly three times the independent observations that a
  21-session window allowed from the same history.

THE TEST THAT ACTUALLY MATTERS
------------------------------
Johnson's sample ends around 2015. Published anomalies decay -- that is
itself a documented regularity, and a decade has passed. So the run is
split at 2016 and BOTH eras are reported:

    2011-2015   the paper's own era. If the effect is absent here, our
                implementation is wrong and nothing else can be read.
    2016-2024   after publication. This is the real question.

A result that holds in the first window and vanishes in the second is not a
strategy; it is a description of history. The most recent 30% is SEALED
regardless.

THE HYPOTHESES

S1  POSITIVE CONTROL. VRP positive on average at both horizons. It must be.

S2  PRIMARY. Slope predicts VRP, low slope giving low VRP. Directional
    prior, taken from the paper rather than from this data.

S3  CONTROL. Is Slope incremental to the LEVEL factor? Johnson says yes;
    PC1 carries 92.6% of curve variance, so if the level explains the
    effect then Slope is a restatement. This is the same control that left
    the gamma result labelled unproven.

S4  DECAY. Paper era against post-publication era, reported side by side.

Overlap corrected throughout: forward windows sampled daily share all but
one day, and the naive standard error is too tight by sqrt(horizon).

COST: a long straddle crosses four spreads round trip -- two legs in, two
out. Roughly 1.5 vol points on liquid SPY, three times the single-option
figure. ASSUMPTION (microstructure): stated, not measured; replace it with
a real chain before sizing anything.

Run: python scripts/screen_slope.py
"""

from __future__ import annotations

import datetime as dt
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.research_stats import (  # noqa: E402
    Screen, overlap_corrected, welch_overlap,
)
from quantdesk.data.vix_curve import (  # noqa: E402
    build, describe_loadings, factors, is_level_shaped, is_slope_shaped,
)

HOLDOUT_FRACTION = 0.30
PAPER_END = dt.date(2016, 1, 1)
COST_VOL_PTS = 1.5

#: (implied tenor, forward trading sessions). VIX9D covers 9 calendar days,
#: which is about 7 sessions; VIX covers 30, about 21.
HORIZONS = (("VIX9D", 7), ("VIX", 21))
PRIMARY = ("VIX9D", 7)


def forward_rv(rows, i: int, horizon: int) -> float | None:
    if i + horizon >= len(rows):
        return None
    rets = [rows[j + 1]["spy"] / rows[j]["spy"] - 1.0
            for j in range(i, i + horizon)]
    if len(rets) < 2:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return 100.0 * math.sqrt(var) * math.sqrt(252.0)


def bucket_by(values, xs, n=3):
    """Split `values` into terciles of `xs`, low to high."""
    cuts = st.quantiles(xs, n=n)
    out = {"low": [], "mid": [], "high": []}
    for v, x in zip(values, xs):
        key = "low" if x < cuts[0] else ("high" if x > cuts[1] else "mid")
        out[key].append(v)
    return out


def main() -> int:
    rows = build()
    f = factors(rows)

    print("{0:,} sessions {1} .. {2}".format(
        len(rows), rows[0]["date"], rows[-1]["date"]))
    describe_loadings(f)
    if not (is_level_shaped(f) and is_slope_shaped(f)):
        print("\nFACTORS ARE NOT LEVEL/SLOPE SHAPED -- stopping. A component "
              "named Slope whose loadings do not change sign is mislabelled, "
              "and every number below would inherit that.")
        return 1
    print()

    cut = int(len(rows) * (1 - HOLDOUT_FRACTION))
    print("in-sample {0:,} ({1} .. {2}); holdout {3:,} SEALED\n".format(
        cut, rows[0]["date"], rows[cut - 1]["date"], len(rows) - cut))

    screen = Screen()

    for tenor, horizon in HORIZONS:
        tag = "{0}/{1}d".format(tenor, horizon)
        primary = (tenor, horizon) == PRIMARY
        print("=" * 70)
        print("{0}{1}".format(tag, "   <-- PRIMARY" if primary else ""))
        print("=" * 70)

        idx, vrp, slope, level = [], [], [], []
        for i in range(cut):
            rv = forward_rv(rows, i, horizon)
            if rv is None:
                continue
            idx.append(i)
            vrp.append(rows[i][tenor] - rv)
            slope.append(f["slope"][i])
            level.append(f["level"][i])

        # ------------------------------------------------------------ S1
        s = overlap_corrected(vrp, horizon)
        neg = sum(1 for v in vrp if v < 0) / len(vrp)
        print("S1 control   VRP mean {0:+.2f} vol pts  t={1:+.2f}  "
              "({2:.1%} of sessions negative)".format(
                  s["mean"], s["t"], neg))

        # ------------------------------------------------------------ S2
        b = bucket_by(vrp, slope)
        w = welch_overlap(b["low"], b["high"], horizon)
        print("S2 PRIMARY   slope low {0:+.2f}   mid {1:+.2f}   high {2:+.2f}"
              .format(st.mean(b["low"]), st.mean(b["mid"]),
                      st.mean(b["high"])))
        print("             low minus high: {0:+.2f} vol pts  t={1:+.2f}  "
              "95% CI [{2:+.2f}, {3:+.2f}]".format(
                  w["diff"], w["t"], w["lo"], w["hi"]))
        screen.register_difference("S2 slope, " + tag, w, "vol pts")

        # ------------------------------------------------------------ S3
        lv = bucket_by(vrp, level)
        wl = welch_overlap(lv["low"], lv["high"], horizon)
        print("S3 control   LEVEL low minus high: {0:+.2f}  t={1:+.2f}"
              .format(wl["diff"], wl["t"]))
        screen.register_difference("S3 level, " + tag, wl, "vol pts")

        # Slope within level terciles: does it survive conditioning?
        for lab in ("low", "mid", "high"):
            sub_v, sub_s = [], []
            cuts = st.quantiles(level, n=3)
            for v, sl, lvv in zip(vrp, slope, level):
                key = ("low" if lvv < cuts[0]
                       else ("high" if lvv > cuts[1] else "mid"))
                if key == lab:
                    sub_v.append(v)
                    sub_s.append(sl)
            if len(sub_v) < horizon * 6:
                continue
            sb = bucket_by(sub_v, sub_s)
            if min(len(x) for x in sb.values()) < horizon * 2:
                continue
            ww = welch_overlap(sb["low"], sb["high"], horizon)
            print("             within {0:<4} level: slope low-high "
                  "{1:+.2f}  t={2:+.2f}".format(lab, ww["diff"], ww["t"]))
            screen.register_difference(
                "S3 slope within {0} level, {1}".format(lab, tag), ww,
                "vol pts")

        # ------------------------------------------------------------ S4
        print("S4 decay     ", end="")
        parts = []
        for era, lo, hi in (("2011-2015", dt.date(2000, 1, 1), PAPER_END),
                            ("2016-on", PAPER_END, dt.date(2100, 1, 1))):
            ev, es = [], []
            for i, v, sl in zip(idx, vrp, slope):
                if lo <= rows[i]["date"] < hi:
                    ev.append(v)
                    es.append(sl)
            if len(ev) < horizon * 12:
                parts.append("{0}: too few".format(era))
                continue
            eb = bucket_by(ev, es)
            if min(len(x) for x in eb.values()) < horizon * 3:
                parts.append("{0}: too few".format(era))
                continue
            we = welch_overlap(eb["low"], eb["high"], horizon)
            parts.append("{0}: {1:+.2f} (t={2:+.2f}, n={3})".format(
                era, we["diff"], we["t"], len(ev)))
            screen.register_difference(
                "S4 {0}, {1}".format(era, tag), we, "vol pts")
        print("   |   ".join(parts))
        print()

    screen.summary()
    print("\nA long straddle crosses four spreads round trip, roughly "
          "{0:.1f} vol points.".format(COST_VOL_PTS))
    print("Hypotheses and the Johnson direction were fixed before the first "
          "statistic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
