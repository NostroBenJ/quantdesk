"""
control_vol.py -- POSITIVE CONTROL for the volatility research programme.

PRE-REGISTERED. Written, with its pass criteria, before any statistic
below was computed.

WHAT THIS IS FOR
----------------
Before asking whether anything predicts realized volatility INCREMENTALLY
to VIX, the pipeline has to demonstrate it can recover facts that are
already known. If it cannot reproduce the textbook relationship between
implied and subsequent realized volatility, then any novel result it later
produces is uninterpretable -- it would be impossible to tell a discovery
from a bug.

This is deliberately a test we expect to PASS. That is what makes it
useful: it can only fail if something is wrong, and the failure would be
diagnostic rather than disappointing.

THE PASS CRITERIA, fixed in advance
-----------------------------------
C1  VRP POSITIVE. Mean (VIX - subsequent realized vol) > 0 with t > 3.
    Among the most replicated facts in the options literature: index
    implied vol exceeds subsequent realized vol by roughly 2-4 vol points.
    FAIL => the forward alignment is broken, most likely a sign or an
    off-by-one in which window is "subsequent".

C2  VIX PREDICTS. Regressing forward realized vol on VIX gives a positive
    slope with t > 5. FAIL => the join between the two series is wrong.

C3  SLOPE BELOW ONE. That slope should be < 1. VIX is a biased forecast:
    it overreacts, so a one-point rise in VIX predicts less than a
    one-point rise in realized vol. A slope at or above 1 would mean
    either the bias is absent (contradicting decades of literature) or
    realized vol has been computed on a different scale from implied --
    the likelier explanation, and exactly the kind of unit error that
    silently invalidates everything downstream.

C4  MATERIAL FIT. R-squared > 0.20. Implied vol is genuinely informative
    about future realized vol; a near-zero R2 means the series are
    misaligned even if a t-statistic happens to look fine.

C5  PLACEBO. VIX shuffled against the same targets must predict NOTHING:
    |t| < 3 and R2 < 0.02. This is the check that catches a pipeline
    which would make any input look predictive. If the placebo passes as
    "significant", every other number here is worthless.

C6  OVERLAP MATTERS. The naive standard error must be about sqrt(horizon)
    too tight. Demonstrated rather than asserted, because the whole
    project's error bars depend on this correction being real.

C7  VIX BEATS TRAILING VOL. In an encompassing regression of forward RV on
    BOTH VIX and trailing realized vol, VIX must survive. This is the one
    criterion here that is a genuine question rather than a formality --
    if implied vol adds nothing over simply extrapolating recent realized
    vol, then the entire premise of trading the vol surface is wrong, and
    better to learn that now than after building on it.

HORIZON: 21 trading days, matching VIX's own 30-calendar-day tenor. A
second run at 7 days against VIX9D checks the result is not an artefact of
one horizon.

Run: python scripts/control_vol.py
"""

from __future__ import annotations

import math
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.research_stats import overlap_corrected  # noqa: E402
from quantdesk.data.vix_curve import build  # noqa: E402
from quantdesk.data.vol_forecast import describe, forward_vol, ols  # noqa: E402

#: (implied tenor, forward trading days). VIX covers 30 calendar days,
#: about 21 trading; VIX9D covers 9 calendar, about 7 trading.
HORIZONS = (("VIX", 21), ("VIX9D", 7))

TRAILING_WINDOW = 21


def main() -> int:
    rows = build()
    prices = [r["spy"] for r in rows]
    print("{0:,} sessions {1} .. {2}\n".format(
        len(rows), rows[0]["date"], rows[-1]["date"]))

    verdicts: list[tuple[str, bool, str]] = []

    def judge(code, ok, detail):
        verdicts.append((code, ok, detail))
        print("    {0} {1}: {2}".format(
            "PASS" if ok else "FAIL", code, detail))

    for tenor, h in HORIZONS:
        primary = tenor == "VIX"
        print("=" * 72)
        print("{0} vs {1}-day forward realized vol{2}".format(
            tenor, h, "   <-- PRIMARY" if primary else ""))
        print("=" * 72)

        implied, fwd, trail = [], [], []
        for i in range(TRAILING_WINDOW, len(rows) - h - 1):
            f = forward_vol(prices, i, h)
            t = forward_vol(prices, i - TRAILING_WINDOW, TRAILING_WINDOW)
            if f is None or t is None:
                continue
            implied.append(rows[i][tenor])
            fwd.append(f)
            trail.append(t)

        print("  {0:,} overlapping observations, ~{1:,.0f} independent\n"
              .format(len(fwd), len(fwd) / h))

        # ------------------------------------------------------------ C1
        vrp = [a - b for a, b in zip(implied, fwd)]
        s = overlap_corrected(vrp, h)
        print("  C1  variance risk premium")
        print("      mean {0:+.2f} vol pts   t={1:+.2f}   "
              "({2:.1%} of sessions negative)".format(
                  s["mean"], s["t"],
                  sum(1 for v in vrp if v < 0) / len(vrp)))
        judge("C1", s["mean"] > 0 and s["t"] > 3,
              "implied exceeds subsequent realized by {0:+.2f} pts, t={1:+.2f}"
              .format(s["mean"], s["t"]))

        # ------------------------------------------------------------ C2/C3/C4
        fit = ols(fwd, [implied], horizon=h, names=[tenor])
        print("\n  C2/C3/C4  forward RV regressed on {0}".format(tenor))
        describe(fit)
        slope, tstat = fit["beta"][1], fit["t"][1]
        judge("C2", slope > 0 and tstat > 5,
              "slope {0:+.4f}, t={1:+.2f}".format(slope, tstat))
        judge("C3", slope < 1.0,
              "slope {0:.4f} {1} 1 -- VIX {2}".format(
                  slope, "<" if slope < 1 else ">=",
                  "overreacts, as documented" if slope < 1
                  else "does NOT overreact; suspect a scale mismatch"))
        judge("C4", fit["r2"] > 0.20, "R2 {0:.4f}".format(fit["r2"]))

        # ------------------------------------------------------------ C5
        shuffled = implied[:]
        random.Random(0).shuffle(shuffled)
        pl = ols(fwd, [shuffled], horizon=h, names=["shuffled " + tenor])
        print("\n  C5  PLACEBO: the same series, shuffled")
        describe(pl)
        judge("C5", abs(pl["t"][1]) < 3 and pl["r2"] < 0.02,
              "shuffled t={0:+.2f}, R2={1:.4f} -- {2}".format(
                  pl["t"][1], pl["r2"],
                  "predicts nothing, as it must" if abs(pl["t"][1]) < 3
                  else "PREDICTS, so the pipeline is broken"))

        # ------------------------------------------------------------ C6
        naive = ols(fwd, [implied], horizon=1, names=[tenor])
        ratio = fit["se"][1] / naive["se"][1]
        print("\n  C6  overlap correction")
        print("      naive SE {0:.5f} -> corrected {1:.5f}   "
              "ratio {2:.3f} (sqrt({3}) = {4:.3f})".format(
                  naive["se"][1], fit["se"][1], ratio, h, math.sqrt(h)))
        judge("C6", abs(ratio - math.sqrt(h)) < 1e-6,
              "naive errors are {0:.2f}x too tight".format(math.sqrt(h)))

        # ------------------------------------------------------------ C7
        enc = ols(fwd, [implied, trail], horizon=h,
                  names=[tenor, "trailing RV"])
        print("\n  C7  encompassing: does {0} survive against trailing RV?"
              .format(tenor))
        describe(enc)
        judge("C7", abs(enc["t"][1]) > 3,
              "{0} t={1:+.2f} controlling for trailing vol; trailing t={2:+.2f}"
              .format(tenor, enc["t"][1], enc["t"][2]))
        print()

    print("=" * 72)
    ok = sum(1 for _, p, _ in verdicts if p)
    print("POSITIVE CONTROL: {0} of {1} criteria passed".format(
        ok, len(verdicts)))
    for code, passed, detail in verdicts:
        if not passed:
            print("  FAILED {0}: {1}".format(code, detail))
    if ok == len(verdicts):
        print("\nThe pipeline reproduces the known implied-vs-realized")
        print("relationship. Novel results from it are now interpretable.")
    else:
        print("\nSTOP. A control failed, so any novel result would be")
        print("uninterpretable -- fix the pipeline before asking it anything.")
    return 0 if ok == len(verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
