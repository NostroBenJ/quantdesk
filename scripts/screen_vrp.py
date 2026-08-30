"""
screen_vrp.py -- does the variance risk premium predict SPY returns?

PRE-REGISTERED. Hypotheses, horizons and quantile cuts written down before
the first run. Same sealed holdout as screen_term_structure: first 70%
in-sample, last 30% untouched by this file.

WHY THIS ONE IS DIFFERENT FROM EVERYTHING TESTED SO FAR
-------------------------------------------------------
Every hypothesis screened in this project so far was either a chart pattern
(no prior) or a vol forecast (a near-tautology). This one has a published,
replicated, directional prior: Bollerslev, Tauchen & Zhou (2009) find the
variance risk premium predicts forward equity returns, strongest at roughly
a one-quarter horizon. It has been re-tested across markets and decades by
people with no stake in this account.

That matters because it is the only thing separating a hypothesis from a
search. A prior found in someone else's data is evidence; a pattern found
in this data is a coincidence until it is not.

THE MECHANISM, stated so it can be wrong
----------------------------------------
VRP = VIX - trailing realised vol. It is what investors pay above the
recent cost of variance, so it proxies risk aversion. High VRP means fear
is expensive, which historically has been compensated: bear the risk others
are paying to avoid, collect the premium. Low or negative VRP means the
opposite and should not pay.

CAUSALITY
---------
VRP at i uses VIX at i and realised vol over the 21 sessions ENDING at i.
Forward returns start at i's close. `vol_indices.verify()` pins both ends:
trailing vol cannot see the future, forward return cannot see the past.
Using FORWARD realised vol here would predict returns beautifully and be
untradeable, because it reads the answer key.

THE HYPOTHESES

H1  Forward SPY return by VRP tercile. Directional prior: high VRP should
    pay MORE than low VRP. One-sided, stated in advance.
H2  The published horizon. BTZ find the effect strongest near a quarter,
    so 63 sessions is tested explicitly rather than only short horizons.
H3  Extreme negative VRP (bottom decile) -- realised vol above implied,
    which is where the premium has been overwhelmed. Should underperform.
H4  CONTROL: does the term-structure slope add anything beyond VRP? If
    the two carry the same information, the simpler one wins and VIX3M
    can be dropped entirely.

Run: python scripts/screen_vrp.py
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.research_stats import (  # noqa: E402
    Screen, describe_split, overlap_corrected, welch_overlap,
)
from quantdesk.data.vol_indices import (  # noqa: E402
    build, forward_return, non_overlapping, term_slope, variance_risk_premium,
)

HOLDOUT_FRACTION = 0.30
COST_BP = 1.3
LOOKBACK = 21


def main() -> int:
    rows = build()
    cut = int(len(rows) * (1 - HOLDOUT_FRACTION))
    ins = rows[:cut]

    vrps = {i: variance_risk_premium(ins, i, LOOKBACK) for i in range(len(ins))}
    valid = [v for v in vrps.values() if v is not None]

    print("in-sample {0:,} sessions ({1} .. {2}); holdout SEALED".format(
        len(ins), ins[0]["date"], ins[-1]["date"]))
    print("VRP = VIX - trailing {0}d realised vol, in vol points".format(
        LOOKBACK))
    print("  median {0:+.2f}   10th {1:+.2f}   90th {2:+.2f}   "
          "negative on {3:.1%} of sessions\n".format(
              st.median(valid), st.quantiles(valid, n=10)[0],
              st.quantiles(valid, n=10)[8],
              sum(1 for v in valid if v < 0) / len(valid)))

    terciles = st.quantiles(valid, n=3)
    lo_decile = st.quantiles(valid, n=10)[0]

    def bucket(v: float) -> str:
        if v < terciles[0]:
            return "low VRP"
        if v > terciles[1]:
            return "high VRP"
        return "mid VRP"

    screen = Screen(cost_bp=COST_BP)

    # ------------------------------------------------------------ H1, H2
    print("H1/H2  forward return by VRP tercile")
    print("       (prior: HIGH VRP should pay more than LOW)")
    for horizon in (5, 21, 63):
        buckets: dict[str, list[float]] = {}
        for i in non_overlapping(ins, horizon, start=LOOKBACK):
            v, r = vrps.get(i), forward_return(ins, i, horizon)
            if v is None or r is None:
                continue
            buckets.setdefault(bucket(v), []).append(10000.0 * r)
        for name in ("low VRP", "mid VRP", "high VRP"):
            screen.record("{0}, {1}d".format(name, horizon),
                          buckets.get(name, []))
        describe_split(buckets, "high VRP", "low VRP")
        print()

    # ---------------------------------------------------------------- H3
    print("H3  extreme negative VRP (bottom decile: realised above implied)")
    for horizon in (21, 63):
        tail, rest = [], []
        for i in non_overlapping(ins, horizon, start=LOOKBACK):
            v, r = vrps.get(i), forward_return(ins, i, horizon)
            if v is None or r is None:
                continue
            (tail if v <= lo_decile else rest).append(10000.0 * r)
        screen.record("bottom decile, {0}d".format(horizon), tail)
        screen.record("rest, {0}d".format(horizon), rest)
    print()

    # ---------------------------------------------------------------- H4
    print("H4  CONTROL: does the term slope add anything beyond VRP?")
    print("    (if these agree, drop VIX3M and use the simpler signal)")
    for horizon in (21, 63):
        slopes = [term_slope(ins[i]) for i in range(len(ins))]
        slope_med = st.median(slopes)
        agree_hi, disagree = [], []
        for i in non_overlapping(ins, horizon, start=LOOKBACK):
            v, r = vrps.get(i), forward_return(ins, i, horizon)
            if v is None or r is None:
                continue
            high_vrp = v > terciles[1]
            steep = term_slope(ins[i]) > slope_med
            # Where the two signals agree vs where they disagree. If the
            # slope carries nothing extra, these are the same population.
            (agree_hi if high_vrp == steep else disagree).append(10000.0 * r)
        screen.record("signals agree, {0}d".format(horizon), agree_hi)
        screen.record("signals disagree, {0}d".format(horizon), disagree)

    print()
    screen.summary()

    # ------------------------------------------------ re-estimation, not
    # a new hypothesis. H1/H2 above are underpowered at long horizons
    # because non-overlapping sampling throws away all but 1/h of the
    # data. Overlapping windows estimate the same mean far better; the
    # standard error is widened by sqrt(h) so nothing is claimed that the
    # independent information does not support. No new test is being
    # asked -- this is the same question, measured properly, so the
    # Bonferroni count above still governs.
    print("\n" + "=" * 72)
    print("RE-ESTIMATION of H1/H2 with overlapping windows")
    print("Same hypothesis, better estimator. The naive SE on overlapping")
    print("data is sqrt(h) too tight; these are corrected.")
    print("=" * 72)
    for horizon in (21, 63):
        buckets: dict[str, list[float]] = {}
        for i in range(LOOKBACK, len(ins) - horizon):
            v, r = vrps.get(i), forward_return(ins, i, horizon)
            if v is None or r is None:
                continue
            buckets.setdefault(bucket(v), []).append(10000.0 * r)
        print("\n  horizon {0}d".format(horizon))
        for name in ("low VRP", "mid VRP", "high VRP"):
            s = overlap_corrected(buckets.get(name, []), horizon)
            if s is None:
                continue
            print("    {0:<10} n={1:<5} ({2:>5.0f} indep)  mean {3:>+8.2f}bp  "
                  "t={4:>+5.2f}  (naive t would be {5:>+5.2f})".format(
                      name, s["n"], s["n_indep"], s["mean"], s["t"],
                      s["naive_t"]))
        w = welch_overlap(buckets.get("high VRP", []),
                          buckets.get("low VRP", []), horizon)
        if w:
            print("    -> high minus low: {0:+.2f}bp  t={1:+.2f}  "
                  "95% CI [{2:+.2f}, {3:+.2f}]".format(
                      w["diff"], w["t"], w["lo"], w["hi"]))
    print("\nThe prior says high VRP should outpay low. Read the SIGN and the")
    print("interval, not just the t -- a consistent sign across horizons on")
    print("an effect too small to resolve is what an underpowered test of a")
    print("real effect looks like, and also what noise looks like.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
