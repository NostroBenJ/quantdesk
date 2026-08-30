"""
screen_term_structure.py -- is the VIX term structure worth trading?

PRE-REGISTERED. These four hypotheses, their priors, and the horizons were
written down before the script was run once. Nothing below was chosen after
seeing a result, and H3 is recorded as the weakest in advance so that a hit
there cannot be promoted afterwards.

In-sample is the first 70% (2007-12 .. ~2021). The last 30% is SEALED and
this file never touches it.

THE HYPOTHESES

H0  POSITIVE CONTROL -- forward realised vol by term-structure regime.
    This one MUST work. VIX3M-VIX is the option market's own statement
    about near versus far variance, so if backwardation does not predict
    higher near-term realised vol, the pipeline is broken and every other
    result here is uninterpretable. A screen without a control cannot tell
    "no effect" from "no working measurement", and that distinction is the
    whole difference between a finding and a bug.

H1  Forward SPY return by regime. Two-sided on purpose: "buy the fear"
    (stress is overpriced, forward returns are high) and "stress persists"
    (forward returns are low) are both defensible ex ante, so predicting a
    direction after the fact would be a story, not a prior.

H2  Extreme backwardation -- the bottom decile of slope. Prior: whatever
    effect exists should concentrate where the signal is largest. If the
    tail shows nothing while the middle does, that is a warning, not a
    discovery.

H3  CHANGE in slope rather than level. Weakest prior of the four: a
    5-day change is a noisier object than a level, and there is no
    published result I am leaning on. Stated as weak in advance.

Costs: an MES round trip is ~$1.50 commission plus a tick of slippage on a
~$38,600 contract, so roughly 1.3bp. Every effect below is reported in bp
against that floor -- a statistically real effect smaller than the floor is
still not a trade.

Run: python scripts/screen_term_structure.py
"""

from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.vol_indices import (  # noqa: E402
    build, forward_realised_vol, forward_return, non_overlapping, term_slope,
)
from quantdesk.data.research_stats import (  # noqa: E402
    Screen, describe_split,
)

HOLDOUT_FRACTION = 0.30
COST_BP = 1.3           # MES round trip, in basis points of notional


def main() -> int:
    rows = build()
    cut = int(len(rows) * (1 - HOLDOUT_FRACTION))
    insample = rows[:cut]

    print("{0:,} joint sessions total; in-sample {1:,} ({2} .. {3})".format(
        len(rows), len(insample), insample[0]["date"], insample[-1]["date"]))
    print("holdout of {0:,} sessions is SEALED and not read here.\n".format(
        len(rows) - cut))

    slopes = [term_slope(r) for r in insample]
    q = st.quantiles(slopes, n=10)
    lo_decile, hi_decile = q[0], q[8]
    tertiles = st.quantiles(slopes, n=3)
    print("term slope (VIX3M - VIX), in-sample:")
    print("  median {0:+.2f}   10th pct {1:+.2f}   90th pct {2:+.2f}".format(
        st.median(slopes), lo_decile, hi_decile))
    inverted = sum(1 for s in slopes if s < 0)
    print("  backwardated (slope < 0) on {0:,} of {1:,} sessions "
          "({2:.1%})\n".format(inverted, len(slopes), inverted / len(slopes)))

    def regime(slope: float) -> str:
        if slope < tertiles[0]:
            return "flat/inverted"
        if slope > tertiles[1]:
            return "steep contango"
        return "middle"

    screen = Screen(cost_bp=COST_BP)

    # ---------------------------------------------------------------- H0
    print("H0  POSITIVE CONTROL: forward realised vol by regime")
    print("    (must work, or the measurement is broken)")
    for horizon in (5, 21):
        buckets: dict[str, list[float]] = {}
        for i in non_overlapping(insample, horizon):
            rv = forward_realised_vol(insample, i, horizon)
            if rv is not None:
                buckets.setdefault(regime(term_slope(insample[i])),
                                   []).append(rv)
        for name in ("flat/inverted", "middle", "steep contango"):
            screen.record("{0}, {1}d fwd RV".format(name, horizon),
                          buckets.get(name, []), unit="vol pts", costed=False)
        describe_split(buckets, "flat/inverted", "steep contango",
                       unit="vol pts")
    print()

    # ---------------------------------------------------------------- H1
    print("H1  forward SPY return by regime (two-sided prior)")
    for horizon in (5, 21):
        buckets = {}
        for i in non_overlapping(insample, horizon):
            r = forward_return(insample, i, horizon)
            if r is not None:
                buckets.setdefault(regime(term_slope(insample[i])),
                                   []).append(10000.0 * r)
        for name in ("flat/inverted", "middle", "steep contango"):
            screen.record("{0}, {1}d fwd".format(name, horizon),
                          buckets.get(name, []))
        describe_split(buckets, "flat/inverted", "steep contango")
    print()

    # ---------------------------------------------------------------- H2
    print("H2  extreme backwardation (bottom decile of slope)")
    for horizon in (5, 21):
        tail, rest = [], []
        for i in non_overlapping(insample, horizon):
            r = forward_return(insample, i, horizon)
            if r is None:
                continue
            (tail if term_slope(insample[i]) <= lo_decile
             else rest).append(10000.0 * r)
        screen.record("bottom decile, {0}d".format(horizon), tail)
        screen.record("rest, {0}d".format(horizon), rest)
    print()

    # ---------------------------------------------------------------- H3
    print("H3  CHANGE in slope over 5 sessions (weakest prior, stated so)")
    for horizon in (5, 21):
        steepening, flattening = [], []
        for i in non_overlapping(insample, horizon, start=5):
            r = forward_return(insample, i, horizon)
            if r is None:
                continue
            delta = term_slope(insample[i]) - term_slope(insample[i - 5])
            (steepening if delta > 0 else flattening).append(10000.0 * r)
        screen.record("steepening, {0}d".format(horizon), steepening)
        screen.record("flattening, {0}d".format(horizon), flattening)

    print()
    screen.summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
