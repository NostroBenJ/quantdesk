"""
screen_volregime.py -- four families that might predict realized vol
INCREMENTALLY to VIX.

PRE-REGISTERED. Written before any statistic below was computed. The last
30% of the joint sample is SEALED and never read here.

THE QUESTION
------------
`control_vol.py` established that the pipeline reproduces the literature:
VRP +3.60 vol pts, VIX slope 0.8151 on forward realized vol, R2 0.369,
shuffled placebo t=+0.30. It also established that VIX subsumes trailing
realized vol entirely (VIX t=+5.92, trailing t=+0.18).

So VIX is the benchmark to beat, and beating it means one thing only:
surviving in a regression that already contains it. This project's single
large t-statistic -- GEX regime predicting next-session realized vol at
t=+6.53 -- died on exactly this test. Every hypothesis below is therefore
specified as a TWO-VARIABLE regression:

    forward_RV  ~  intercept + VIX + X

and the hypothesis is about X's coefficient, never about a bucket mean.
A variable that merely correlates with VIX will show nothing here, which
is the point.

THE FOUR FAMILIES, and why each is a candidate VIX might not price

A  TERM-STRUCTURE SHAPE. VIX is one point on a curve. Johnson (2017,
   JFQA) argues the curve's SLOPE carries a risk premium the level does
   not. The level/slope PCA is already built and verified in vix_curve.py
   (PC1 level 92.6%, PC2 slope 6.7%, loadings -0.640 at 9D to +0.608 at
   1Y). An earlier two-point version of this ran at t=-1.09, underpowered.

B  POSITIONING. COT net non-commercial interest. The premise is that
   crowded speculative positioning precedes volatile unwinds -- a
   quantity VIX, a price, need not reflect.

C  RATES. The 10Y-2Y curve and the pace of rate moves. Macro regime
   variables that plausibly lead equity vol without being in an equity
   option price.

D  VRP MEAN REVERSION. Whether the premium itself is predictable: when
   VIX is far above trailing realized vol, does that gap forecast
   anything beyond what VIX alone says?

THE LOOKAHEAD TRAP IN THIS DATA, closed explicitly
--------------------------------------------------
COT rows carry `date` (the TUESDAY survey) and `release_date` (the FRIDAY
publication). The positioning is not public until Friday afternoon.
Keying on `date` would be three days of lookahead, and three days is
enormous relative to a weekly signal -- it is precisely the kind of error
that produced the one-session lookahead caught earlier in this project,
where a pre-registered null came back at t=-4.95. Every COT observation
here is matched to the most recent report whose RELEASE date is strictly
before the observation date.

ASSUMPTION (data availability): the most recent COT rows carry a null
`release_date`. Those are given the historical +3 calendar day lag rather
than dropped, and the lag is verified against rows where both fields
exist rather than assumed.

SAMPLING AND OVERLAP
--------------------
Sampled WEEKLY to match COT's native frequency, with a 21-day forward
window. Consecutive observations therefore share 16 of 21 days, an
overlap factor of 21/5 = 4.2, and every standard error is inflated by
its square root. Means stay unbiased under overlap; errors do not.

THE HYPOTHESES -- eight, two per family, corrected as one family of eight

A1  VIX term-structure SLOPE (PC2)
A2  VIX9D / VIX ratio -- short-end steepness, a direct alternative
B1  COT net non-commercial as % of open interest
B2  the 52-week z-score of B1 -- extremes rather than levels
C1  10Y minus 2Y
C2  21-day change in the 10Y -- pace, not level
D1  VRP now: VIX minus trailing 21-day realized vol
D2  the 52-week z-score of D1

RESOLUTION is reported before the results. With this few independent
observations there is a floor on what can be detected at all, and a null
below that floor means "underpowered", not "absent". Saying which is the
difference between a finding and a shrug.

Run: python scripts/screen_volregime.py
"""

from __future__ import annotations

import datetime as dt
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.sources.lse_data import (  # noqa: E402
    fetch_bond_yields, fetch_cot,
)
from quantdesk.data.vix_curve import build, factors  # noqa: E402
from quantdesk.data.vol_forecast import forward_vol, ols  # noqa: E402

HOLDOUT_FRACTION = 0.30
HORIZON = 21          # trading days forward
STEP = 5              # sample weekly
TRAIL = 21
ZWINDOW = 52          # weeks, for z-scores
COT_LAG_DAYS = 3      # Tuesday survey -> Friday release


def zscore(series: list[float], i: int, window: int) -> float | None:
    """z of series[i] against the preceding `window` values. Strictly
    backward-looking: the point itself is excluded from its own mean."""
    if i < window:
        return None
    hist = series[i - window:i]
    m = st.mean(hist)
    s = st.pstdev(hist)
    return (series[i] - m) / s if s > 1e-12 else None


def main() -> int:
    rows = build()
    f = factors(rows)
    prices = [r["spy"] for r in rows]
    dates = [r["date"] for r in rows]

    # ---------------------------------------------------------- COT
    cot = fetch_cot("ES")
    lags = []
    for r in cot:
        if r.get("release_date") and r.get("date"):
            lags.append((dt.date.fromisoformat(r["release_date"])
                         - dt.date.fromisoformat(r["date"])).days)
    observed_lag = st.mode(lags) if lags else COT_LAG_DAYS
    print("COT: {0} reports; survey->release lag is {1} days on {2} of {3} "
          "rows".format(len(cot), observed_lag, lags.count(observed_lag),
                        len(lags)))
    if observed_lag != COT_LAG_DAYS:
        print("  NOTE: observed lag differs from the assumed {0}; using the "
              "observed value.".format(COT_LAG_DAYS))

    cot_pts = []
    for r in cot:
        try:
            d = dt.date.fromisoformat(r["date"])
            rel = (dt.date.fromisoformat(r["release_date"])
                   if r.get("release_date")
                   else d + dt.timedelta(days=observed_lag))
            oi = float(r["open_interest"])
            net = (float(r["noncomm_long"]) - float(r["noncomm_short"])) / oi
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            continue
        cot_pts.append((rel, net))
    cot_pts.sort()
    cot_rel = [p[0] for p in cot_pts]
    cot_net = [p[1] for p in cot_pts]
    cot_z = [zscore(cot_net, i, ZWINDOW) for i in range(len(cot_net))]

    def cot_at(d: dt.date):
        """Most recent report RELEASED strictly before d."""
        lo, hi = 0, len(cot_rel)
        while lo < hi:
            mid = (lo + hi) // 2
            if cot_rel[mid] < d:
                lo = mid + 1
            else:
                hi = mid
        j = lo - 1
        return (cot_net[j], cot_z[j]) if j >= 0 else (None, None)

    # -------------------------------------------------------- rates
    def yield_map(sym):
        out = {}
        for r in fetch_bond_yields(sym):
            try:
                out[dt.date.fromisoformat(r["date"])] = float(r["close"])
            except (ValueError, TypeError, KeyError):
                continue
        return out

    y10, y2 = yield_map("US10Y"), yield_map("US2Y")

    def last_on_or_before(m, d, limit_days=10):
        for k in range(limit_days):
            v = m.get(d - dt.timedelta(days=k))
            if v is not None:
                return v
        return None

    # ------------------------------------------------------ assemble
    recs = []
    for i in range(max(TRAIL, ZWINDOW * 5), len(rows) - HORIZON - 1, STEP):
        d = dates[i]
        fwd = forward_vol(prices, i, HORIZON)
        trail = forward_vol(prices, i - TRAIL, TRAIL)
        if fwd is None or trail is None:
            continue
        net, netz = cot_at(d)
        a = last_on_or_before(y10, d)
        b = last_on_or_before(y2, d)
        a_prev = last_on_or_before(y10, d - dt.timedelta(days=30))
        if None in (net, netz, a, b, a_prev):
            continue
        recs.append({
            "date": d, "fwd": fwd, "vix": rows[i]["VIX"],
            "A1": f["slope"][i],
            "A2": rows[i]["VIX9D"] / rows[i]["VIX"],
            "B1": net, "B2": netz,
            "C1": a - b, "C2": a - a_prev,
            "D1": rows[i]["VIX"] - trail,
        })

    d1 = [r["D1"] for r in recs]
    for i, r in enumerate(recs):
        r["D2"] = zscore(d1, i, ZWINDOW)
    recs = [r for r in recs if r["D2"] is not None]

    cut = int(len(recs) * (1 - HOLDOUT_FRACTION))
    ins = recs[:cut]
    overlap = HORIZON / STEP
    n_indep = len(ins) / overlap

    print("\n{0:,} weekly observations {1} .. {2}".format(
        len(recs), recs[0]["date"], recs[-1]["date"]))
    print("in-sample {0:,} ({1} .. {2}); holdout {3:,} SEALED".format(
        len(ins), ins[0]["date"], ins[-1]["date"], len(recs) - cut))
    print("overlap factor {0:.1f}; ~{1:.0f} independent observations".format(
        overlap, n_indep))

    tests = 8
    thresh = abs(__import__("quantdesk.data.research_stats",
                            fromlist=["inv_norm"]).inv_norm(
                                1 - 0.05 / (2 * tests)))
    min_r = thresh / math.sqrt(max(n_indep - 3, 1))
    print("\nRESOLUTION: Bonferroni at {0} tests needs |t| > {1:.2f}, which "
          "with".format(tests, thresh))
    print("~{0:.0f} independent points needs a partial correlation of about "
          "{1:.2f}.".format(n_indep, min_r))
    print("A null weaker than that means UNDERPOWERED, not absent.\n")

    y = [r["fwd"] for r in ins]
    vix = [r["vix"] for r in ins]

    labels = {
        "A1": "VIX curve SLOPE (PC2)", "A2": "VIX9D / VIX ratio",
        "B1": "COT net non-comm %OI", "B2": "COT net, 52w z-score",
        "C1": "10Y minus 2Y", "C2": "10Y 21-day change",
        "D1": "VRP now (VIX - trail RV)", "D2": "VRP, 52w z-score",
    }
    fams = {"A": "TERM STRUCTURE", "B": "POSITIONING",
            "C": "RATES", "D": "VRP MEAN REVERSION"}

    base = ols(y, [vix], horizon=overlap, names=["VIX"])
    print("baseline: forward RV ~ VIX     slope {0:+.4f}  t={1:+.2f}  "
          "R2={2:.4f}\n".format(base["beta"][1], base["t"][1], base["r2"]))

    results = []
    for fam in ("A", "B", "C", "D"):
        print("=" * 72)
        print("{0}  {1}".format(fam, fams[fam]))
        print("=" * 72)
        for key in (fam + "1", fam + "2"):
            x = [r[key] for r in ins]
            fit = ols(y, [vix, x], horizon=overlap, names=["VIX", key])
            if fit is None:
                print("  {0:<26} singular".format(labels[key]))
                continue
            # Collinearity with VIX matters for reading the t-stat.
            mv, mx = st.mean(vix), st.mean(x)
            sv = st.pstdev(vix) or 1e-12
            sx = st.pstdev(x) or 1e-12
            corr = sum((a - mv) * (b - mx) for a, b in zip(vix, x)) / (
                len(x) * sv * sx)
            t = fit["t"][2]
            dr2 = fit["r2"] - base["r2"]
            results.append((key, t, dr2, corr))
            print("  {0:<26} coef {1:+9.4f}  t={2:+6.2f}  dR2 {3:+.4f}  "
                  "corr(VIX) {4:+.2f}  {5}".format(
                      labels[key], fit["beta"][2], t, dr2, corr,
                      "CLEARS" if abs(t) > thresh else ""))
        print()

    print("=" * 72)
    survivors = [r for r in results if abs(r[1]) > thresh]
    print("{0} tests at |t| > {1:.2f}. Survivors: {2}".format(
        len(results), thresh,
        ", ".join(labels[k] for k, _, _, _ in survivors) or "NONE"))
    loose = [r for r in results if 1.96 < abs(r[1]) <= thresh]
    if loose:
        print("Would have passed an uncorrected 1.96: {0} -- {1:.1f} such "
              "hits are expected from {2} tests on noise.".format(
                  ", ".join(labels[k] for k, _, _, _ in loose),
                  0.05 * len(results), len(results)))
    print("\nEvery test held VIX fixed. A variable that only correlates with")
    print("VIX cannot show up here, which is the entire point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
