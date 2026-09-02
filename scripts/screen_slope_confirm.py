"""
screen_slope_confirm.py -- the term-structure slope, properly powered.

PRE-REGISTERED, AND SELECTED ON A PRIOR RESULT. Both facts stated up front
because the second one is what makes this test necessary rather than
sufficient.

WHY THIS VARIABLE AND NOT ANOTHER
---------------------------------
`screen_volregime.py` tested eight variables in four families against VIX
and none cleared correction. A1, the VIX term-structure slope, came
closest at t=-2.16 against a threshold of 2.73. I am re-running IT and not
the other seven. That is selection on an outcome, and it means a
significant result here is NOT independent evidence -- it is the same
observation looked at again with more data.

Three things make it worth doing anyway, and one makes it honest:

* The direction was predicted BEFORE the data by Johnson (2017, JFQA),
  which argues the curve's slope carries a risk premium the level does
  not. A steeper curve forecasting lower subsequent realized vol is his
  sign, not a sign chosen after the fact.
* It was nearly orthogonal to VIX (corr -0.17), which is the property
  that would make it useful rather than a VIX proxy.
* The earlier run was crippled by a design flaw of mine: I joined every
  family to COT, whose history starts in 2015, truncating the sample to
  ~79 independent observations when the slope never needed COT at all.
  Fixing that is repairing a mistake, not shopping for a p-value.

* And the honest part: the SEALED HOLDOUT settles it. In-sample here can
  only nominate. The holdout has never been read by any script in this
  project and is spent once, on E1, whatever comes back.

THE HOLDOUT BOUNDARY IS A DATE, NOT A FRACTION
----------------------------------------------
`screen_volregime.py` read in-sample data up to 2023-08-31. Taking "the
last 30%" of the longer sample used here would place the cut around 2021
and quietly pull two years of ALREADY-SEEN data into the holdout, which
would no longer be a holdout. The cut is therefore pinned to the same
2023-08-31 date, so the sealed period is genuinely unread.

POWER, the thing being fixed
----------------------------
Full VIX-curve sample from 2011 rather than COT's 2015, and a 7-day
forward window against VIX9D as the primary rather than 21 days. Shorter
horizon means less overlap (7/5 = 1.4 rather than 21/5 = 4.2), so the same
calendar span yields several times the independent observations. The
resolution is printed before the results either way.

THE HYPOTHESES -- three, corrected together

E1  PRIMARY. forward 7-day RV ~ VIX9D + slope. Two-sided: Johnson gives a
    directional prior, but a one-sided test that borrows its direction
    from the earlier run would be circular, so the harder test is used.
E2  Same at 21 days against VIX, matching the original specification.
E3  CONTROL THAT MATTERS. Add the LEVEL factor (PC1, 92.6% of curve
    variance) alongside VIX. If the slope's contribution is really the
    curve's level wearing a different name, it dies here. This is the
    control that killed the GEX result and it is applied to my own
    candidate on purpose.

PLACEBO: the slope, shuffled, must predict nothing. Reported but not
counted as a hypothesis -- it is a check on the pipeline, not a claim.

Run: python scripts/screen_slope_confirm.py
"""

from __future__ import annotations

import datetime as dt
import math
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.research_stats import inv_norm  # noqa: E402
from quantdesk.data.vix_curve import build, factors  # noqa: E402
from quantdesk.data.vol_forecast import describe, forward_vol, ols  # noqa: E402

#: Pinned to screen_volregime.py's in-sample boundary so the holdout is
#: genuinely unread rather than merely the tail of a longer series.
CUT_DATE = dt.date(2023, 8, 31)
STEP = 5                      # weekly sampling
TESTS = 3

SPECS = (
    ("E1", "VIX9D", 7),       # primary
    ("E2", "VIX", 21),
)


def assemble(rows, f, tenor, horizon):
    prices = [r["spy"] for r in rows]
    out = []
    for i in range(0, len(rows) - horizon - 1, STEP):
        fwd = forward_vol(prices, i, horizon)
        if fwd is None:
            continue
        out.append({"date": rows[i]["date"], "fwd": fwd,
                    "implied": rows[i][tenor],
                    "slope": f["slope"][i], "level": f["level"][i]})
    return out


def main() -> int:
    rows = build()
    f = factors(rows)
    thresh = abs(inv_norm(1 - 0.05 / (2 * TESTS)))

    print("VIX curve {0:,} sessions {1} .. {2}".format(
        len(rows), rows[0]["date"], rows[-1]["date"]))
    print("in-sample ends {0}; everything after is SEALED\n".format(CUT_DATE))

    verdict = {}
    for code, tenor, h in SPECS:
        recs = assemble(rows, f, tenor, h)
        ins = [r for r in recs if r["date"] <= CUT_DATE]
        overlap = h / STEP
        n_indep = len(ins) / overlap
        min_r = thresh / math.sqrt(max(n_indep - 3, 1))

        print("=" * 72)
        print("{0}  {1} vs {2}-day forward RV{3}".format(
            code, tenor, h, "   <-- PRIMARY" if code == "E1" else ""))
        print("=" * 72)
        print("  in-sample {0:,} weekly points, overlap {1:.1f}, "
              "~{2:.0f} independent".format(len(ins), overlap, n_indep))
        print("  resolution: |t| > {0:.2f} needs partial corr ~{1:.2f}"
              .format(thresh, min_r))

        y = [r["fwd"] for r in ins]
        iv = [r["implied"] for r in ins]
        sl = [r["slope"] for r in ins]
        lv = [r["level"] for r in ins]

        base = ols(y, [iv], horizon=overlap, names=[tenor])
        fit = ols(y, [iv, sl], horizon=overlap, names=[tenor, "slope"])
        print("\n  baseline R2 {0:.4f}".format(base["r2"]))
        describe(fit)
        t = fit["t"][2]
        verdict[code] = (t, fit["r2"] - base["r2"])
        print("  -> slope t={0:+.2f}  dR2 {1:+.4f}   {2}".format(
            t, fit["r2"] - base["r2"],
            "CLEARS" if abs(t) > thresh else "does not clear"))

        if code == "E1":
            # ------------------------------------------------------ E3
            enc = ols(y, [iv, lv, sl], horizon=overlap,
                      names=[tenor, "level (PC1)", "slope"])
            print("\n  E3  CONTROL: add the LEVEL factor")
            describe(enc)
            te = enc["t"][3]
            verdict["E3"] = (te, enc["r2"] - fit["r2"])
            print("  -> slope t={0:+.2f} with level present   {1}".format(
                te, "SURVIVES" if abs(te) > thresh
                else "dies -- it was the level"))

            # -------------------------------------------------- placebo
            sh = sl[:]
            random.Random(0).shuffle(sh)
            pl = ols(y, [iv, sh], horizon=overlap,
                     names=[tenor, "shuffled slope"])
            print("\n  placebo: shuffled slope t={0:+.2f}, dR2 {1:+.4f}  {2}"
                  .format(pl["t"][2], pl["r2"] - base["r2"],
                          "silent, as it must be" if abs(pl["t"][2]) < 2
                          else "SPEAKS -- pipeline suspect"))
        print()

    print("=" * 72)
    survivors = [k for k, (t, _) in verdict.items() if abs(t) > thresh]
    print("{0} hypotheses at |t| > {1:.2f}. Clears: {2}".format(
        TESTS, thresh, ", ".join(sorted(survivors)) or "NONE"))
    for k in ("E1", "E2", "E3"):
        if k in verdict:
            print("  {0}  t={1:+.2f}  dR2 {2:+.4f}".format(
                k, verdict[k][0], verdict[k][1]))

    e1_ok = "E1" in survivors and "E3" in survivors
    print()
    if e1_ok:
        print("E1 and E3 both clear in-sample. That NOMINATES the slope; it")
        print("does not confirm it, because this variable was chosen after")
        print("seeing it come closest in screen_volregime.py.")
        print("\nNEXT AND ONLY NEXT: spend the sealed holdout on E1, once.")
        print("Run: python scripts/screen_slope_confirm.py --spend-holdout")
    else:
        print("E1 did not clear in-sample with its control. The holdout is")
        print("NOT spent -- there is nothing to confirm, and spending it on a")
        print("candidate that failed its own nomination would waste it.")

    # ----------------------------------------------------------- holdout
    if "--spend-holdout" in sys.argv:
        if not e1_ok:
            print("\nREFUSING: E1 did not clear in-sample. The holdout is for")
            print("confirming a nomination, not for rescuing one.")
            return 1
        code, tenor, h = SPECS[0]
        recs = assemble(rows, f, tenor, h)
        out = [r for r in recs if r["date"] > CUT_DATE]
        overlap = h / STEP
        print("\n" + "=" * 72)
        print("SPENDING THE HOLDOUT -- {0:,} points {1} .. {2}".format(
            len(out), out[0]["date"], out[-1]["date"]))
        print("=" * 72)
        y = [r["fwd"] for r in out]
        iv = [r["implied"] for r in out]
        sl = [r["slope"] for r in out]
        b = ols(y, [iv], horizon=overlap, names=[tenor])
        fo = ols(y, [iv, sl], horizon=overlap, names=[tenor, "slope"])
        describe(fo)
        same_sign = (fo["beta"][2] < 0) == (verdict["E1"][0] < 0)
        print("  -> holdout slope t={0:+.2f}  dR2 {1:+.4f}  sign {2} in-sample"
              .format(fo["t"][2], fo["r2"] - b["r2"],
                      "MATCHES" if same_sign else "FLIPS from"))
        print("\n  The holdout is now spent. Whatever it says stands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
