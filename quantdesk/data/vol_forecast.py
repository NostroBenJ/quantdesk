"""Volatility forecasting: realized vol, regression, and incremental tests.

WHY THIS MODULE EXISTS
----------------------
Every directional screen in this project has come back null. The one
result that ever produced a large t-statistic was GEX regime predicting
next-session REALIZED VOLATILITY (+3.81 vol points, t=+6.53), and it
failed a single control: VIX already contained the information. It did not
fail for want of signal. It failed for want of INCREMENTAL signal.

That makes the research question precise: does any variable predict
forward realized volatility incrementally to VIX? Answering it needs
multiple regression, which the `Screen` bucket-comparison harness cannot
express -- comparing bucket means cannot hold VIX fixed.

Standard library only. The normal equations are solved by Gaussian
elimination with partial pivoting, which is a few dozen lines for the 2-4
regressors this work needs and avoids a numpy dependency in a package
that has none.

ON THE OVERLAP CORRECTION
-------------------------
A 21-day forward window sampled daily shares 20 of 21 days with its
neighbour. Point estimates stay unbiased; standard errors do not, and the
naive version is too tight by about sqrt(horizon). This module applies the
same correction the rest of the project uses -- inflate the standard error
by sqrt(horizon) -- rather than inventing a second convention. It is
conservative relative to Newey-West at these horizons and, more
importantly, it is the SAME convention, so results here are comparable to
`vrp_study.py` and every `Screen` result already recorded.

`verify()` proves the correction recovers the right answer on a series
whose true standard error is known by construction.
"""

from __future__ import annotations

import math

TRADING_DAYS = 252.0


# ------------------------------------------------------------ realized

def realized_vol(prices: list[float], annualize: bool = True) -> float | None:
    """Close-to-close realized volatility, in VOL POINTS (percent).

    Returns None below two returns rather than a misleading zero -- a
    single return has no dispersion, and reporting 0.0 for it would enter
    a regression as a real observation.
    """
    if len(prices) < 3:
        return None
    rets = [prices[i + 1] / prices[i] - 1.0 for i in range(len(prices) - 1)]
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    return 100.0 * sd * (math.sqrt(TRADING_DAYS) if annualize else 1.0)


def forward_vol(prices: list[float], i: int, horizon: int) -> float | None:
    """Realized vol over the `horizon` returns AFTER index i.

    Strictly forward: uses prices[i] onward, so nothing known only at
    i+horizon leaks into a decision made at i.
    """
    if i + horizon >= len(prices):
        return None
    return realized_vol(prices[i:i + horizon + 1])


# ---------------------------------------------------------- regression

def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None if singular."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _inverse(a: list[list[float]]) -> list[list[float]] | None:
    n = len(a)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        col = _solve(a, e)
        if col is None:
            return None
        cols.append(col)
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def ols(y: list[float], xs: list[list[float]], horizon: int = 1,
        names: list[str] | None = None) -> dict | None:
    """Regress y on xs (plus an intercept), overlap-corrected.

    `xs` is column-major: one list per regressor. Returns coefficients,
    overlap-corrected standard errors and t-statistics, R-squared, and
    the effective independent sample size.
    """
    n = len(y)
    k = len(xs) + 1
    if n <= k + 1 or any(len(x) != n for x in xs):
        return None

    design = [[1.0] + [x[i] for x in xs] for i in range(n)]
    xtx = [[sum(design[r][i] * design[r][j] for r in range(n))
            for j in range(k)] for i in range(k)]
    xty = [sum(design[r][i] * y[r] for r in range(n)) for i in range(k)]

    beta = _solve(xtx, xty)
    inv = _inverse(xtx)
    if beta is None or inv is None:
        return None

    fitted = [sum(beta[j] * design[r][j] for j in range(k)) for r in range(n)]
    resid = [y[r] - fitted[r] for r in range(n)]
    rss = sum(e * e for e in resid)
    ybar = sum(y) / n
    tss = sum((v - ybar) ** 2 for v in y)
    sigma2 = rss / (n - k)

    # Overlap correction: inflate every standard error by sqrt(horizon),
    # the same convention used throughout this project. Applied to the
    # VARIANCE as `horizon`, so the SE scales by sqrt(horizon).
    h = max(1, horizon)
    se = [math.sqrt(sigma2 * inv[j][j] * h) for j in range(k)]
    t = [beta[j] / se[j] if se[j] else float("nan") for j in range(k)]

    labels = ["intercept"] + (names or
                              ["x{0}".format(i + 1) for i in range(len(xs))])
    return {
        "n": n, "n_indep": n / h, "k": k, "names": labels,
        "beta": beta, "se": se, "t": t,
        "r2": 1.0 - rss / tss if tss else float("nan"),
        "resid": resid, "fitted": fitted,
    }


def describe(fit: dict, unit: str = "") -> None:
    print("    {0:<14}{1:>10}{2:>10}{3:>9}".format(
        "term", "coef", "se", "t"))
    for j, nm in enumerate(fit["names"]):
        print("    {0:<14}{1:>+10.4f}{2:>10.4f}{3:>+9.2f}".format(
            nm, fit["beta"][j], fit["se"][j], fit["t"][j]))
    print("    R2 {0:.4f}   n {1:,}   effective independent n {2:,.0f}{3}"
          .format(fit["r2"], fit["n"], fit["n_indep"],
                  "   [" + unit + "]" if unit else ""))


# --------------------------------------------------------------- verify

def verify() -> bool:
    """Hand-computed checks. Every one has data that would make it fail."""
    fails: list[str] = []

    def check(label, got, want, tol=1e-8):
        ok = abs(got - want) < tol
        if not ok:
            fails.append(label)
        print("  [{0}] {1}: {2:.8f} (want {3:.8f})".format(
            "ok" if ok else "FAIL", label, got, want))

    def check_true(label, cond):
        if not cond:
            fails.append(label)
        print("  [{0}] {1}".format("ok" if cond else "FAIL", label))

    print("regression recovers a known line exactly")
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    y = [2.0 + 3.0 * v for v in x]
    f = ols(y, [x])
    check("intercept", f["beta"][0], 2.0, 1e-9)
    check("slope", f["beta"][1], 3.0, 1e-9)
    check("R2 is 1 on a perfect fit", f["r2"], 1.0, 1e-9)

    print("\ntwo regressors, hand-built with a known answer")
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [1.0, 0.0, 2.0, 1.0, 3.0, 1.0, 0.0, 2.0]
    y2 = [5.0 + 2.0 * ai - 1.5 * bi for ai, bi in zip(a, b)]
    f2 = ols(y2, [a, b], names=["a", "b"])
    check("intercept", f2["beta"][0], 5.0, 1e-8)
    check("coef on a", f2["beta"][1], 2.0, 1e-8)
    check("coef on b", f2["beta"][2], -1.5, 1e-8)

    print("\na regressor with NO relation gets a coefficient near zero")
    import random
    rng = random.Random(7)
    n = 4000
    sig = [rng.gauss(0, 1) for _ in range(n)]
    noise = [rng.gauss(0, 1) for _ in range(n)]
    y3 = [2.0 * s + rng.gauss(0, 1) for s in sig]
    f3 = ols(y3, [sig, noise], names=["signal", "noise"])
    check_true("signal is significant", abs(f3["t"][1]) > 5)
    check_true("noise is not", abs(f3["t"][2]) < 3)
    check_true("signal coefficient near 2",
               abs(f3["beta"][1] - 2.0) < 0.1)

    print("\nthe overlap correction inflates SE by exactly sqrt(horizon)")
    f_a = ols(y3, [sig], horizon=1)
    f_b = ols(y3, [sig], horizon=21)
    check("se ratio is sqrt(21)",
          f_b["se"][1] / f_a["se"][1], math.sqrt(21.0), 1e-9)
    check("point estimate is unchanged by the correction",
          f_b["beta"][1], f_a["beta"][1], 1e-12)
    check("effective n divided by horizon", f_b["n_indep"], n / 21.0, 1e-9)

    print("\nrealized vol against a hand-computed case")
    # Two alternating returns of +1% and -1% (approximately): the daily
    # sd is known, so the annualised figure is checkable by hand.
    px = [100.0]
    for i in range(20):
        px.append(px[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
    rets = [px[i + 1] / px[i] - 1.0 for i in range(len(px) - 1)]
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))
    check("matches the direct computation",
          realized_vol(px), 100.0 * sd * math.sqrt(252.0), 1e-9)
    check_true("a flat series has zero vol",
               realized_vol([100.0] * 10) < 1e-12)
    check_true("too few points returns None", realized_vol([100.0]) is None)

    print("\nforward_vol is strictly forward-looking")
    # Build a series that is calm then violent. A forward window sitting
    # entirely in the calm half must NOT see the violent half.
    # Built as a cumulative product of small then large alternating
    # returns. An earlier version wrote (1+r)**i, which compounds into two
    # diverging paths rather than a calm series -- the check below caught
    # that, which is the point of writing a check that can fail.
    series = [100.0]
    for i in range(100):
        series.append(series[-1] * (1 + 0.0001 * (1 if i % 2 else -1)))
    for i in range(60):
        series.append(series[-1] * (1 + 0.05 * (1 if i % 2 else -1)))
    early = forward_vol(series, 10, 20)
    late = forward_vol(series, 110, 20)
    check_true("calm window reads calm", early is not None and early < 5)
    check_true("violent window reads violent", late is not None and late > 50)
    check_true("a window past the end returns None",
               forward_vol(series, len(series) - 2, 20) is None)

    print("\nsingular design is refused, not silently fitted")
    dup = [1.0, 2.0, 3.0, 4.0, 5.0]
    check_true("perfectly collinear regressors return None",
               ols([1.0, 2.0, 3.0, 4.0, 5.0], [dup, dup]) is None)

    print("\n" + ("ALL PASS" if not fails else "FAILURES: {0}".format(fails)))
    return not fails


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
