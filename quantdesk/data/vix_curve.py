"""The whole VIX term structure, and the Slope factor from it.

WHY THE WHOLE CURVE
-------------------
An earlier screen used a two-point slope, VIX3M minus VIX, and found the
right direction at t=-1.09. Johnson (2017, JFQA) uses a principal component
of the ENTIRE term structure and reports that this single factor predicts
the excess returns of variance swaps, VIX futures and S&P 500 straddles,
incrementally to other proxies for the conditional variance risk premium.

A two-point difference is one noisy contrast. A principal component of five
tenors is the same contrast estimated from five times the information.
That is the fix, and it costs nothing -- Cboe publishes every tenor free.

    VIX9D    9 days     from 2011-01-04
    VIX      30 days    from 1990-01-02
    VIX3M    93 days    from 2009-09-18
    VIX6M    6 months   from 2008-01-02
    VIX1Y    1 year     from 2007-01-03

VIX9D binds the joint sample to 2011, which is the price of having the
short end -- and the short end is where a slope factor carries most of its
information.

ON THE FACTORS
--------------
Term-structure PCA has a standard shape and it is worth CHECKING rather
than assuming: the first component is LEVEL (every loading the same sign,
the whole curve moving together) and the second is SLOPE (loadings change
sign across maturities, the short end moving against the long). `factors()`
returns both and `describe_loadings()` prints them, because a "slope"
factor whose loadings do not actually change sign is not a slope factor and
every result built on it would be mislabelled.

Standard library only -- the eigendecomposition is a Jacobi rotation, which
is a few dozen lines for a symmetric 5x5 and avoids a numpy dependency in
a package that has none.

Run `python -m quantdesk.data.vix_curve` for the self-test,
`--build` to fetch and cache.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import math
import os
import urllib.request

CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{0}_History.csv"
TENORS = ("VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y")
#: Approximate maturity in calendar days, for reading loadings in order.
TENOR_DAYS = {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 182, "VIX1Y": 365}
CACHE = "vix_curve.csv"
UA = {"User-Agent": "Mozilla/5.0"}


def _fetch(symbol: str) -> dict[dt.date, float]:
    req = urllib.request.Request(CBOE.format(symbol), headers=UA)
    with urllib.request.urlopen(req, timeout=40) as fh:
        text = fh.read().decode("utf-8")
    out: dict[dt.date, float] = {}
    for row in list(csv.reader(io.StringIO(text)))[1:]:
        if len(row) < 5 or not row[0]:
            continue
        try:
            # Cboe writes M/D/YYYY.
            month, day, year = (int(x) for x in row[0].split("/"))
            close = float(row[4])
        except (ValueError, IndexError):
            continue
        if close > 0:
            out[dt.date(year, month, day)] = close
    return out


def build(refresh: bool = False, cache: str = CACHE) -> list[dict]:
    """Joined daily curve plus SPY. Inner join -- a row missing any tenor
    cannot form a curve, and carrying one forward invents an observation."""
    if not refresh and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return [{"date": dt.date.fromisoformat(r["date"]),
                     "spy": float(r["spy"]),
                     **{t: float(r[t]) for t in TENORS}}
                    for r in csv.DictReader(fh)]

    curves = {t: _fetch(t) for t in TENORS}
    from .vol_indices import _spy                      # reuse the SPY fetch
    spy = {dt.date.fromisoformat(k): v for k, v in _spy("20y").items()}

    days = sorted(set.intersection(*[set(c) for c in curves.values()])
                  & set(spy))
    rows = [{"date": d, "spy": spy[d],
             **{t: curves[t][d] for t in TENORS}} for d in days]
    with open(cache, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "spy", *TENORS])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "date": r["date"].isoformat()})
    return rows


# ------------------------------------------------------------------ PCA

def _jacobi(a: list[list[float]], iterations: int = 100):
    """Eigenvalues and eigenvectors of a symmetric matrix, by rotation.

    Returns (eigenvalues, eigenvectors-as-columns), sorted descending.
    Exact for symmetric input up to convergence; the matrix here is 5x5, so
    this converges in a handful of sweeps.
    """
    n = len(a)
    m = [row[:] for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(iterations):
        off = math.sqrt(sum(m[i][j] ** 2
                            for i in range(n) for j in range(n) if i != j))
        if off < 1e-12:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(m[p][q]) < 1e-15:
                    continue
                theta = 0.5 * math.atan2(2.0 * m[p][q], m[q][q] - m[p][p])
                c, s = math.cos(theta), math.sin(theta)
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    vals = [m[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -vals[i])
    return ([vals[i] for i in order],
            [[v[r][i] for i in order] for r in range(n)])


def factors(rows: list[dict], tenors=TENORS):
    """Standardise the curve, then extract LEVEL and SLOPE.

    Each tenor is z-scored first so that a high-variance short end does not
    dominate purely because it moves more. The components are then in units
    of standard deviations of the curve, which is what makes them
    comparable across regimes.
    """
    cols = {t: [r[t] for r in rows] for t in tenors}
    mu = {t: sum(v) / len(v) for t, v in cols.items()}
    sd = {t: math.sqrt(sum((x - mu[t]) ** 2 for x in v) / (len(v) - 1))
          for t, v in cols.items()}
    z = [[(r[t] - mu[t]) / sd[t] for t in tenors] for r in rows]

    n, k = len(z), len(tenors)
    cov = [[sum(z[i][a] * z[i][b] for i in range(n)) / (n - 1)
            for b in range(k)] for a in range(k)]
    vals, vecs = _jacobi(cov)
    loadings = [[vecs[r][c] for r in range(k)] for c in range(k)]

    # Orient each component so its long-end loading is positive. PCA signs
    # are arbitrary, and an unpinned sign silently flips the direction of
    # every result built on the factor.
    for c in range(k):
        if loadings[c][-1] < 0:
            loadings[c] = [-x for x in loadings[c]]

    scores = [[sum(row[j] * loadings[c][j] for j in range(k))
               for c in range(k)] for row in z]
    total = sum(vals)
    return {
        "tenors": list(tenors),
        "eigenvalues": vals,
        "explained": [v / total for v in vals],
        "loadings": loadings,
        "level": [s[0] for s in scores],
        "slope": [s[1] for s in scores],
        "mu": mu, "sd": sd,
    }


def describe_loadings(f: dict) -> None:
    print("  {0:<10}{1}".format(
        "component", "".join("{0:>9}".format(t) for t in f["tenors"])))
    for c, name in ((0, "PC1 level"), (1, "PC2 slope")):
        print("  {0:<10}{1}   {2:.1%} of variance".format(
            name, "".join("{0:>9.3f}".format(x) for x in f["loadings"][c]),
            f["explained"][c]))


def is_slope_shaped(f: dict, component: int = 1) -> bool:
    """Does the component actually change sign across the curve?

    A factor called Slope whose loadings all share a sign is a second level
    factor wearing the wrong name, and every result built on it would be
    mislabelled. Checked rather than assumed.
    """
    signs = [x > 0 for x in f["loadings"][component]]
    return any(signs) and not all(signs)


def is_level_shaped(f: dict, component: int = 0) -> bool:
    signs = [x > 0 for x in f["loadings"][component]]
    return all(signs) or not any(signs)


# --------------------------------------------------------------- verify

def verify() -> bool:
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

    print("Jacobi against a known eigendecomposition")
    # [[2,1],[1,2]] has eigenvalues 3 and 1, vectors (1,1)/sqrt2, (1,-1)/sqrt2
    vals, vecs = _jacobi([[2.0, 1.0], [1.0, 2.0]])
    check("largest eigenvalue", vals[0], 3.0, 1e-9)
    check("second eigenvalue", vals[1], 1.0, 1e-9)
    check("eigenvector is unit length",
          math.hypot(vecs[0][0], vecs[1][0]), 1.0, 1e-9)
    check("eigenvector components equal in magnitude",
          abs(abs(vecs[0][0]) - abs(vecs[1][0])), 0.0, 1e-9)

    print("\ndiagonal matrix returns its own diagonal")
    vals, _ = _jacobi([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    check("sorted descending", vals[0], 5.0, 1e-9)
    check("smallest last", vals[2], 1.0, 1e-9)

    print("\neigenvalues sum to the trace")
    m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]
    vals, _ = _jacobi(m)
    check("trace preserved", sum(vals), 4.0 + 3.0 + 2.0, 1e-8)

    print("\nfactors on a SYNTHETIC curve with a known structure")
    # Build curves that are a level move plus a slope move, by construction.
    import random
    rng = random.Random(0)
    rows = []
    for _ in range(600):
        lvl = rng.gauss(20, 5)
        tilt = rng.gauss(0, 2)
        rows.append({
            "date": dt.date(2020, 1, 1), "spy": 300.0,
            "VIX9D": lvl - 1.5 * tilt, "VIX": lvl - 0.7 * tilt,
            "VIX3M": lvl + 0.1 * tilt, "VIX6M": lvl + 0.8 * tilt,
            "VIX1Y": lvl + 1.5 * tilt,
        })
    f = factors(rows)
    describe_loadings(f)
    check_true("PC1 is a LEVEL factor (all loadings one sign)",
               is_level_shaped(f))
    check_true("PC2 is a SLOPE factor (loadings change sign)",
               is_slope_shaped(f))
    check_true("the two together explain nearly everything",
               f["explained"][0] + f["explained"][1] > 0.98)
    check_true("PC2 loadings increase with maturity",
               all(a <= b + 1e-9 for a, b in
                   zip(f["loadings"][1], f["loadings"][1][1:])))

    print("\nsign is pinned, not left to chance")
    f2 = factors([{**r, **{t: -0.0 + r[t] for t in TENORS}} for r in rows])
    check_true("long-end loading always positive",
               f["loadings"][0][-1] > 0 and f2["loadings"][0][-1] > 0)

    print("\n" + ("ALL PASS" if not fails else "FAILURES: {0}".format(fails)))
    return not fails


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        rows = build(refresh=True)
        print("{0:,} joint sessions  {1} .. {2}".format(
            len(rows), rows[0]["date"], rows[-1]["date"]))
        f = factors(rows)
        describe_loadings(f)
        print("  slope-shaped:", is_slope_shaped(f))
    else:
        raise SystemExit(0 if verify() else 1)
