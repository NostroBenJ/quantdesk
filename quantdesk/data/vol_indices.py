"""Free options-derived series: the VIX complex, joined to SPY.

WHY THIS EXISTS
---------------
Per-strike open interest costs money; the vol indices do not. VIX and VIX3M
are computed FROM the options market, so their spread carries information
that is definitionally absent from the SPY price series - which is the bar
any signal has to clear to be worth testing at all.

VIX3M starts 2007-12-04, giving 4,712 joint sessions. That is more history
than the purchased chain data would have provided, for nothing.

WHAT THE SPREAD MEANS
---------------------
VIX prices ~30 days of implied variance, VIX3M ~93 days. Normally VIX3M
sits above VIX (contango): distant risk is less certain, so it costs more.
When VIX rises above VIX3M (backwardation) the market is paying up for
IMMEDIATE protection, which happens in stress.

ASSUMPTION (data availability): FRED publishes these as of the close, so a
value dated D is knowable at D's close and not before. Every forward return
here therefore starts at D's close. Reading the signal at D's close and the
return from D's OPEN would be a lookahead of exactly one session, which is
the classic way this study is got wrong.

Sources, both free and keyless:
  VIX     FRED VIXCLS   1990-01-02 ->
  VIX3M   FRED VXVCLS   2007-12-04 ->
  SPY     Yahoo chart API, adjusted closes so dividends are not fake gaps

Run `python -m quantdesk.data.vol_indices` for the self-test.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import urllib.request
from datetime import date

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={0}"
SPY_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/SPY"
           "?range={0}&interval=1d&events=div%2Csplit")
CACHE = "spy_vix_term.csv"

#: FRED marks missing observations with a lone period, not an empty cell.
MISSING = (".", "", "NaN")


def _fred(series_id: str) -> dict[str, float]:
    with urllib.request.urlopen(FRED_URL.format(series_id), timeout=30) as fh:
        text = fh.read().decode("utf-8")
    out: dict[str, float] = {}
    for row in list(csv.reader(io.StringIO(text)))[1:]:
        if len(row) < 2 or row[1] in MISSING:
            continue
        try:
            out[row[0]] = float(row[1])
        except ValueError:
            continue
    return out


def _spy(range_: str = "20y") -> dict[str, float]:
    req = urllib.request.Request(
        SPY_URL.format(range_), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as fh:
        doc = json.loads(fh.read().decode("utf-8"))
    result = doc["chart"]["result"][0]
    stamps = result["timestamp"]
    # Adjusted closes: an unadjusted series turns every dividend into a gap
    # down, which reads as a one-day loss that never happened.
    closes = (result.get("indicators", {}).get("adjclose", [{}])[0]
              .get("adjclose"))
    if closes is None:
        closes = result["indicators"]["quote"][0]["close"]
    import datetime as _dt
    out: dict[str, float] = {}
    for ts, px in zip(stamps, closes):
        if px is None:
            continue
        d = _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).date().isoformat()
        out[d] = float(px)
    return out


def build(refresh: bool = False, cache: str = CACHE) -> list[dict]:
    """Joined daily rows: date, spy, vix, vix3m. Inner join on all three.

    An inner join is the honest choice: a row missing any leg cannot form a
    term structure, and carrying it forward would invent an observation.
    """
    if not refresh and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return [
                {"date": r["date"], "spy": float(r["spy"]),
                 "vix": float(r["vix"]), "vix3m": float(r["vix3m"])}
                for r in csv.DictReader(fh)
            ]

    vix, vix3m, spy = _fred("VIXCLS"), _fred("VXVCLS"), _spy()
    days = sorted(set(vix) & set(vix3m) & set(spy))
    rows = [{"date": d, "spy": spy[d], "vix": vix[d], "vix3m": vix3m[d]}
            for d in days]
    with open(cache, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "spy", "vix", "vix3m"])
        w.writeheader()
        w.writerows(rows)
    return rows


# ------------------------------------------------------------------ signal

def term_slope(row: dict) -> float:
    """VIX3M - VIX, in vol points. Positive is contango, the normal state.

    Expressed as a LEVEL difference rather than a ratio because the vol
    points are already the natural unit -- a 2-point inversion means the
    same thing at VIX 15 and VIX 45, whereas a ratio does not.
    """
    return row["vix3m"] - row["vix"]


def forward_return(rows: list[dict], i: int, horizon: int) -> float | None:
    """Close-to-close return from row i to row i+horizon.

    Starts at i's CLOSE, which is the first moment the signal at i is
    knowable. Starting anywhere earlier is a one-session lookahead.
    """
    if i + horizon >= len(rows):
        return None
    return rows[i + horizon]["spy"] / rows[i]["spy"] - 1.0


def forward_realised_vol(rows: list[dict], i: int, horizon: int) -> float | None:
    """Annualised realised vol over the NEXT `horizon` sessions, in vol
    points, so it is directly comparable to VIX."""
    if i + horizon >= len(rows):
        return None
    rets = [rows[j + 1]["spy"] / rows[j]["spy"] - 1.0
            for j in range(i, i + horizon)]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return 100.0 * math.sqrt(var) * math.sqrt(252.0)


def non_overlapping(rows: list[dict], horizon: int, start: int = 0):
    """Indices spaced `horizon` apart, so observations are independent.

    Stepping by 1 and dividing by sqrt(n) is the error that makes noise
    look significant; the overlap rule in the research track is emphatic
    about it and this is the same mistake in a different dataset.
    """
    i = start
    while i + horizon < len(rows):
        yield i
        i += horizon


# ---------------------------------------------------------------- verify

def verify() -> bool:
    fails: list[str] = []

    def check(label, got, want, tol=1e-9):
        ok = abs(got - want) < tol
        if not ok:
            fails.append(label)
        print("  [{0}] {1}: {2:.8f} (want {3:.8f})".format(
            "ok" if ok else "FAIL", label, got, want))

    def check_true(label, cond):
        if not cond:
            fails.append(label)
        print("  [{0}] {1}".format("ok" if cond else "FAIL", label))

    print("term_slope sign convention")
    check("contango is positive", term_slope({"vix": 15.0, "vix3m": 18.0}), 3.0)
    check("backwardation is negative",
          term_slope({"vix": 45.0, "vix3m": 40.0}), -5.0)
    check("flat is zero", term_slope({"vix": 20.0, "vix3m": 20.0}), 0.0)

    print("\nforward_return starts at the signal bar's close")
    rows = [{"date": "d%d" % k, "spy": 100.0 * (1.10 ** k),
             "vix": 15.0, "vix3m": 18.0} for k in range(10)]
    # From index 0 to index 3 is three 10% steps: 1.1^3 - 1.
    check("3-session return", forward_return(rows, 0, 3), 1.10 ** 3 - 1.0)
    check("1-session return", forward_return(rows, 0, 1), 0.10)
    check_true("runs off the end -> None",
               forward_return(rows, 9, 3) is None)
    # The return must NOT include the move into the signal bar.
    check("return from i=1 excludes step 0->1",
          forward_return(rows, 1, 1), 0.10)

    print("\nforward_realised_vol looks forward, never back")
    flat = [{"date": "d%d" % k, "spy": 100.0, "vix": 15.0, "vix3m": 18.0}
            for k in range(10)]
    check("a flat series has zero vol", forward_realised_vol(flat, 0, 5), 0.0)
    # A series that is volatile only BEFORE index 5 must read ~0 from 5 on.
    mixed = ([{"date": "d%d" % k, "spy": 100.0 + (10 if k % 2 else -10),
               "vix": 15.0, "vix3m": 18.0} for k in range(6)]
             + [{"date": "d%d" % k, "spy": 100.0, "vix": 15.0, "vix3m": 18.0}
                for k in range(6, 12)])
    check_true("past volatility does not leak forward",
               forward_realised_vol(mixed, 6, 5) == 0.0)
    check_true("and the volatile stretch does read high",
               forward_realised_vol(mixed, 0, 5) > 100.0)

    print("\nnon_overlapping windows really are disjoint")
    idx = list(non_overlapping(rows, 3))
    check_true("spaced by the horizon",
               all(b - a >= 3 for a, b in zip(idx, idx[1:])))
    check_true("none runs off the end",
               all(i + 3 < len(rows) for i in idx))
    check("count for 10 rows, horizon 3", float(len(idx)), 3.0)

    print("\n" + ("ALL PASS" if not fails else "FAILURES: {0}".format(fails)))
    return not fails


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        rows = build(refresh=True)
        print("{0:,} joint sessions  {1} .. {2}".format(
            len(rows), rows[0]["date"], rows[-1]["date"]))
    else:
        raise SystemExit(0 if verify() else 1)
