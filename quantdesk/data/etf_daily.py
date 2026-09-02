"""Daily OHLC for a basket of ETFs, cached.

Built for the strategy-library shortlist, which needs two things the
existing caches do not have: OPENING prices (the overnight anomaly is a
close-to-open effect, so a close-only series cannot test it at all) and
several tickers at once for the momentum and trend strategies.

ASSUMPTION (data availability): Yahoo's chart API is the source, and its
adjusted closes handle dividends while its OPENS are unadjusted. That
mismatch matters here more than anywhere else in this project -- an
overnight return computed as adjusted_close(t) to raw_open(t+1) picks up
every dividend as a fake overnight gap. So this module keeps RAW opens and
RAW closes together, and the dividend adjustment is simply absent from
both legs rather than present in one. For a decomposition of one session
into its overnight and intraday halves, internal consistency matters more
than dividend accuracy.

Standard library only.

Run `python -m quantdesk.data.etf_daily --build` to fetch.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import urllib.request

CHART = ("https://query1.finance.yahoo.com/v8/finance/chart/{0}"
         "?range={1}&interval=1d")
UA = {"User-Agent": "Mozilla/5.0"}
CACHE = "etf_daily.csv"

#: SPY plus the nine original SPDR sectors plus a cross-asset set. Chosen
#: for length of history, not for what worked -- every one of these existed
#: before the sample starts, so there is no survivorship in the selection.
SECTORS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB")
ASSETS = ("SPY", "TLT", "GLD", "IEF", "EFA", "VNQ")
TICKERS = tuple(dict.fromkeys(("SPY",) + SECTORS + ASSETS))


def fetch(ticker: str, range_: str = "20y") -> dict[dt.date, dict]:
    req = urllib.request.Request(CHART.format(ticker, range_), headers=UA)
    with urllib.request.urlopen(req, timeout=40) as fh:
        doc = json.loads(fh.read().decode("utf-8"))
    result = doc["chart"]["result"][0]
    stamps = result["timestamp"]
    q = result["indicators"]["quote"][0]
    out: dict[dt.date, dict] = {}
    for i, ts in enumerate(stamps):
        o, c = q["open"][i], q["close"][i]
        if o is None or c is None or o <= 0 or c <= 0:
            continue
        d = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()
        out[d] = {"open": float(o), "close": float(c),
                  "high": float(q["high"][i] or c),
                  "low": float(q["low"][i] or c),
                  "volume": float(q["volume"][i] or 0)}
    return out


def build(refresh: bool = False, cache: str = CACHE) -> dict[str, dict]:
    """{ticker: {date: bar}}. Each ticker keeps its own calendar -- an
    inner join across all of them would truncate to the youngest ETF and
    throw away a decade of SPY."""
    if not refresh and os.path.exists(cache):
        out: dict[str, dict] = {}
        with open(cache, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.setdefault(r["ticker"], {})[
                    dt.date.fromisoformat(r["date"])] = {
                        k: float(r[k]) for k in
                        ("open", "high", "low", "close", "volume")}
        return out

    data = {}
    for t in TICKERS:
        try:
            data[t] = fetch(t)
            print("  {0:<5} {1:,} sessions".format(t, len(data[t])))
        except Exception as exc:                       # noqa: BLE001
            print("  {0:<5} FAILED {1}".format(t, type(exc).__name__))
    with open(cache, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "date", "open", "high", "low", "close", "volume"])
        for t, bars in data.items():
            for d in sorted(bars):
                b = bars[d]
                w.writerow([t, d.isoformat(), b["open"], b["high"],
                            b["low"], b["close"], b["volume"]])
    return data


def market_holidays(dates: list[dt.date]) -> set[dt.date]:
    """Weekdays with no session, inferred from gaps in the calendar.

    Derived rather than hard-coded: a weekday the market did not trade IS
    a holiday, and a table would drift out of date and silently mislabel
    the pre-holiday effect it exists to measure.
    """
    have = set(dates)
    out: set[dt.date] = set()
    cur, last = min(dates), max(dates)
    while cur <= last:
        if cur.weekday() < 5 and cur not in have:
            out.add(cur)
        cur += dt.timedelta(days=1)
    return out


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        data = build(refresh=True)
        print("\n{0} tickers cached".format(len(data)))
        spy = data["SPY"]
        days = sorted(spy)
        print("SPY {0:,} sessions  {1} .. {2}".format(
            len(days), days[0], days[-1]))
        hol = market_holidays(days)
        print("{0} inferred market holidays".format(len(hol)))
    else:
        print(__doc__)
