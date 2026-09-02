"""London Strategic Edge intraday futures, paged and cached.

WHY THIS EXISTS
---------------
The Goldbot ablation needed 5-minute GOLD and had only hourly. Databento
would sell 1-minute GC for roughly $25-40; this vendor's free tier carries
`GC.F` at 5-minute resolution from 2016-05-01, which is 10.3 years against
the 2.4 years of hourly Yahoo the project had been making do with.

Free data earns less trust than paid data, not more, so `verify_bars()`
below checks the things that actually go wrong rather than assuming a
clean tape. Run it before believing any result built on this.

THE PAGING RULE
---------------
The server caps a response at 5,000 rows however large a `limit` is
requested -- asking for a million returns one month and no error. It also
accepts only a DATE as `start`, rejecting a full timestamp outright, so
the cursor advances by calendar day rather than by bar. 5,000 five-minute
bars is about 17 days of a near-24-hour session, so each page overlaps the
last by up to a day; duplicates are dropped by timestamp rather than
assumed absent.

Two failure modes are handled explicitly because both are silent:

* A page that returns rows but no NEW maximum timestamp would loop
  forever. Progress is asserted, not hoped for.
* A gap longer than one page (a holiday week, a vendor outage) would end
  the walk early and look like the end of history. The walk continues
  past an empty page by stepping the cursor forward, and only stops after
  several consecutive empties.

Rows are written incrementally, so an interrupted run resumes from the
cache instead of re-fetching what it already paid for in rate limit.

ASSUMPTION (data availability): timestamps come back as ISO-8601 UTC with
a trailing Z. They are stored verbatim and parsed at the boundary, so a
vendor change of format fails loudly at parse time rather than silently
shifting every bar by an offset.

The API key is read from LSE_API_KEY and is never written to the cache,
logged, or printed.

Run `python -m quantdesk.data.sources.lse_data --fetch GC.F` to build,
without arguments for the self-test.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import time
from pathlib import Path

PAGE_ROWS = 5000            # server-side cap; requesting more is ignored
PAUSE_SECONDS = 0.4         # polite spacing between requests
MAX_EMPTY_PAGES = 3         # tolerate holiday gaps before declaring the end

RESOLUTION_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440,
}


def cache_path(symbol: str, resolution: str) -> Path:
    safe = symbol.replace("/", "_").replace(".", "_")
    return Path("{0}_{1}_lse.csv".format(safe.lower(), resolution))


def _parse(ts: str) -> dt.datetime:
    """ISO-8601 UTC -> aware datetime. Rejects anything naive.

    A silently localised series is shifted by the author's offset, which
    then looks like alpha -- so this refuses rather than guesses.
    """
    out = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if out.tzinfo is None:
        raise ValueError("naive timestamp from vendor: {0!r}".format(ts))
    return out.astimezone(dt.timezone.utc)


def load(symbol: str, resolution: str = "5m") -> list[dict]:
    """Read the cache. Stdlib only -- no vendor client needed to use data
    already on disk."""
    path = cache_path(symbol, resolution)
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out.append({
                    "ts": _parse(r["ts"]),
                    "open": float(r["open"]), "high": float(r["high"]),
                    "low": float(r["low"]), "close": float(r["close"]),
                    "volume": float(r["volume"] or 0),
                })
            except (ValueError, KeyError):
                continue
    return out


def fetch(symbol: str, resolution: str = "5m", start: str = "2016-01-01",
          verbose: bool = True) -> list[dict]:
    """Page the whole history into the cache, resuming if one exists.

    The vendor client is imported here rather than at module scope so the
    package stays importable in a bare interpreter.
    """
    from lse import LSE                                   # noqa: PLC0415

    if not os.environ.get("LSE_API_KEY"):
        raise SystemExit("LSE_API_KEY is not set in the environment.")

    step = dt.timedelta(minutes=RESOLUTION_MINUTES.get(resolution, 5))
    # How much wall-clock one full page covers, used to step past gaps.
    page_span = max(dt.timedelta(days=1), (step * PAGE_ROWS) - dt.timedelta(days=3))

    have = load(symbol, resolution)
    seen = {b["ts"] for b in have}
    # The server takes a DATE only. Resume from the last cached bar's own
    # day rather than the day after -- the tail of that day may be
    # missing, and duplicates are dropped below anyway.
    cursor = max(seen).date() if seen else dt.date.fromisoformat(start)
    if verbose and have:
        print("resuming: {0:,} cached bars, from {1}".format(
            len(have), cursor))

    client = LSE()
    path = cache_path(symbol, resolution)
    new_file = not path.exists()
    empties = 0
    added = 0

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["ts", "open", "high", "low", "close", "volume"])

        today = dt.datetime.now(dt.timezone.utc).date()
        while cursor <= today:
            rows = client.candles(
                symbol, resolution, limit=PAGE_ROWS, order="asc",
                start=cursor.isoformat())

            fresh = []
            for r in rows or []:
                ts = _parse(r["timestamp"])
                if ts in seen:
                    continue
                seen.add(ts)
                fresh.append((ts, r))

            if not fresh:
                # An empty page is ambiguous: end of history, or a gap
                # wider than one request. Step past it a few times before
                # concluding the tape has ended.
                empties += 1
                if empties >= MAX_EMPTY_PAGES:
                    break
                cursor += page_span
                continue

            empties = 0
            fresh.sort(key=lambda x: x[0])
            for ts, r in fresh:
                writer.writerow([r["timestamp"], r["open"], r["high"],
                                 r["low"], r["close"], r["volume"]])
                added += 1
            fh.flush()

            # Advance to the newest day received. Progress is asserted
            # rather than trusted: a page that returns rows without moving
            # the cursor forward would spin forever on the same window.
            newest_day = fresh[-1][0].date()
            if newest_day < cursor:
                raise RuntimeError(
                    "cursor went backwards: {0} -> {1}".format(
                        cursor, newest_day))
            if newest_day == cursor and len(fresh) < PAGE_ROWS:
                # Whole page fell inside the cursor day and did not fill:
                # that day is exhausted, so step off it explicitly.
                cursor = cursor + dt.timedelta(days=1)
            else:
                cursor = newest_day

            if verbose:
                print("  {0} .. {1}  (+{2:,}, {3:,} total)".format(
                    fresh[0][0].date(), fresh[-1][0].date(), len(fresh),
                    added))
            time.sleep(PAUSE_SECONDS)

    if verbose:
        print("added {0:,} bars".format(added))
    return load(symbol, resolution)


# --------------------------------------------------------------- verify

def verify_bars(bars: list[dict], resolution: str = "5m") -> bool:
    """Check the tape for the things that actually go wrong.

    Free data is not more trustworthy than paid data, so these are checks
    rather than assumptions. Every one of them has a way to fail.
    """
    fails: list[str] = []

    def check(label, cond, detail=""):
        if not cond:
            fails.append(label)
        print("  [{0}] {1}{2}".format(
            "ok" if cond else "FAIL", label,
            "  " + detail if detail else ""))

    if not bars:
        print("  [FAIL] no bars loaded")
        return False

    check("bars are strictly increasing in time",
          all(a["ts"] < b["ts"] for a, b in zip(bars, bars[1:])))
    check("no duplicate timestamps",
          len({b["ts"] for b in bars}) == len(bars),
          "{0:,} unique of {1:,}".format(
              len({b["ts"] for b in bars}), len(bars)))
    bad_ohlc = [b for b in bars
                if not (b["low"] <= b["open"] <= b["high"]
                        and b["low"] <= b["close"] <= b["high"])]
    check("open and close sit inside high-low", not bad_ohlc,
          "{0} violations".format(len(bad_ohlc)))
    check("no non-positive prices",
          all(min(b["open"], b["high"], b["low"], b["close"]) > 0
              for b in bars))

    step = dt.timedelta(minutes=RESOLUTION_MINUTES.get(resolution, 5))
    on_grid = sum(1 for b in bars
                  if b["ts"].minute % max(1, step.seconds // 60) == 0)
    check("timestamps land on the resolution grid",
          on_grid == len(bars),
          "{0:,} of {1:,}".format(on_grid, len(bars)))

    # A weekday with no bars at all is a vendor gap, not a market fact.
    days = {b["ts"].date() for b in bars}
    span = (max(days) - min(days)).days
    weekdays = sum(1 for i in range(span + 1)
                   if (min(days) + dt.timedelta(days=i)).weekday() < 5)
    covered = sum(1 for d in days if d.weekday() < 5)
    print("  [--] weekday coverage: {0:,} of {1:,} ({2:.1%})".format(
        covered, weekdays, covered / weekdays if weekdays else 0))
    print("  [--] span {0} .. {1}, {2:,} bars".format(
        min(days), max(days), len(bars)))

    print("\n" + ("ALL PASS" if not fails else "FAILURES: {0}".format(fails)))
    return not fails


if __name__ == "__main__":
    import sys
    if "--fetch" in sys.argv:
        i = sys.argv.index("--fetch")
        sym = sys.argv[i + 1] if len(sys.argv) > i + 1 else "GC.F"
        res = sys.argv[i + 2] if len(sys.argv) > i + 2 else "5m"
        got = fetch(sym, res)
        print()
        verify_bars(got, res)
    else:
        cached = load("GC.F", "5m")
        if not cached:
            print(__doc__)
        else:
            print("GC.F 5m from cache")
            verify_bars(cached, "5m")
