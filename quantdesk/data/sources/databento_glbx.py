"""Databento CME Globex OHLCV bars -> a usable ES series.

WHY ES AND NOT SPY
------------------
Every screen so far measured a signal carefully and then compared it to a
proxy: SPY daily closes. That is fine at daily frequency, where SPY and ES
move together, but the gamma hypothesis is intraday and partly overnight,
and SPY does not trade overnight at all. ES is also the contract actually
traded. So the outcome variable stops being a proxy.

WHAT THE FILE CONTAINS, AND THE TRAP IN IT
------------------------------------------
`ES.FUT` returns every ES instrument, not one series. In a year that is 27
instruments: nine quarterly OUTRIGHTS and eighteen CALENDAR SPREADS. The
spreads trade around 40-70 points while the outrights trade near 6,500, so
mixing them produces a price series that jumps by two orders of magnitude
and a "return" of -99% at every switch.

Spreads are 3.2% of volume, so a naive volume-weighted approach mostly
works and then breaks in a way that looks like a market event.

ROLLING
-------
There is no single continuous ES. On each session the front month is the
most heavily traded outright, and it changes four times a year. `front_month`
picks by daily volume, which is what "front month" means in practice.

ASSUMPTION (microstructure): no back-adjustment is applied, because nothing
here needs it. Within-session returns never span a roll, and levels are
compared to the parity-implied forward rather than to a spliced history. A
study that DOES span rolls must adjust first -- an unadjusted splice puts a
fake gap of several points into the series on four days a year.

ON COMPARING ES TO THE GAMMA LEVELS: `implied_spot` recovers the FORWARD
from put-call parity, and ES is itself a forward. So the two are directly
comparable, which the raw index level would not be -- the index sits below
both by carry.
"""

from __future__ import annotations

import collections
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

PRICE_SCALE = 1e-9

#: Below this, an "ES" instrument is a calendar spread, not an outright.
#: Outrights trade near 6,500 and spreads near 50, so the gap is three
#: orders of magnitude wide -- any threshold in between works, and this one
#: is stated rather than tuned.
OUTRIGHT_MIN_PRICE = 1000.0


@dataclass(frozen=True)
class Bar:
    instrument_id: int
    ts: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def read_bars(path: str | Path, outrights_only: bool = True) -> list[Bar]:
    """All OHLCV bars in a file, spreads filtered out by default."""
    import databento as db                      # presentation-layer import

    bars: list[Bar] = []
    for r in db.DBNStore.from_file(str(path)):
        close = int(r.close) * PRICE_SCALE
        if outrights_only and close < OUTRIGHT_MIN_PRICE:
            continue
        bars.append(Bar(
            instrument_id=int(r.instrument_id),
            ts=dt.datetime.fromtimestamp(
                int(r.ts_event) / 1e9, dt.timezone.utc),
            open=int(r.open) * PRICE_SCALE,
            high=int(r.high) * PRICE_SCALE,
            low=int(r.low) * PRICE_SCALE,
            close=close,
            volume=int(r.volume),
        ))
    return bars


def front_month(bars: list[Bar]) -> dict[dt.date, int]:
    """The most-traded outright on each session date.

    "Front month" is not a property you can read off a bar; it is whichever
    contract the volume has moved to. Picking by daily volume gets the roll
    right without a calendar, including the days either side of it when
    volume is genuinely split.
    """
    volume: dict[dt.date, collections.Counter] = {}
    for b in bars:
        volume.setdefault(b.ts.date(), collections.Counter())[
            b.instrument_id] += b.volume
    return {day: counter.most_common(1)[0][0]
            for day, counter in volume.items() if counter}


def session_bars(bars: list[Bar]) -> dict[dt.date, list[Bar]]:
    """Front-month minute bars, grouped by session date, time-ordered."""
    front = front_month(bars)
    out: dict[dt.date, list[Bar]] = {}
    for b in bars:
        day = b.ts.date()
        if front.get(day) == b.instrument_id:
            out.setdefault(day, []).append(b)
    for day in out:
        out[day].sort(key=lambda b: b.ts)
    return out


def roll_dates(bars: list[Bar]) -> list[dt.date]:
    """Sessions where the front month changed.

    Reported rather than hidden: any study spanning one of these needs a
    back-adjusted series, and this is the list of days to check.
    """
    front = front_month(bars)
    days = sorted(front)
    return [d for prev, d in zip(days, days[1:]) if front[prev] != front[d]]


def summary(bars: list[Bar]) -> dict:
    sessions = session_bars(bars)
    rolls = roll_dates(bars)
    days = sorted(sessions)
    return {
        "bars_total": len(bars),
        "sessions": len(sessions),
        "first_session": days[0] if days else None,
        "last_session": days[-1] if days else None,
        "instruments": len({b.instrument_id for b in bars}),
        "rolls": rolls,
        "median_bars_per_session": (
            sorted(len(v) for v in sessions.values())[len(sessions) // 2]
            if sessions else 0),
    }
