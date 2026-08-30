"""CME Globex OHLCV reader and the ES front-month roll.

No network, no purchased data. The trap being tested is that `ES.FUT`
returns 27 instruments, not one series: nine quarterly outrights near 6,500
and eighteen calendar spreads near 50. Mixing them yields a price series
that jumps three orders of magnitude and a "return" of -99% at each switch.
"""

from __future__ import annotations

import datetime as dt

import pytest

from quantdesk.data.sources import databento_glbx as glbx
from quantdesk.data.sources.databento_glbx import (
    OUTRIGHT_MIN_PRICE,
    Bar,
    front_month,
    read_bars,
    roll_dates,
    session_bars,
    summary,
)

UTC = dt.timezone.utc


class FakeOHLCV:
    def __init__(self, instrument_id, ts, close, volume, open_=None):
        self.instrument_id = instrument_id
        self.ts_event = ts
        self.open = int((open_ if open_ is not None else close) * 1e9)
        self.high = int(close * 1e9)
        self.low = int(close * 1e9)
        self.close = int(close * 1e9)
        self.volume = volume


def at(day, hour=14, minute=0):
    return int(dt.datetime(2025, 9, day, hour, minute,
                           tzinfo=UTC).timestamp() * 1e9)


def patch(monkeypatch, records):
    import databento  # noqa: F401  (only to confirm the seam we replace)

    class FakeStore:
        @staticmethod
        def from_file(path):
            return iter(records)

    monkeypatch.setattr(glbx, "read_bars", read_bars)   # keep the real one
    monkeypatch.setitem(__import__("sys").modules, "databento",
                        type("M", (), {"DBNStore": FakeStore}))


def bar(instrument_id, day, close, volume=100, hour=14, minute=0):
    return Bar(instrument_id=instrument_id,
               ts=dt.datetime(2025, 9, day, hour, minute, tzinfo=UTC),
               open=close, high=close, low=close, close=close, volume=volume)


# ------------------------------------------------------- spread filtering


def test_spreads_are_filtered_out(monkeypatch):
    """A calendar spread at 50 and an outright at 6500 in one series would
    produce a -99% return at every switch."""
    patch(monkeypatch, [
        FakeOHLCV(1, at(1), 6500.0, 1000),
        FakeOHLCV(2, at(1), 50.0, 900),        # spread
    ])
    bars = read_bars("x")
    assert len(bars) == 1
    assert bars[0].instrument_id == 1


def test_spreads_can_be_kept_deliberately(monkeypatch):
    patch(monkeypatch, [
        FakeOHLCV(1, at(1), 6500.0, 1000),
        FakeOHLCV(2, at(1), 50.0, 900),
    ])
    assert len(read_bars("x", outrights_only=False)) == 2


def test_threshold_sits_between_the_two_populations():
    """Stated, not tuned: outrights ~6500, spreads ~50."""
    assert 100.0 < OUTRIGHT_MIN_PRICE < 6000.0


def test_prices_are_scaled_from_fixed_point(monkeypatch):
    patch(monkeypatch, [FakeOHLCV(1, at(1), 6427.25, 10)])
    assert read_bars("x")[0].close == pytest.approx(6427.25)


# ------------------------------------------------------------ front month


def test_front_month_is_the_most_traded_contract_each_day():
    bars = [bar(100, 1, 6500.0, volume=5_000),
            bar(200, 1, 6520.0, volume=1_000)]
    assert front_month(bars)[dt.date(2025, 9, 1)] == 100


def test_front_month_switches_when_volume_moves():
    """The roll is not a date you look up; it is where the volume went."""
    bars = [bar(100, 1, 6500.0, volume=9_000),
            bar(200, 1, 6520.0, volume=1_000),
            bar(100, 2, 6505.0, volume=1_000),
            bar(200, 2, 6525.0, volume=9_000)]
    front = front_month(bars)
    assert front[dt.date(2025, 9, 1)] == 100
    assert front[dt.date(2025, 9, 2)] == 200
    assert roll_dates(bars) == [dt.date(2025, 9, 2)]


def test_no_roll_reported_when_the_contract_is_stable():
    bars = [bar(100, d, 6500.0, volume=9_000) for d in (1, 2, 3)]
    assert roll_dates(bars) == []


def test_session_bars_exclude_the_back_month():
    bars = [bar(100, 1, 6500.0, volume=9_000, minute=0),
            bar(100, 1, 6501.0, volume=9_000, minute=1),
            bar(200, 1, 6520.0, volume=1_000, minute=0)]
    day = session_bars(bars)[dt.date(2025, 9, 1)]
    assert len(day) == 2
    assert {b.instrument_id for b in day} == {100}


def test_session_bars_are_time_ordered():
    bars = [bar(100, 1, 6500.0, minute=5),
            bar(100, 1, 6501.0, minute=1),
            bar(100, 1, 6502.0, minute=3)]
    day = session_bars(bars)[dt.date(2025, 9, 1)]
    assert [b.ts.minute for b in day] == [1, 3, 5]


def test_within_session_returns_never_span_a_roll():
    """The reason no back-adjustment is needed. Each session's bars are one
    contract, so a return computed inside a session is always clean."""
    bars = [bar(100, 1, 6500.0, volume=9_000, minute=0),
            bar(200, 2, 6900.0, volume=9_000, minute=0)]
    for day, day_bars in session_bars(bars).items():
        assert len({b.instrument_id for b in day_bars}) == 1


# ------------------------------------------------------------------ summary


def test_summary_reports_rolls_rather_than_hiding_them():
    bars = [bar(100, 1, 6500.0, volume=9_000),
            bar(200, 2, 6520.0, volume=9_000)]
    s = summary(bars)
    assert s["sessions"] == 2
    assert s["rolls"] == [dt.date(2025, 9, 2)]
    assert s["first_session"] == dt.date(2025, 9, 1)
    assert s["last_session"] == dt.date(2025, 9, 2)


def test_empty_input_does_not_crash():
    s = summary([])
    assert s["sessions"] == 0
    assert s["first_session"] is None
    assert s["rolls"] == []
