"""Data layer: clock conventions, bar validation, partial-bar policy, storage."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from conftest import make_bars
from quantdesk.core.clock import DAY_1, Timeframe, ensure_utc, utc
from quantdesk.core.types import Bar, BarCompleteness, Direction, Signal, Trade
from quantdesk.data.bars import (
    DataQualityError,
    PartialBarPolicy,
    apply_partial_policy,
    average_true_range,
    find_gaps,
    label_completeness,
    normalise,
    parkinson_volatility,
    realized_volatility,
)
from quantdesk.data.store import BarStore
from quantdesk.data.sources.csv_source import CsvBarSource


# --------------------------------------------------------------------------- clock


def test_naive_datetimes_are_rejected():
    """Silently localising a naive timestamp shifts a whole dataset."""
    with pytest.raises(ValueError, match="naive datetime"):
        ensure_utc(datetime(2026, 1, 5, 14, 30))


def test_intraday_annualisation_uses_session_calendar():
    """5-minute bars: 78 per session * 252 sessions. Getting this wrong by using
    252 understates vol by sqrt(78) ~= 8.8x and oversizes every position."""
    tf = Timeframe.parse("5m")
    assert tf.periods_per_year == pytest.approx(19_656.0)
    assert tf.annualisation_factor == pytest.approx(math.sqrt(19_656.0))


def test_daily_annualisation_is_252():
    assert DAY_1.periods_per_year == pytest.approx(252.0)
    # A "daily" bar defined as one RTH session must agree.
    assert Timeframe("1d", 23_400).periods_per_year == pytest.approx(252.0)


def test_close_of_is_open_plus_duration():
    tf = Timeframe.parse("15m")
    assert tf.close_of(utc(2026, 1, 5, 14, 30)) == utc(2026, 1, 5, 14, 45)


# --------------------------------------------------------------------------- types


def test_bar_rejects_close_outside_range():
    with pytest.raises(ValueError, match="outside"):
        Bar(
            symbol="SPY",
            ts_open=utc(2026, 1, 5, 14, 30),
            ts_close=utc(2026, 1, 5, 14, 35),
            open=100.0,
            high=101.0,
            low=99.0,
            close=105.0,
            volume=1.0,
        )


def test_signal_confidence_bounds():
    with pytest.raises(ValueError, match="confidence"):
        Signal(
            module="m",
            ts=utc(2026, 1, 5),
            symbol="SPY",
            direction=Direction.LONG,
            confidence=1.5,
        )


def test_trade_pnl_is_gross_minus_costs():
    """Verified by hand: long 2 @ 100 -> 110, multiplier 2 => 40 gross, 5 costs."""
    trade = Trade(
        symbol="MNQ",
        module="m",
        direction=Direction.LONG,
        ts_entry=utc(2026, 1, 5, 14, 30),
        ts_exit=utc(2026, 1, 5, 15, 30),
        entry_price=100.0,
        exit_price=110.0,
        quantity=2.0,
        multiplier=2.0,
        commission=3.0,
        slippage=2.0,
    )
    assert trade.gross_pnl == pytest.approx(40.0)
    assert trade.net_pnl == pytest.approx(35.0)


def test_short_trade_pnl_sign():
    trade = Trade(
        symbol="SPY",
        module="m",
        direction=Direction.SHORT,
        ts_entry=utc(2026, 1, 5, 14, 30),
        ts_exit=utc(2026, 1, 5, 15, 30),
        entry_price=100.0,
        exit_price=90.0,
        quantity=1.0,
    )
    assert trade.gross_pnl == pytest.approx(10.0)


# ------------------------------------------------------------------- partial bars


def test_partial_bar_is_labelled_and_dropped(tf_5m):
    """The forming bar must never reach a strategy."""
    bars = make_bars(5, tf_5m)
    # as_of falls inside the final bar's window.
    as_of = bars[-1].ts_close - timedelta(seconds=60)
    labelled = label_completeness(bars, tf_5m, as_of)

    assert labelled[-1].completeness is BarCompleteness.PARTIAL
    assert all(b.completeness is BarCompleteness.COMPLETE for b in labelled[:-1])

    kept = apply_partial_policy(labelled, PartialBarPolicy.DROP)
    assert len(kept) == 4

    labelled_kept = apply_partial_policy(labelled, PartialBarPolicy.KEEP_LABELLED)
    assert len(labelled_kept) == 5

    with pytest.raises(DataQualityError):
        apply_partial_policy(labelled, PartialBarPolicy.RAISE)


def test_bar_not_spanning_full_period_is_partial(tf_5m):
    """A session-truncated bar is not comparable to its neighbours."""
    bars = make_bars(3, tf_5m)
    stunted = Bar(
        symbol=bars[-1].symbol,
        ts_open=bars[-1].ts_open,
        ts_close=bars[-1].ts_open + timedelta(seconds=120),  # only 2 of 5 minutes
        open=bars[-1].open,
        high=bars[-1].high,
        low=bars[-1].low,
        close=bars[-1].close,
        volume=bars[-1].volume,
    )
    labelled = label_completeness([*bars[:-1], stunted], tf_5m, utc(2030, 1, 1))
    assert labelled[-1].completeness is BarCompleteness.PARTIAL


def test_normalise_deduplicates_keeping_latest(tf_5m):
    """Re-fetching an overlapping window must upsert, not duplicate."""
    bars = make_bars(4, tf_5m)
    revised = Bar(
        symbol=bars[1].symbol,
        ts_open=bars[1].ts_open,
        ts_close=bars[1].ts_close,
        open=bars[1].open,
        high=bars[1].high,
        low=bars[1].low,
        close=bars[1].close,
        volume=999_999.0,  # vendor revised the volume
    )
    out = normalise([*bars, revised], tf_5m, utc(2030, 1, 1))
    assert len(out) == 4
    assert out[1].volume == pytest.approx(999_999.0)


def test_find_gaps_reports_missing_periods(tf_5m):
    bars = make_bars(6, tf_5m)
    with_hole = bars[:2] + bars[4:]
    gaps = find_gaps(with_hole, tf_5m)
    assert len(gaps) == 1
    assert gaps[0] == (bars[1].ts_close, bars[4].ts_open)


# ------------------------------------------------------------------- vol estimators


def test_realized_vol_recovers_a_known_value(tf_5m):
    """Numerical check, not a regression against a printed number.

    Alternating log returns of +r and -r give sum(r^2)/n = r^2 exactly, so the
    per-bar vol must be r and the annualised vol r * sqrt(periods_per_year).
    """
    r = 0.001
    bars = make_bars(101, tf_5m, wiggle=0.0)
    # Rebuild with exactly alternating returns.
    closes = [100.0]
    for i in range(100):
        closes.append(closes[-1] * math.exp(r if i % 2 == 0 else -r))
    rebuilt = [
        Bar(
            symbol="SPY",
            ts_open=b.ts_open,
            ts_close=b.ts_close,
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=1000.0,
            completeness=BarCompleteness.COMPLETE,
        )
        for b, c in zip(bars, closes)
    ]
    per_bar = realized_volatility(rebuilt, tf_5m, annualise=False)
    assert per_bar == pytest.approx(r, rel=1e-12)

    annual = realized_volatility(rebuilt, tf_5m, annualise=True)
    assert annual == pytest.approx(r * tf_5m.annualisation_factor, rel=1e-12)


def test_realized_vol_requires_enough_bars(tf_5m):
    with pytest.raises(DataQualityError):
        realized_volatility(make_bars(2, tf_5m), tf_5m)


def test_parkinson_recovers_a_known_range(tf_5m):
    """Constant high/low ratio => closed form sqrt(ln(h/l)^2 / (4 ln 2))."""
    bars = make_bars(50, tf_5m)
    ratio = 1.02
    fixed = [
        Bar(
            symbol="SPY",
            ts_open=b.ts_open,
            ts_close=b.ts_close,
            open=100.0,
            high=100.0 * ratio,
            low=100.0,
            close=100.0,
            volume=1000.0,
            completeness=BarCompleteness.COMPLETE,
        )
        for b in bars
    ]
    expected = math.sqrt(math.log(ratio) ** 2 / (4 * math.log(2)))
    assert parkinson_volatility(fixed, tf_5m, annualise=False) == pytest.approx(expected)


def test_atr_is_gap_aware(tf_5m):
    """True range must exceed the intrabar range when the market gaps.

    Built explicitly rather than via make_bars, which produces a continuous
    series where each bar opens at the previous close - and therefore has no gap
    for a gap-aware measure to notice.
    """
    template = make_bars(3, tf_5m)
    gapped = [
        Bar(symbol="SPY", ts_open=template[0].ts_open, ts_close=template[0].ts_close,
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0,
            completeness=BarCompleteness.COMPLETE),
        # Opens 10 points above the prior close: intrabar range 2, true range 11.
        Bar(symbol="SPY", ts_open=template[1].ts_open, ts_close=template[1].ts_close,
            open=110.0, high=111.0, low=109.0, close=110.0, volume=1000.0,
            completeness=BarCompleteness.COMPLETE),
    ]
    atr = average_true_range(gapped, period=1)
    intrabar = gapped[1].high - gapped[1].low
    assert intrabar == pytest.approx(2.0)
    assert atr == pytest.approx(11.0)
    assert atr > intrabar


# ------------------------------------------------------------------------ storage


def test_store_roundtrip_and_upsert(tmp_path, tf_5m):
    bars = make_bars(10, tf_5m)
    with BarStore(tmp_path / "bars.sqlite") as store:
        assert store.write(bars, tf_5m, "test") == 10
        back = store.read("SPY", tf_5m)
        assert len(back) == 10
        assert back[0].ts_open == bars[0].ts_open
        assert back[0].ts_open.tzinfo is timezone.utc

        # Re-writing the same window updates in place rather than duplicating.
        store.write(bars, tf_5m, "test")
        assert len(store.read("SPY", tf_5m)) == 10

        coverage = store.coverage("SPY", tf_5m)
        assert coverage is not None
        first, last, count = coverage
        assert count == 10
        assert first == bars[0].ts_open and last == bars[-1].ts_close


def test_store_hides_partial_bars_by_default(tmp_path, tf_5m):
    bars = make_bars(5, tf_5m)
    as_of = bars[-1].ts_close - timedelta(seconds=60)
    labelled = label_completeness(bars, tf_5m, as_of)
    with BarStore(tmp_path / "bars.sqlite") as store:
        store.write(labelled, tf_5m, "test")
        assert len(store.read("SPY", tf_5m)) == 4
        assert len(store.read("SPY", tf_5m, complete_only=False)) == 5


def test_store_upgrades_a_partial_bar_on_refetch(tmp_path, tf_5m):
    """Running ingest twice must heal the partial bar, not keep two rows."""
    bars = make_bars(5, tf_5m)
    early = label_completeness(bars, tf_5m, bars[-1].ts_close - timedelta(seconds=60))
    later = label_completeness(bars, tf_5m, utc(2030, 1, 1))
    with BarStore(tmp_path / "bars.sqlite") as store:
        store.write(early, tf_5m, "test")
        store.write(later, tf_5m, "test")
        assert len(store.read("SPY", tf_5m)) == 5


# ------------------------------------------------------------------------- sources


def test_csv_source_parses_offsets_and_filters_window(tmp_path, tf_5m):
    """Written against the real TradingView export shape already on disk."""
    path = tmp_path / "tv.csv"
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2026-07-06T14:40:00-04:00,751.07,751.36,751.07,751.24,1000\n"
        "2026-07-06T14:45:00-04:00,751.25,751.67,751.21,751.59,1200\n"
        "2026-07-06T14:50:00-04:00,751.60,751.90,751.50,751.80,900\n",
        encoding="utf-8",
    )
    source = CsvBarSource(path, "SPY", tf_5m)
    bars = source.fetch("SPY", tf_5m, utc(2026, 7, 6, 18, 40), utc(2026, 7, 6, 18, 50))

    assert len(bars) == 2  # end is exclusive
    assert bars[0].ts_open == utc(2026, 7, 6, 18, 40)  # 14:40 EDT -> 18:40 UTC
    assert bars[0].ts_close == utc(2026, 7, 6, 18, 45)


def test_csv_source_refuses_missing_volume_by_default(tmp_path, tf_5m):
    """Volume-relative cost models cannot run on invented liquidity."""
    path = tmp_path / "novol.csv"
    path.write_text(
        "time,open,high,low,close\n2026-07-06T14:40:00-04:00,751.07,751.36,751.07,751.24\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no 'volume' column"):
        CsvBarSource(path, "SPY", tf_5m).fetch(
            "SPY", tf_5m, utc(2026, 7, 6), utc(2026, 7, 7)
        )


def test_csv_source_rejects_naive_timestamps(tmp_path, tf_5m):
    path = tmp_path / "naive.csv"
    path.write_text(
        "time,open,high,low,close,volume\n2026-07-06 14:40:00,751.07,751.36,751.07,751.24,10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no timezone"):
        CsvBarSource(path, "SPY", tf_5m).fetch(
            "SPY", tf_5m, utc(2026, 7, 6), utc(2026, 7, 7)
        )
