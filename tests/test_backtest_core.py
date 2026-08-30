"""Causality, lookahead detection, costs and fills.

The lookahead tests are the ones that matter. They do not assert that features
are causal - they construct features that cheat, in three different ways, and
prove the guard catches each one.
"""

from __future__ import annotations

import math

import pytest

from conftest import make_bars
from quantdesk.core.clock import utc
from quantdesk.core.types import Bar, BarCompleteness, Order, Side
from quantdesk.backtest.costs import (
    CommissionSchedule,
    CostError,
    SlippageParams,
    VolatilityScaledCostModel,
    ZeroCostModel,
    side_crosses_spread,
)
from quantdesk.backtest.fills import ExponentialQueueFill, NextBarOpenFill
from quantdesk.backtest.lookahead import assert_causal, check_causality, perturb_future
from quantdesk.backtest.view import LookaheadError, MarketView


# ---------------------------------------------------------------------- the view


def test_view_shows_only_closed_bars(bars_flat):
    view = MarketView("SPY", bars_flat)
    cutoff = bars_flat[9].ts_close
    history = view.history(cutoff)

    assert len(history) == 10
    assert history[-1].ts_open == bars_flat[9].ts_open
    assert all(b.ts_close <= cutoff for b in history)


def test_view_boundary_is_inclusive_of_the_just_closed_bar(bars_flat):
    """At exactly ts_close the bar IS knowable - that instant is the decision point."""
    view = MarketView("SPY", bars_flat)
    assert view.visible_count(bars_flat[0].ts_close) == 1
    # One microsecond earlier it is not.
    from datetime import timedelta

    assert view.visible_count(bars_flat[0].ts_close - timedelta(microseconds=1)) == 0


def test_view_rejects_partial_bars(tf_5m):
    bars = make_bars(10, tf_5m, complete=False)
    with pytest.raises(LookaheadError, match="non-COMPLETE"):
        MarketView("SPY", bars)


def test_view_rejects_overlapping_bars(tf_5m):
    bars = make_bars(5, tf_5m)
    overlapping = Bar(
        symbol="SPY",
        ts_open=bars[2].ts_open,  # opens before bar[2] closed
        ts_close=bars[3].ts_close,
        open=bars[3].open,
        high=bars[3].high,
        low=bars[3].low,
        close=bars[3].close,
        volume=bars[3].volume,
        completeness=BarCompleteness.COMPLETE,
    )
    with pytest.raises(LookaheadError, match="[Oo]verlapping"):
        MarketView("SPY", [*bars[:3], overlapping, *bars[4:]])


def test_next_bar_after_returns_none_at_end(bars_flat):
    """No bar left to trade in means the order expires, not fills at the close."""
    view = MarketView("SPY", bars_flat)
    assert view.next_bar_after(bars_flat[-1].ts_close) is None
    assert view.next_bar_after(bars_flat[-2].ts_close) is bars_flat[-1]


def test_state_handed_to_a_module_cannot_reach_forward(bars_flat):
    view = MarketView("SPY", bars_flat)
    state = view.state(bars_flat[50].ts_close)
    assert state.last.ts_open == bars_flat[50].ts_open
    assert len(state) == 51
    # `history` is a tuple: a module cannot append to it or mutate it in place.
    with pytest.raises((AttributeError, TypeError)):
        state.history.append(bars_flat[51])  # type: ignore[attr-defined]


# ----------------------------------------------------------------- lookahead guard


def honest_feature(bars, i):
    """20-bar mean of closes up to and including i."""
    window = bars[max(0, i - 19) : i + 1]
    return sum(b.close for b in window) / len(window)


def blatant_cheat(bars, i):
    """Reads tomorrow's close."""
    return bars[min(i + 1, len(bars) - 1)].close


def subtle_cheat(bars, i):
    """Normalises by the maximum of the WHOLE series - including the future.

    This is the realistic version of the bug: nobody writes `bars[i+1]`, they
    write a normalisation over the full array and never think about it again.
    """
    return bars[i].close / max(b.close for b in bars)


def length_cheat(bars, i):
    """Uses the series length, which encodes how much future exists."""
    return bars[i].close * len(bars)


def test_honest_feature_passes_both_checks(bars_trending):
    report = assert_causal(honest_feature, bars_trending)
    assert report.passed
    assert "causal" in report.summary()


@pytest.mark.parametrize("cheater", [blatant_cheat, subtle_cheat, length_cheat])
def test_guard_catches_each_kind_of_leak(bars_trending, cheater):
    report = check_causality(cheater, bars_trending, name=cheater.__name__)
    assert not report.passed
    assert "LOOKAHEAD" in report.summary()
    with pytest.raises(LookaheadError):
        assert_causal(cheater, bars_trending)


def test_truncation_and_perturbation_catch_different_bugs(bars_trending):
    """Justifies running both rather than picking one.

    `length_cheat` survives perturbation - scaling future prices does not change
    the array length - but truncation catches it immediately.
    """
    report = check_causality(length_cheat, bars_trending, name="length_cheat")
    assert report.truncation_failures
    assert not report.perturbation_failures


def test_perturb_future_leaves_the_past_untouched(bars_flat):
    perturbed = perturb_future(bars_flat, 10, factor=2.0)
    assert perturbed[10].close == bars_flat[10].close
    assert perturbed[11].close == pytest.approx(bars_flat[11].close * 2.0)


# ------------------------------------------------------------------------- costs


def test_slippage_scales_with_volatility(spy, bars_flat):
    model = VolatilityScaledCostModel()
    bar = bars_flat[-1]
    low_vol = model.slippage_per_unit(spy, 10, bar, 0.001)
    high_vol = model.slippage_per_unit(spy, 10, bar, 0.010)
    assert high_vol > low_vol
    # A flat-constant model would be equal here; that is the bug being avoided.
    assert high_vol / low_vol > 2.0


def test_impact_grows_sublinearly_with_size(spy, bars_flat):
    """Square-root impact: 4x the size gives ~2x the impact, not 4x."""
    params = SlippageParams(spread_vol_fraction=0.0, min_half_spread_ticks=0.0)
    model = VolatilityScaledCostModel(params=params)
    bar = bars_flat[-1]
    small = model.slippage_per_unit(spy, 100, bar, 0.005)
    big = model.slippage_per_unit(spy, 400, bar, 0.005)
    assert big / small == pytest.approx(2.0, rel=1e-9)


def test_short_gamma_pays_the_spread_and_long_gamma_need_not(spy, bars_flat):
    """Bennett: a long-gamma hedge rests on the bid and offer; short must cross."""
    model = VolatilityScaledCostModel()
    bar = bars_flat[-1]
    crossing = model.slippage_per_unit(spy, 10, bar, 0.005, crosses_spread=True)
    resting = model.slippage_per_unit(spy, 10, bar, 0.005, crosses_spread=False)
    assert crossing > resting
    assert side_crosses_spread(Side.BUY, is_long_gamma=False) is True
    assert side_crosses_spread(Side.BUY, is_long_gamma=True) is False


def test_half_spread_has_a_tick_floor(spy, bars_flat):
    """Nothing trades tighter than the tick, however calm the tape."""
    model = VolatilityScaledCostModel(params=SlippageParams(impact_coefficient=0.0))
    cost = model.slippage_per_unit(spy, 1, bars_flat[-1], 0.0)
    assert cost == pytest.approx(0.5 * spy.tick_size)


def test_impact_refuses_to_price_zero_volume(spy, tf_5m):
    """Zero volume must raise, not silently imply infinite free liquidity."""
    bars = make_bars(5, tf_5m, volume=0.0)
    model = VolatilityScaledCostModel()
    with pytest.raises(CostError, match="volume"):
        model.slippage_per_unit(spy, 10, bars[-1], 0.005)


def test_commission_respects_minimum_ticket(mnq):
    schedule = CommissionSchedule(per_unit=0.35, per_trade_minimum=1.0)
    assert schedule.charge(1, 20_000.0, 2.0) == pytest.approx(1.0)
    assert schedule.charge(10, 20_000.0, 2.0) == pytest.approx(3.5)


# -------------------------------------------------------------------------- fills


def test_fill_happens_at_the_next_bar_open(spy, tf_5m):
    """The core anti-lookahead guarantee, at the fill level.

    Deliberately built with a GAP between the decision bar's close and the
    execution bar's open. On a continuous series the two prices are identical and
    the assertion cannot fail - a test that cannot fail proves nothing.
    """
    template = make_bars(2, tf_5m)
    decision = Bar(
        symbol="SPY", ts_open=template[0].ts_open, ts_close=template[0].ts_close,
        open=100.0, high=100.5, low=99.5, close=100.0, volume=10_000.0,
        completeness=BarCompleteness.COMPLETE,
    )
    execution = Bar(  # gaps up 5 points overnight
        symbol="SPY", ts_open=template[1].ts_open, ts_close=template[1].ts_close,
        open=105.0, high=106.0, low=104.5, close=105.5, volume=10_000.0,
        completeness=BarCompleteness.COMPLETE,
    )
    order = Order(
        ts=decision.ts_close, symbol="SPY", side=Side.BUY, quantity=5,
        reference_price=decision.close,
    )
    fill = NextBarOpenFill().fill(order, execution, spy, ZeroCostModel(), 0.001)

    assert fill is not None
    assert fill.price == pytest.approx(105.0)      # the execution bar's open
    assert fill.price != pytest.approx(100.0)      # NOT the price that decided it
    assert fill.ts == execution.ts_open


def test_buys_pay_up_and_sells_get_hit(spy, bars_flat):
    model = VolatilityScaledCostModel()
    bar = bars_flat[11]
    buy = NextBarOpenFill().fill(
        Order(ts=bar.ts_open, symbol="SPY", side=Side.BUY, quantity=1), bar, spy, model, 0.002
    )
    sell = NextBarOpenFill().fill(
        Order(ts=bar.ts_open, symbol="SPY", side=Side.SELL, quantity=1), bar, spy, model, 0.002
    )
    assert buy is not None and sell is not None
    assert buy.price > bar.open
    assert sell.price < bar.open


def test_size_above_participation_cap_fills_partially(spy, tf_5m):
    bars = make_bars(3, tf_5m, volume=1_000.0)
    order = Order(ts=bars[1].ts_open, symbol="SPY", side=Side.BUY, quantity=500)
    fill = NextBarOpenFill(max_participation=0.10).fill(
        order, bars[1], spy, ZeroCostModel(), 0.001
    )
    assert fill is not None
    assert fill.quantity == pytest.approx(100.0)  # 10% of 1000
    assert fill.is_partial
    assert fill.requested_quantity == 500


def test_fill_price_is_clamped_into_the_bar_range(spy, bars_flat):
    """A huge slippage estimate must not invent a price nobody traded at."""
    model = VolatilityScaledCostModel(params=SlippageParams(spread_vol_fraction=50.0))
    bar = bars_flat[11]
    fill = NextBarOpenFill().fill(
        Order(ts=bar.ts_open, symbol="SPY", side=Side.BUY, quantity=1), bar, spy, model, 0.05
    )
    assert fill is not None
    assert bar.low <= fill.price <= bar.high


def test_exponential_queue_fill_matches_the_stated_intensity(spy, bars_flat):
    """Lucic & Tse fill intensity: lambda(d) = lambda_0 * exp(-kappa * d)."""
    model = ExponentialQueueFill(kappa=0.75, depth_ticks=2.0)
    assert model.fill_fraction == pytest.approx(math.exp(-1.5))

    order = Order(ts=bars_flat[10].ts_close, symbol="SPY", side=Side.BUY, quantity=1000)
    fill = model.fill(order, bars_flat[11], spy, ZeroCostModel(), 0.001)
    assert fill is not None
    assert fill.quantity == pytest.approx(float(int(1000 * math.exp(-1.5))))
    assert fill.is_partial


def test_deeper_quotes_fill_less(spy, bars_flat):
    order = Order(ts=bars_flat[10].ts_close, symbol="SPY", side=Side.BUY, quantity=1000)
    shallow = ExponentialQueueFill(depth_ticks=1.0).fill(
        order, bars_flat[11], spy, ZeroCostModel(), 0.001
    )
    deep = ExponentialQueueFill(depth_ticks=5.0).fill(
        order, bars_flat[11], spy, ZeroCostModel(), 0.001
    )
    assert shallow is not None
    assert deep is None or deep.quantity < shallow.quantity


def test_fills_are_reproducible(spy, bars_flat):
    """Two identical runs must agree, or parameter changes cannot be told from noise."""
    order = Order(ts=bars_flat[10].ts_close, symbol="SPY", side=Side.BUY, quantity=100)
    model = ExponentialQueueFill(kappa=0.5, depth_ticks=1.0)
    a = model.fill(order, bars_flat[11], spy, ZeroCostModel(), 0.001)
    b = model.fill(order, bars_flat[11], spy, ZeroCostModel(), 0.001)
    assert a is not None and b is not None
    assert a.quantity == b.quantity and a.price == b.price
