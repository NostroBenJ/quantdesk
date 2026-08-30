"""Metrics, walk-forward windowing, and the grading logic.

Metrics are checked against hand-computed values, not against whatever the code
printed the first time it ran.
"""

from __future__ import annotations

import math

import pytest

from quantdesk.core.clock import utc
from quantdesk.core.types import Direction, Trade
from quantdesk.backtest.metrics import (
    MIN_MEANINGFUL_TRADES,
    equity_stats,
    TradeStats,
    drawdown_series,
    max_drawdown,
    period_returns,
    sharpe_ratio,
    sortino_ratio,
)
from quantdesk.backtest.report import (
    CostSensitivity,
    Grade,
    cost_sensitivity,
    grade_strategy,
)
from quantdesk.backtest.walkforward import (
    WalkForwardResult,
    rolling_windows,
    run_walk_forward,
)


def trade(pnl: float, costs: float = 0.0, module: str = "m") -> Trade:
    """A trade whose NET P&L is exactly `pnl` after `costs`."""
    return Trade(
        symbol="SPY",
        module=module,
        direction=Direction.LONG,
        ts_entry=utc(2026, 1, 5, 14, 30),
        ts_exit=utc(2026, 1, 5, 15, 30),
        entry_price=100.0,
        exit_price=100.0 + pnl + costs,
        quantity=1.0,
        commission=costs,
    )


# ------------------------------------------------------------------------ metrics


def test_profit_factor_hand_computed():
    """Wins 10 + 20 = 30, losses 5 + 5 = 10 => PF 3.0, expectancy 20/4 = 5."""
    stats = TradeStats.from_trades([trade(10), trade(20), trade(-5), trade(-5)])
    assert stats.profit_factor == pytest.approx(3.0)
    assert stats.expectancy == pytest.approx(5.0)
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.net_pnl == pytest.approx(20.0)


def test_no_losses_reports_infinity_not_a_big_number():
    stats = TradeStats.from_trades([trade(10), trade(20)])
    assert stats.profit_factor == math.inf
    assert not stats.is_meaningful  # 2 trades


def test_sample_size_gate():
    small = TradeStats.from_trades([trade(1) for _ in range(MIN_MEANINGFUL_TRADES - 1)])
    big = TradeStats.from_trades([trade(1) for _ in range(MIN_MEANINGFUL_TRADES)])
    assert not small.is_meaningful
    assert big.is_meaningful


def test_edge_to_cost_ratio():
    """Gross 10 per trade against cost 2 per trade => 5.0."""
    stats = TradeStats.from_trades([trade(8, costs=2) for _ in range(10)])
    assert stats.gross_expectancy == pytest.approx(10.0)
    assert stats.cost_per_trade == pytest.approx(2.0)
    assert stats.edge_to_cost_ratio == pytest.approx(5.0)


def test_max_drawdown_hand_computed():
    """100 -> 120 -> 60: peak 120, trough 60 => 50% drawdown."""
    equity = [100.0, 120.0, 60.0, 90.0]
    worst, duration = max_drawdown(equity)
    assert worst == pytest.approx(0.5)
    assert duration == 2  # two consecutive underwater points
    assert drawdown_series(equity)[2] == pytest.approx(-0.5)


def test_period_returns():
    assert period_returns([100.0, 110.0, 99.0]) == pytest.approx([0.1, -0.1])


def test_sharpe_annualisation_uses_the_supplied_calendar():
    """Same return series, different calendar => ratio scales as sqrt(ppy)."""
    returns = [0.01, -0.005, 0.02, 0.0, -0.01, 0.015] * 10
    daily = sharpe_ratio(returns, periods_per_year=252)
    intraday = sharpe_ratio(returns, periods_per_year=19_656)
    assert intraday / daily == pytest.approx(math.sqrt(19_656 / 252), rel=1e-9)


def test_sharpe_of_a_constant_series_is_zero():
    assert sharpe_ratio([0.01] * 20, 252) == 0.0


def test_sortino_divides_by_all_periods_not_just_losers():
    """Guards the classic implementation error that inflates the ratio.

    Returns: three at +1%, one at -2%. Downside deviation over ALL four periods
    is sqrt(0.0004/4) = 0.01. Mean is 0.0025. So the unannualised ratio is 0.25.
    """
    returns = [0.01, 0.01, 0.01, -0.02]
    result = sortino_ratio(returns, periods_per_year=1.0)
    assert result == pytest.approx(0.25, rel=1e-9)


# ------------------------------------------------------------------ walk-forward


def test_rolling_windows_do_not_overlap_in_test():
    windows = rolling_windows(n_bars=1000, train_size=400, test_size=100)
    assert len(windows) == 6
    for a, b in zip(windows, windows[1:]):
        assert a.test_end <= b.test_start


def test_embargo_separates_train_from_test():
    """The seam is where in-sample leaks into out-of-sample."""
    windows = rolling_windows(n_bars=1000, train_size=400, test_size=100, embargo=50)
    for w in windows:
        assert w.test_start - w.train_end == 50
        assert w.embargo_size == 50


def test_anchored_windows_grow_the_training_set():
    rolling = rolling_windows(1000, 300, 100)
    anchored = rolling_windows(1000, 300, 100, anchored=True)
    assert all(w.train_start == 0 for w in anchored)
    assert anchored[-1].train_size > rolling[-1].train_size


def test_no_window_is_emitted_past_the_end_of_data():
    windows = rolling_windows(n_bars=450, train_size=400, test_size=100)
    assert windows == []


def test_run_walk_forward_never_shows_test_bars_to_fit():
    """Structural guarantee: `fit` only ever receives the training slice."""
    bars = list(range(1000))  # opaque objects are fine, nothing indexes into them
    windows = rolling_windows(1000, 400, 100, embargo=10)
    seen_by_fit: list[tuple[int, int]] = []

    def fit(train):
        seen_by_fit.append((train[0], train[-1]))

    def run_slice(slice_):
        return [trade(1.0) for _ in range(5)]

    result = run_walk_forward(bars, windows, run_slice, fit)

    for window, (first, last) in zip(windows, seen_by_fit):
        assert first == window.train_start
        assert last == window.train_end - 1
        assert last < window.test_start  # never touches test bars


def test_degradation_ratio():
    windows = rolling_windows(1000, 400, 100)
    is_stats = [TradeStats.from_trades([trade(20), trade(-5)]) for _ in windows]
    oos_stats = [TradeStats.from_trades([trade(10), trade(-5)]) for _ in windows]
    result = WalkForwardResult(
        windows=list(windows),
        in_sample=is_stats,
        out_of_sample=oos_stats,
        oos_trades=[trade(10), trade(-5)] * len(windows),
    )
    assert result.combined_is.profit_factor == pytest.approx(4.0)
    assert result.combined_oos.profit_factor == pytest.approx(2.0)
    assert result.degradation == pytest.approx(0.5)
    assert result.consistent_windows == len(windows)


# ----------------------------------------------------------------------- grading


def _sweep(gross_per_trade: float, cost_per_trade_at_1x: float, n: int = 60):
    """Build a run_at_multiple callable with a known linear cost structure."""

    def run_at_multiple(multiple: float):
        cost = cost_per_trade_at_1x * multiple
        out = []
        for i in range(n):
            gross = gross_per_trade if i % 2 == 0 else -gross_per_trade * 0.5
            out.append(trade(gross - cost, costs=cost))
        return out

    return run_at_multiple


def test_cost_sensitivity_finds_the_breakeven_multiple():
    sens = cost_sensitivity(_sweep(10.0, 2.0), multiples=(0.0, 1.0, 2.0, 3.0))
    assert sens.stats[0].profit_factor > sens.stats[-1].profit_factor
    be = sens.breakeven_multiple
    assert 0.0 < be < math.inf


def test_strategy_that_only_works_gross_is_graded_fail():
    """The exact failure this system exists to surface.

    30 winners and 30 losers. Gross: +5 / -3 => PF 1.67, a real signal.
    Costs of 2 per trade turn that into +3 / -5 => PF 0.60. This is the
    PF 0.75-0.85 wall, and the grade must be FAIL rather than 'promising'.
    """
    gross = TradeStats.from_trades(
        [trade(5.0) if i % 2 == 0 else trade(-3.0) for i in range(60)]
    )
    net = TradeStats.from_trades(
        [trade(3.0, costs=2.0) if i % 2 == 0 else trade(-5.0, costs=2.0) for i in range(60)]
    )
    assert gross.profit_factor == pytest.approx(5 / 3)
    assert net.profit_factor == pytest.approx(0.6)

    verdict = grade_strategy("costs eat it", net=net, gross=gross)
    assert verdict.grade is Grade.FAIL
    assert not verdict.may_paper_trade
    assert any("below 1.0" in r for r in verdict.reasons)
    # And it must say WHERE the problem is, not just that there is one.
    assert any("costs are eating it" in r for r in verdict.reasons)


def test_small_sample_is_graded_insufficient_not_pass():
    net = TradeStats.from_trades([trade(50.0) for _ in range(5)])
    verdict = grade_strategy("tiny", net=net)
    assert verdict.grade is Grade.INSUFFICIENT_DATA
    assert not verdict.may_paper_trade


def test_thin_edge_over_costs_is_marginal_not_pass():
    """PF above 1 but edge only ~1.3x costs: passes the naive test, fails this one."""
    net = TradeStats.from_trades(
        [trade(4.0, costs=3.0) if i % 4 else trade(-3.0, costs=3.0) for i in range(60)]
    )
    verdict = grade_strategy("thin", net=net)
    assert net.profit_factor > 1.0
    assert verdict.grade is Grade.MARGINAL
    assert not verdict.may_paper_trade  # MARGINAL does not graduate


def test_verdict_renders_the_reasons():
    net = TradeStats.from_trades([trade(10.0, costs=1.0) for _ in range(60)])
    verdict = grade_strategy("clean", net=net, gross=net)
    text = verdict.render()
    assert "VERDICT" in text and "gross vs net" in text


def test_ruin_is_flagged_rather_than_reported_as_a_percentage():
    """Once equity crosses zero a drawdown percentage stops meaning anything.

    Equity 100 -> 200 -> -190 gives (value-peak)/peak = -1.95, which renders as
    a "195% drawdown" - a number that reads as bad-but-survivable and is in fact
    an account that no longer exists. Real accounts get closed out first, so
    every statistic past that point describes a fiction.
    """
    healthy = equity_stats([100.0, 120.0, 90.0, 110.0], periods_per_year=252)
    assert healthy.ruined is False
    assert healthy.min_equity == pytest.approx(90.0)
    assert 0 < healthy.max_drawdown < 1.0

    blown = equity_stats([100.0, 200.0, -190.0], periods_per_year=252)
    assert blown.ruined is True
    assert blown.min_equity == pytest.approx(-190.0)
    assert blown.max_drawdown > 1.0  # the raw figure is still computed...
    # ...but `ruined` is what a report must lead with.


def test_exactly_zero_equity_counts_as_ruin():
    """A zero balance is a closed account, not a 100% drawdown you trade out of."""
    assert equity_stats([100.0, 50.0, 0.0], periods_per_year=252).ruined is True
