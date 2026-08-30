"""End-to-end engine behaviour, and the structured event log.

The first two tests are the ones that justify the whole design: a module is
proven unable to see the bar it is about to trade in, and a fill is proven to
happen at the next bar's open rather than at the price that triggered it.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from conftest import make_bars
from quantdesk.core.clock import utc
from quantdesk.core.types import Bar, BarCompleteness, Direction, MarketState, Signal
from quantdesk.backtest.costs import (
    CommissionSchedule,
    SlippageParams,
    VolatilityScaledCostModel,
    ZeroCostModel,
)
from quantdesk.backtest.engine import (
    BacktestEngine,
    EngineConfig,
    FixedSizer,
    VolatilityTargetSizer,
)
from quantdesk.backtest.report import cost_sensitivity, grade_strategy
from quantdesk.backtest.view import MarketView
from quantdesk.obs.logging import EventLog
from quantdesk.signals.base import ModuleLedger, SignalModule


class AlwaysLong(SignalModule):
    warmup = 5

    def generate_signal(self, market_state: MarketState) -> Signal:
        return Signal(
            module=self.name,
            ts=market_state.ts,
            symbol=market_state.symbol,
            direction=Direction.LONG,
            confidence=1.0,
            reasoning="unconditionally long (engine test fixture)",
        )


class AlwaysFlat(SignalModule):
    def generate_signal(self, market_state: MarketState) -> Signal:
        return self.flat(market_state)


class Alternating(SignalModule):
    """Flips every `period` bars, to generate round trips."""

    warmup = 5

    def __init__(self, period: int = 10) -> None:
        self.period = period
        self._calls = 0

    def reset(self) -> None:
        self._calls = 0

    def generate_signal(self, market_state: MarketState) -> Signal:
        self._calls += 1
        direction = (
            Direction.LONG if (self._calls // self.period) % 2 == 0 else Direction.SHORT
        )
        return Signal(
            module=self.name,
            ts=market_state.ts,
            symbol=market_state.symbol,
            direction=direction,
            confidence=1.0,
            reasoning=f"alternating, call {self._calls}",
        )


class Spy(SignalModule):
    """Records the last bar it was shown, so we can prove what it could see."""

    warmup = 5

    def __init__(self) -> None:
        self.seen: list[tuple] = []

    def reset(self) -> None:
        self.seen = []

    def generate_signal(self, market_state: MarketState) -> Signal:
        self.seen.append((market_state.ts, market_state.last.ts_open, len(market_state)))
        return self.flat(market_state)


# ------------------------------------------------------------------- causality


def test_module_never_sees_the_bar_it_will_trade_in(spy, bars_trending, tf_5m):
    """At every decision point, the newest visible bar is the one that just closed."""
    engine = BacktestEngine(spy, tf_5m, cost_model=ZeroCostModel())
    module = Spy()
    view = MarketView("SPY", bars_trending)
    engine.run(view, module)

    assert module.seen
    by_open = {b.ts_open: i for i, b in enumerate(bars_trending)}
    for decision_ts, newest_open, count in module.seen:
        index = by_open[newest_open]
        # The newest visible bar closed exactly at the decision instant...
        assert bars_trending[index].ts_close == decision_ts
        # ...and everything after it is strictly in the future. (The final
        # decision point has no successor - that order expires unfilled.)
        if index + 1 < len(bars_trending):
            assert bars_trending[index + 1].ts_open >= decision_ts
        assert count == index + 1

    # And the module was called on the final bar too, where nothing can fill.
    assert module.seen[-1][1] == bars_trending[-1].ts_open


def test_entry_fills_at_the_next_bar_open(spy, tf_5m):
    """With zero costs the fill price must equal the execution bar's open exactly."""
    bars = make_bars(40, tf_5m, drift=0.001)
    engine = BacktestEngine(
        spy, tf_5m, cost_model=ZeroCostModel(), sizer=FixedSizer(1.0),
        config=EngineConfig(volatility_lookback=10),
    )
    result = engine.run(MarketView("SPY", bars), AlwaysLong())

    assert result.trades
    first = result.trades[0]
    warmup = max(AlwaysLong.warmup, 10 + 1)
    assert first.entry_price == pytest.approx(bars[warmup + 1].open)
    assert first.ts_entry == bars[warmup + 1].ts_open


def test_final_bar_order_expires_rather_than_filling(spy, tf_5m):
    """No future bar to trade in means no fill - not a free fill at the close."""
    bars = make_bars(30, tf_5m)
    engine = BacktestEngine(
        spy, tf_5m, cost_model=ZeroCostModel(), sizer=FixedSizer(1.0),
        config=EngineConfig(volatility_lookback=5, flatten_at_end=False),
    )
    # Alternating with a short period guarantees a flip on the final bar.
    result = engine.run(MarketView("SPY", bars), Alternating(period=1))
    assert result.orders_expired >= 1
    assert result.orders_submitted > result.orders_filled


# --------------------------------------------------------------------- mechanics


def test_flat_module_trades_nothing(spy, bars_flat, tf_5m):
    engine = BacktestEngine(spy, tf_5m, cost_model=ZeroCostModel())
    result = engine.run(MarketView("SPY", bars_flat), AlwaysFlat())
    assert result.orders_submitted == 0
    assert result.trades == []
    assert result.equity_curve  # equity is still marked every bar
    assert all(v == pytest.approx(100_000.0) for _, v in result.equity_curve)


def test_equity_is_marked_every_bar_not_only_at_exits(spy, tf_5m):
    """Drawdown must be measured on the path actually experienced."""
    bars = make_bars(60, tf_5m, drift=0.001)
    engine = BacktestEngine(
        spy, tf_5m, cost_model=ZeroCostModel(), sizer=FixedSizer(1.0),
        config=EngineConfig(volatility_lookback=10),
    )
    result = engine.run(MarketView("SPY", bars), AlwaysLong())
    warmup = max(AlwaysLong.warmup, 11)
    # One equity point per bar from warmup, plus one for the forced liquidation.
    assert len(result.equity_curve) == (len(bars) - warmup) + 1
    assert result.equity_stats().n_periods == len(result.equity_curve)


def test_forced_liquidation_is_flagged_loudly(spy, tf_5m):
    bars = make_bars(40, tf_5m, drift=0.001)
    engine = BacktestEngine(
        spy, tf_5m, cost_model=ZeroCostModel(), sizer=FixedSizer(1.0),
        config=EngineConfig(volatility_lookback=10, flatten_at_end=True),
    )
    result = engine.run(MarketView("SPY", bars), AlwaysLong())
    assert result.forced_liquidation
    assert any("force-closed" in w for w in result.warnings)


def test_round_trips_are_recorded_per_module(spy, tf_5m):
    bars = make_bars(120, tf_5m, drift=0.0002, wiggle=0.001)
    engine = BacktestEngine(
        spy, tf_5m, cost_model=ZeroCostModel(), sizer=FixedSizer(1.0),
        config=EngineConfig(volatility_lookback=10),
    )
    result = engine.run(MarketView("SPY", bars), Alternating(period=10))
    assert len(result.trades) >= 2
    stats = result.ledger.stats()
    assert "Alternating" in stats
    assert stats["Alternating"].n == len(result.trades)
    assert result.ledger.report_lines()


def test_costs_reduce_net_pnl_relative_to_gross(spy, tf_5m):
    """Same signal, same bars: the only difference is friction."""
    bars = make_bars(120, tf_5m, drift=0.0002, wiggle=0.001)
    view = MarketView("SPY", bars)
    config = EngineConfig(volatility_lookback=10)

    free = BacktestEngine(
        spy, tf_5m, cost_model=ZeroCostModel(), sizer=FixedSizer(1.0), config=config
    ).run(view, Alternating(period=10))
    costed = BacktestEngine(
        spy,
        tf_5m,
        cost_model=VolatilityScaledCostModel(CommissionSchedule(per_unit=0.01)),
        sizer=FixedSizer(1.0),
        config=config,
    ).run(view, Alternating(period=10))

    assert costed.trade_stats().total_costs > 0
    assert costed.trade_stats().net_pnl < free.trade_stats().net_pnl


def test_volatility_target_sizer_shrinks_size_as_vol_rises(spy, tf_5m):
    calm = make_bars(80, tf_5m, wiggle=0.0005)
    wild = make_bars(80, tf_5m, wiggle=0.02)
    config = EngineConfig(volatility_lookback=10)
    sizer = VolatilityTargetSizer(risk_fraction=0.02)

    sizes = []
    for bars in (calm, wild):
        engine = BacktestEngine(
            spy, tf_5m, cost_model=ZeroCostModel(), sizer=sizer, config=config
        )
        result = engine.run(MarketView("SPY", bars), AlwaysLong())
        sizes.append(result.trades[0].quantity if result.trades else 0.0)

    assert sizes[0] > sizes[1], "higher volatility must produce a smaller position"


def test_engine_refuses_to_trade_without_enough_warmup(spy, tf_5m):
    bars = make_bars(8, tf_5m)
    engine = BacktestEngine(spy, tf_5m, config=EngineConfig(volatility_lookback=20))
    result = engine.run(MarketView("SPY", bars), AlwaysLong())
    assert result.trades == []
    assert any("warmup" in w for w in result.warnings)


# ------------------------------------------------------- full honesty pipeline


def test_full_pipeline_produces_a_graded_verdict(spy, tf_5m):
    """Backtest -> cost sweep -> verdict, the way it is meant to be run."""
    bars = make_bars(400, tf_5m, drift=0.0001, wiggle=0.0015)
    view = MarketView("SPY", bars)
    config = EngineConfig(volatility_lookback=20)

    def run_at_multiple(multiple: float):
        model = VolatilityScaledCostModel(
            CommissionSchedule(per_unit=0.005),
            SlippageParams(),
            multiple=multiple,
        )
        engine = BacktestEngine(
            spy, tf_5m, cost_model=model, sizer=FixedSizer(10.0), config=config
        )
        return engine.run(view, Alternating(period=8)).trades

    sens = cost_sensitivity(run_at_multiple, multiples=(0.0, 1.0, 2.0))
    gross = sens.stats[0]
    net = sens.stats[1]

    # Costs must strictly reduce measured performance.
    assert net.total_costs > gross.total_costs
    assert net.net_pnl < gross.net_pnl

    verdict = grade_strategy("alternating fixture", net=net, gross=gross, sensitivity=sens)
    assert verdict.grade.value in {"PASS", "MARGINAL", "FAIL", "INSUFFICIENT DATA"}
    assert verdict.reasons
    assert "VERDICT" in verdict.render()


# ------------------------------------------------------------------ event log


def test_event_log_writes_one_json_object_per_line(tmp_path):
    log = EventLog(tmp_path, run_id="test-run")
    ts = utc(2026, 1, 5, 14, 30)
    log.emit("signal", ts=ts, symbol="SPY", direction=Direction.LONG, confidence=0.8)
    log.skip(ts, "daily loss limit reached", symbol="SPY")
    log.risk_check(ts, "max_concurrent_positions", passed=False, limit=3, current=3)

    path = tmp_path / "2026-01-05.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert records[0]["event"] == "signal"
    assert records[0]["direction"] == 1  # enum serialised by value
    assert records[1]["event"] == "skip"
    assert records[2]["passed"] is False
    assert all(r["run_id"] == "test-run" for r in records)


def test_event_log_handles_non_finite_floats(tmp_path):
    """An infinite profit factor must not produce a file other parsers reject."""
    log = EventLog(tmp_path)
    log.emit("stats", ts=utc(2026, 1, 5), profit_factor=float("inf"))
    record = log.read_day(date(2026, 1, 5))[0]
    assert record["profit_factor"] == {"__float__": "inf"}


def test_event_log_reads_a_day_back(tmp_path):
    log = EventLog(tmp_path)
    for i in range(5):
        log.emit("signal", ts=utc(2026, 1, 5, 14, 30) + timedelta(minutes=i), n=i)
    assert len(log.read_day(date(2026, 1, 5))) == 5
    assert log.read_day(date(2026, 1, 6)) == []
