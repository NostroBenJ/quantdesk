"""The event loop. Strictly causal by construction.

THE TIMELINE, WHICH IS THE WHOLE POINT
--------------------------------------
For each bar i:

    t = bars[i].ts_close        <- the instant bar i became knowable
    state = view.state(t)       <- contains bars 0..i, nothing after
    signal = module.generate_signal(state)
    order  = size(signal)       <- stamped at t
    fill   = fill_model(order, bars[i+1])   <- executes INSIDE the next bar

A decision made from bar i's close can only ever be executed in bar i+1. There is
no code path that fills at bars[i].close, because the fill model is only ever
handed `bars[i+1]`. The last bar of the series therefore produces no fill: its
orders are counted as `orders_expired`, not filled at the close.

Equity is marked at every bar close, so drawdown is measured on the path actually
experienced rather than only at trade exits - which is the difference between a
drawdown figure that would have breached a prop-firm limit and one that would not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from ..core.clock import Timeframe
from ..core.types import (
    Bar,
    Direction,
    Fill,
    InstrumentSpec,
    Order,
    Side,
    Signal,
    Trade,
)
from ..data.bars import realized_volatility
from ..signals.base import ModuleLedger, SignalModule
from .costs import CostModel, VolatilityScaledCostModel
from .fills import FillModel, NextBarOpenFill
from .metrics import EquityStats, TradeStats, equity_stats
from .view import MarketView


class Sizer(Protocol):
    """How a signal becomes a quantity. The risk layer will replace this."""

    def quantity(self, signal: Signal, bar: Bar, equity: float, bar_volatility: float) -> float:
        ...


@dataclass(frozen=True, slots=True)
class FixedSizer:
    """Constant quantity, scaled by signal confidence. For engine testing only.

    Fixed size is not a risk policy - it makes every trade a different amount of
    risk depending on where volatility happens to be. Real sizing is
    `VolatilityTargetSizer`, and the eventual risk layer owns it.
    """

    quantity_per_unit_confidence: float = 1.0

    def quantity(self, signal: Signal, bar: Bar, equity: float, bar_volatility: float) -> float:
        return self.quantity_per_unit_confidence * signal.confidence


@dataclass(frozen=True, slots=True)
class VolatilityTargetSizer:
    """Size so each position carries the same expected risk.

        quantity = (equity * risk_fraction) / (k * sigma_bar * price * multiplier)

    `sigma_bar` is per-bar, not annualised, so the risk budget is per-bar too.
    Mixing the two is a factor-of-sqrt(periods_per_year) error, which at 5-minute
    bars is ~140x - the kind of mistake that shows up as an instantly blown
    account rather than as a slightly wrong number.
    """

    risk_fraction: float = 0.01
    stop_multiple: float = 2.0
    max_quantity: float = 1e9

    def quantity(self, signal: Signal, bar: Bar, equity: float, bar_volatility: float) -> float:
        if bar_volatility <= 0 or bar.close <= 0:
            return 0.0
        risk_cash = equity * self.risk_fraction * signal.confidence
        risk_per_unit = self.stop_multiple * bar_volatility * bar.close
        if risk_per_unit <= 0:
            return 0.0
        return min(risk_cash / risk_per_unit, self.max_quantity)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    initial_equity: float = 100_000.0
    #: Bars of history used to estimate the per-bar volatility that drives both
    #: sizing and slippage.
    volatility_lookback: int = 20
    #: How much history each signal module sees. None = everything to date.
    signal_lookback: int | None = 200
    #: Close any open position at the final bar. Reported separately so a result
    #: that depends on it is visible.
    flatten_at_end: bool = True


@dataclass
class BacktestResult:
    symbol: str
    module: str
    timeframe: Timeframe
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_partially_filled: int = 0
    orders_unfilled: int = 0
    orders_expired: int = 0
    forced_liquidation: bool = False
    warnings: list[str] = field(default_factory=list)
    ledger: ModuleLedger = field(default_factory=ModuleLedger)

    @property
    def equity(self) -> list[float]:
        return [value for _, value in self.equity_curve]

    def trade_stats(self) -> TradeStats:
        return TradeStats.from_trades(self.trades)

    def equity_stats(self) -> EquityStats:
        return equity_stats(self.equity, self.timeframe.periods_per_year)

    @property
    def fill_rate(self) -> float:
        return self.orders_filled / self.orders_submitted if self.orders_submitted else 0.0


class BacktestEngine:
    """Single-symbol, single-module backtest. Composition happens a layer up."""

    def __init__(
        self,
        instrument: InstrumentSpec,
        timeframe: Timeframe,
        cost_model: CostModel | None = None,
        fill_model: FillModel | None = None,
        sizer: Sizer | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.instrument = instrument
        self.timeframe = timeframe
        self.cost_model = cost_model or VolatilityScaledCostModel()
        self.fill_model = fill_model or NextBarOpenFill()
        self.sizer = sizer or FixedSizer()
        self.config = config or EngineConfig()

    def run(self, view: MarketView, module: SignalModule) -> BacktestResult:
        cfg = self.config
        result = BacktestResult(
            symbol=view.symbol, module=module.name, timeframe=self.timeframe
        )
        bars = view.bars
        warmup = max(module.warmup, cfg.volatility_lookback + 1)
        if len(bars) <= warmup + 1:
            result.warnings.append(
                f"only {len(bars)} bars for a warmup of {warmup}: nothing was traded"
            )
            return result

        module.reset()
        cash = cfg.initial_equity
        quantity = 0.0
        avg_price = 0.0
        entry_ts: datetime | None = None
        entry_costs = 0.0
        entry_reason = ""

        for i in range(warmup, len(bars)):
            bar = bars[i]
            ts = bar.ts_close
            equity = cash + quantity * bar.close * self.instrument.multiplier
            result.equity_curve.append((ts, equity))

            sigma = self._bar_volatility(bars, i)
            state = view.state(ts, cfg.signal_lookback)
            signal = module.generate_signal(state)
            result.signals.append(signal)

            target = self._target_quantity(signal, bar, equity, sigma)
            delta = target - quantity
            if abs(delta) < self._min_increment():
                continue

            result.orders_submitted += 1
            exec_bar = bars[i + 1] if i + 1 < len(bars) else None
            if exec_bar is None:
                # Decision made on the final bar. There is no future bar to trade
                # in, so it expires. Filling it at this bar's close would be a
                # free trade at a price the decision itself was derived from.
                result.orders_expired += 1
                continue

            order = Order(
                ts=ts,
                symbol=view.symbol,
                side=Side.BUY if delta > 0 else Side.SELL,
                quantity=abs(delta),
                reason=signal.reasoning,
                reference_price=bar.close,
            )
            fill = self.fill_model.fill(
                order, exec_bar, self.instrument, self.cost_model, sigma
            )
            if fill is None:
                result.orders_unfilled += 1
                continue
            result.orders_filled += 1
            if fill.is_partial:
                result.orders_partially_filled += 1

            signed = fill.quantity * int(fill.side)
            cash -= signed * fill.price * self.instrument.multiplier
            cash -= fill.total_cost

            closing = quantity != 0 and (
                (quantity > 0 > signed) or (quantity < 0 < signed)
            )
            if closing:
                closed_qty = min(abs(quantity), abs(signed))
                trade = Trade(
                    symbol=view.symbol,
                    module=module.name,
                    direction=Direction.LONG if quantity > 0 else Direction.SHORT,
                    ts_entry=entry_ts or fill.ts,
                    ts_exit=fill.ts,
                    entry_price=avg_price,
                    exit_price=fill.price,
                    quantity=closed_qty,
                    multiplier=self.instrument.multiplier,
                    commission=entry_costs + fill.commission,
                    slippage=fill.slippage,
                    reasoning=entry_reason or signal.reasoning,
                )
                result.trades.append(trade)
                result.ledger.record(trade)
                entry_costs = 0.0

            new_quantity = quantity + signed
            if quantity == 0 or (quantity > 0) == (signed > 0):
                # Opening or adding: weighted-average the entry price.
                total = abs(quantity) + abs(signed)
                avg_price = (
                    (avg_price * abs(quantity) + fill.price * abs(signed)) / total
                    if total
                    else fill.price
                )
                if quantity == 0:
                    entry_ts = fill.ts
                    entry_reason = signal.reasoning
                entry_costs += fill.total_cost
            elif new_quantity != 0 and (new_quantity > 0) != (quantity > 0):
                # Reversed through flat: the remainder is a fresh position.
                avg_price = fill.price
                entry_ts = fill.ts
                entry_reason = signal.reasoning
                entry_costs = fill.total_cost
            quantity = new_quantity
            if quantity == 0:
                entry_ts = None
                entry_reason = ""

        if quantity != 0 and cfg.flatten_at_end:
            last = bars[-1]
            result.forced_liquidation = True
            result.warnings.append(
                f"position of {quantity:g} force-closed at the final bar's close "
                f"({last.ts_close.isoformat()}). This is a modelling convenience, not a "
                "fill you would have got - treat the last trade's P&L as provisional."
            )
            trade = Trade(
                symbol=view.symbol,
                module=module.name,
                direction=Direction.LONG if quantity > 0 else Direction.SHORT,
                ts_entry=entry_ts or last.ts_close,
                ts_exit=last.ts_close,
                entry_price=avg_price,
                exit_price=last.close,
                quantity=abs(quantity),
                multiplier=self.instrument.multiplier,
                commission=entry_costs,
                slippage=0.0,
                reasoning=entry_reason + " | forced liquidation at end of data",
            )
            result.trades.append(trade)
            result.ledger.record(trade)
            cash += quantity * last.close * self.instrument.multiplier
            quantity = 0.0
            result.equity_curve.append((last.ts_close, cash))

        return result

    def _target_quantity(
        self, signal: Signal, bar: Bar, equity: float, sigma: float
    ) -> float:
        if not signal.is_actionable:
            return 0.0
        size = self.sizer.quantity(signal, bar, equity, sigma)
        if not self.instrument.allow_fractional:
            size = float(int(size))
        return size * int(signal.direction)

    def _min_increment(self) -> float:
        return 1e-9 if self.instrument.allow_fractional else 1.0

    def _bar_volatility(self, bars: Sequence[Bar], index: int) -> float:
        """Per-bar (NOT annualised) volatility from the trailing window.

        Uses bars[index - lookback : index + 1], all of which have closed at
        bars[index].ts_close. Nothing here can see past `index` - which is what
        `tests/test_lookahead.py` verifies by perturbing the future.
        """
        lookback = self.config.volatility_lookback
        window = bars[max(0, index - lookback) : index + 1]
        if len(window) < 3:
            return 0.0
        return realized_volatility(window, self.timeframe, annualise=False)
