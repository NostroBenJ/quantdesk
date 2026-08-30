"""Plain dataclasses that every layer speaks. No behaviour beyond validation.

These are the interface between modules. Keeping them dumb and frozen means a
signal module physically cannot mutate the market state it was handed, which is
one fewer way for the future to leak backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from .clock import ensure_utc


class Direction(Enum):
    """Which way a signal points. Value is the sign, so it multiplies cleanly."""

    LONG = 1
    FLAT = 0
    SHORT = -1

    def __int__(self) -> int:
        return self.value


class Side(Enum):
    BUY = 1
    SELL = -1

    def __int__(self) -> int:
        return self.value


class BarCompleteness(Enum):
    """Whether a bar covers its full time window.

    PARTIAL is the dangerous one: the currently-forming bar. Its close is not the
    period's close, it is 'the price right now'. Feeding a partial bar to a
    strategy that expects closes means the strategy sees a price before the
    period that produced it has finished - lookahead, in the most literal sense.

    Vendors do not label this. Yahoo will happily hand you a partial final bar
    with no flag on it. `data/bars.py` infers the label; the engine drops them.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class AssetClass(Enum):
    EQUITY = "equity"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"
    INDEX = "index"


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Contract mechanics. Every number that turns price into money lives here.

    Defaults are for a 1x-multiplier cash equity. For MNQ you want
    multiplier=2.0, tick_size=0.25 (so one tick = $0.50).
    """

    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    multiplier: float = 1.0
    tick_size: float = 0.01
    currency: str = "USD"
    # ASSUMPTION (broker behaviour): quantities are integral for futures and
    # options; fractional shares exist for equities at some brokers but we
    # disallow them by default because prop-firm platforms do not offer them.
    allow_fractional: bool = False

    @property
    def tick_value(self) -> float:
        """Cash value of one tick for one contract."""
        return self.tick_size * self.multiplier

    def round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar, normalised.

    `ts_open` identifies the bar. `ts_close` is when it became knowable, and is
    the only timestamp the engine compares against 'now'. See core/clock.py.
    """

    symbol: str
    ts_open: datetime
    ts_close: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    completeness: BarCompleteness = BarCompleteness.UNKNOWN

    def __post_init__(self) -> None:
        if self.ts_close <= self.ts_open:
            raise ValueError(
                f"{self.symbol}: ts_close {self.ts_close} must be after ts_open {self.ts_open}"
            )
        lo, hi = self.low, self.high
        if hi < lo:
            raise ValueError(f"{self.symbol} @ {self.ts_open}: high {hi} < low {lo}")
        for name, price in (("open", self.open), ("close", self.close)):
            if not (lo <= price <= hi):
                raise ValueError(
                    f"{self.symbol} @ {self.ts_open}: {name} {price} outside [{lo}, {hi}]"
                )
        if self.volume < 0:
            raise ValueError(f"{self.symbol} @ {self.ts_open}: negative volume")

    @property
    def is_usable(self) -> bool:
        """A bar the engine is willing to compute features from."""
        return self.completeness is BarCompleteness.COMPLETE

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0

    @property
    def true_range(self) -> float:
        """Intra-bar range only. The gap-aware version needs the previous close
        and lives in data/bars.py, because it is a series operation."""
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class Signal:
    """What a signal module emits. The common interface across all alpha modules.

    `confidence` is in [0, 1] and is NOT a probability unless the module says so
    in `reasoning`; it is a relative conviction used by the risk layer for
    scaling. `reasoning` is free text for the narrative layer and the journal -
    it never feeds a calculation.
    """

    module: str
    ts: datetime
    symbol: str
    direction: Direction
    confidence: float
    reasoning: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"{self.module}: confidence {self.confidence} outside [0, 1]"
            )

    @property
    def is_actionable(self) -> bool:
        return self.direction is not Direction.FLAT and self.confidence > 0.0


@dataclass(frozen=True, slots=True)
class Order:
    """An intent to trade, emitted at `ts` - which is always a bar close."""

    ts: datetime
    symbol: str
    side: Side
    quantity: float
    reason: str = ""
    # Price the decision was made at. NOT the expected fill price - recording it
    # separately is what lets the report show decision-to-fill slippage honestly.
    reference_price: float | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")


@dataclass(frozen=True, slots=True)
class Fill:
    """A (possibly partial) execution."""

    ts: datetime
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    # Quantity the order asked for, so partial fills are visible downstream.
    requested_quantity: float = 0.0

    @property
    def is_partial(self) -> bool:
        return self.requested_quantity > 0 and self.quantity < self.requested_quantity

    @property
    def notional(self) -> float:
        return self.price * self.quantity

    @property
    def total_cost(self) -> float:
        """All-in friction in cash for this fill."""
        return self.commission + self.slippage


@dataclass(frozen=True, slots=True)
class Trade:
    """A round trip: entry to exit. The unit of performance measurement."""

    symbol: str
    module: str
    direction: Direction
    ts_entry: datetime
    ts_exit: datetime
    entry_price: float
    exit_price: float
    quantity: float
    multiplier: float = 1.0
    commission: float = 0.0
    slippage: float = 0.0
    reasoning: str = ""

    @property
    def gross_pnl(self) -> float:
        """P&L before any friction. The number that flatters you."""
        move = (self.exit_price - self.entry_price) * int(self.direction)
        return move * self.quantity * self.multiplier

    @property
    def costs(self) -> float:
        return self.commission + self.slippage

    @property
    def net_pnl(self) -> float:
        """P&L after friction. The only number that pays rent."""
        return self.gross_pnl - self.costs

    @property
    def holding_period(self) -> float:
        """Seconds held."""
        return (self.ts_exit - self.ts_entry).total_seconds()


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    multiplier: float = 1.0

    @property
    def direction(self) -> Direction:
        if self.quantity > 0:
            return Direction.LONG
        if self.quantity < 0:
            return Direction.SHORT
        return Direction.FLAT

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0

    def unrealised_pnl(self, mark: float) -> float:
        return (mark - self.average_price) * self.quantity * self.multiplier

    def exposure(self, mark: float) -> float:
        """Signed notional at `mark`."""
        return mark * self.quantity * self.multiplier


@dataclass(frozen=True, slots=True)
class MarketState:
    """Everything a signal module is allowed to see at one instant.

    Construct these only via `backtest.view.MarketView`, which guarantees that
    `history` contains no bar whose ts_close is after `ts`. Building one by hand
    bypasses the causality guarantee.
    """

    ts: datetime
    symbol: str
    history: tuple[Bar, ...]
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_utc(self.ts)

    @property
    def last(self) -> Bar:
        if not self.history:
            raise ValueError(f"{self.symbol}: no history available at {self.ts}")
        return self.history[-1]

    @property
    def closes(self) -> tuple[float, ...]:
        return tuple(b.close for b in self.history)

    def __len__(self) -> int:
        return len(self.history)
