"""Fill simulation. Nothing ever fills at the price that generated the signal.

Two models:

* `NextBarOpenFill` - the honest default for bar-driven strategies. A decision
  made at bar N's close executes at bar N+1's open, plus slippage, capped by a
  participation share of that bar's volume. Size above the cap fills partially.

* `ExponentialQueueFill` - passive/limit orders, using the fill intensity from
  Lucic & Tse (2024): arrival rate decays as `lambda(d) = lambda_0 * exp(-kappa*d)`
  in the distance `d` from mid. Quote closer, fill more.

DETERMINISM
-----------
`ExponentialQueueFill` converts that intensity into a filled FRACTION rather than
sampling a random yes/no. That is deliberate. Random fills make two runs of the
same backtest disagree, and once results are noisy it becomes impossible to tell
a parameter change from a seed change - which is precisely the confusion that
makes walk-forward results unfalsifiable. Pass an explicit `rng` if you want
stochastic fills for a robustness study, and then run many paths.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from ..core.types import Bar, Fill, InstrumentSpec, Order, Side
from .costs import CostModel


class FillModel(ABC):
    @abstractmethod
    def fill(
        self,
        order: Order,
        bar: Bar,
        instrument: InstrumentSpec,
        cost_model: CostModel,
        bar_volatility: float,
        crosses_spread: bool = True,
    ) -> Fill | None:
        """Return the resulting Fill, or None if the order could not execute."""


def _quantise(quantity: float, instrument: InstrumentSpec) -> float:
    if instrument.allow_fractional:
        return quantity
    return float(int(quantity))


class NextBarOpenFill(FillModel):
    """Fill at the next bar's open, adjusted for slippage, capped by volume."""

    def __init__(self, max_participation: float | None = None) -> None:
        #: None means "take the cap from the cost model's SlippageParams", so the
        #: two cannot drift apart.
        self.max_participation = max_participation

    def _cap(self, cost_model: CostModel) -> float:
        if self.max_participation is not None:
            return self.max_participation
        params = getattr(cost_model, "params", None)
        return getattr(params, "max_participation", 1.0)

    def fill(
        self,
        order: Order,
        bar: Bar,
        instrument: InstrumentSpec,
        cost_model: CostModel,
        bar_volatility: float,
        crosses_spread: bool = True,
    ) -> Fill | None:
        cap_fraction = self._cap(cost_model)
        quantity = order.quantity
        if bar.volume > 0 and cap_fraction < 1.0:
            allowed = _quantise(cap_fraction * bar.volume, instrument)
            quantity = min(quantity, allowed)
        quantity = _quantise(quantity, instrument)
        if quantity <= 0:
            return None

        per_unit = cost_model.slippage_per_unit(
            instrument, quantity, bar, bar_volatility, crosses_spread
        )
        # Slippage always works against you: buys pay up, sells get hit down.
        raw_price = bar.open + int(order.side) * per_unit
        # A fill cannot happen outside the bar's own range. Without this clamp a
        # large slippage estimate can produce a price nobody traded at, which in
        # a short position can silently manufacture profit.
        price = min(max(raw_price, bar.low), bar.high)

        commission = cost_model.commission(instrument, quantity, price)
        slippage_cash = abs(price - bar.open) * quantity * instrument.multiplier

        return Fill(
            ts=bar.ts_open,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage_cash,
            requested_quantity=order.quantity,
        )


class ExponentialQueueFill(FillModel):
    """Passive limit-order fills with exponentially decaying fill probability.

    From Lucic & Tse (2024), whose companion notebook calibrates kappa so that the
    average number of orders halves when quoted depth increases by one volatility
    point. Generalised here to ticks so it applies to futures and equities:

        filled_fraction = exp(-kappa * depth_in_ticks)

    `depth_ticks` is how far behind the touch you are resting. Zero means you are
    crossing, and this model degenerates to a full fill - at which point you
    should be using NextBarOpenFill and paying the spread.
    """

    def __init__(
        self,
        kappa: float = 0.75,
        depth_ticks: float = 1.0,
        rng: random.Random | None = None,
    ) -> None:
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if depth_ticks < 0:
            raise ValueError("depth_ticks cannot be negative")
        self.kappa = kappa
        self.depth_ticks = depth_ticks
        self.rng = rng

    @property
    def fill_fraction(self) -> float:
        return math.exp(-self.kappa * self.depth_ticks)

    def fill(
        self,
        order: Order,
        bar: Bar,
        instrument: InstrumentSpec,
        cost_model: CostModel,
        bar_volatility: float,
        crosses_spread: bool = False,
    ) -> Fill | None:
        fraction = self.fill_fraction
        if self.rng is not None:
            fraction = 1.0 if self.rng.random() < fraction else 0.0

        quantity = _quantise(order.quantity * fraction, instrument)
        if quantity <= 0:
            return None

        # Resting behind the touch means a better price than the open by the depth
        # you gave up - that is the whole point of quoting passively.
        edge = self.depth_ticks * instrument.tick_size
        raw_price = bar.open - int(order.side) * edge
        price = min(max(raw_price, bar.low), bar.high)

        per_unit = cost_model.slippage_per_unit(
            instrument, quantity, bar, bar_volatility, crosses_spread
        )
        price = min(max(price + int(order.side) * per_unit, bar.low), bar.high)

        commission = cost_model.commission(instrument, quantity, price)
        slippage_cash = max(0.0, (price - bar.open) * int(order.side)) * quantity * instrument.multiplier

        return Fill(
            ts=bar.ts_open,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage_cash,
            requested_quantity=order.quantity,
        )
