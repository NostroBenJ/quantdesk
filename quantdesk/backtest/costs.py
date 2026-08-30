"""Transaction costs. Slippage is a function of volatility and size, never a constant.

A flat "2 ticks per trade" assumption is wrong in the single most dangerous way:
it is roughly right in calm markets and wildly optimistic in exactly the fast
markets where a signal fires most often. That correlation - more trades when
costs are worst - is what turns a backtested PF of 1.3 into a live PF of 0.8.

WHAT IS MODELLED
----------------
1. Commission: per-unit plus a minimum ticket.
2. Half-spread: paid only when the order crosses. Bennett is explicit that a long
   gamma position can rest on the bid and offer while a short gamma position must
   cross, so `crosses_spread` is a parameter and not an assumption.
3. Market impact: square-root of participation rate. Standard practitioner form,
   and it has the right shape - impact grows sublinearly in size, so a model that
   scales impact linearly will under-penalise small orders and over-penalise large.

WHAT IS NOT MODELLED, AND WILL FLATTER YOU
------------------------------------------
* Queue position. Resting orders are assumed to fill by participation share.
* Adverse selection on resting orders: in reality the fills you get on a limit
  order are disproportionately the ones you did not want.
* Overnight gaps through a stop. A stop is filled at the gapped price here only
  if you feed the engine a bar that gaps; it does not model exchange behaviour.
* Borrow cost on shorts, and financing.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.types import Bar, InstrumentSpec, Side


class CostError(ValueError):
    """Raised when a cost model is asked for a number it cannot honestly produce."""


@dataclass(frozen=True, slots=True)
class CommissionSchedule:
    """Per-unit commission plus fees.

    ASSUMPTION (broker behaviour): defaults are placeholders. Replace them with
    your actual schedule before believing any net number - for prop-firm
    evaluations in particular, commissions are often the difference between
    passing and failing, and they vary per firm.
    """

    per_unit: float = 0.0
    per_trade_minimum: float = 0.0
    exchange_fee_per_unit: float = 0.0
    percent_of_notional: float = 0.0

    def charge(self, quantity: float, price: float, multiplier: float) -> float:
        units = abs(quantity)
        cost = units * (self.per_unit + self.exchange_fee_per_unit)
        cost += self.percent_of_notional * abs(price * units * multiplier)
        return max(cost, self.per_trade_minimum) if units else 0.0


@dataclass(frozen=True, slots=True)
class SlippageParams:
    """Knobs for the volatility/size slippage model. All config-driven."""

    #: Half-spread as a fraction of the bar's own volatility (sigma * price).
    #: ASSUMPTION (microstructure): 0.15 is a working figure for liquid US
    #: index products. Free data carries no bid/ask, so this is an ESTIMATE and
    #: is the largest single source of error in any net P&L this engine reports.
    #: Calibrate it against real fills as soon as you have any.
    spread_vol_fraction: float = 0.15

    #: Floor on the half-spread, in ticks. Nothing trades tighter than this.
    min_half_spread_ticks: float = 0.5

    #: Square-root impact coefficient: impact = eta * sigma * sqrt(participation).
    impact_coefficient: float = 0.5

    #: Fraction of a bar's volume you assume you can take without extra impact.
    #: Above this, `fills.py` will start partially filling.
    max_participation: float = 0.10


class CostModel(ABC):
    @abstractmethod
    def commission(self, instrument: InstrumentSpec, quantity: float, price: float) -> float:
        ...

    @abstractmethod
    def slippage_per_unit(
        self,
        instrument: InstrumentSpec,
        quantity: float,
        bar: Bar,
        bar_volatility: float,
        crosses_spread: bool = True,
    ) -> float:
        """Slippage in PRICE units per unit traded, always positive."""

    def total_cost(
        self,
        instrument: InstrumentSpec,
        quantity: float,
        price: float,
        bar: Bar,
        bar_volatility: float,
        crosses_spread: bool = True,
    ) -> tuple[float, float]:
        """Return (commission_cash, slippage_cash) for a fill."""
        commission = self.commission(instrument, quantity, price)
        per_unit = self.slippage_per_unit(
            instrument, quantity, bar, bar_volatility, crosses_spread
        )
        slippage = per_unit * abs(quantity) * instrument.multiplier
        return commission, slippage


class ZeroCostModel(CostModel):
    """Frictionless. Exists ONLY as the numerator of the cost-sensitivity report -
    the gross curve you compare the net curve against. Never run a strategy
    evaluation on this and call the result a result."""

    def commission(self, instrument: InstrumentSpec, quantity: float, price: float) -> float:
        return 0.0

    def slippage_per_unit(self, instrument, quantity, bar, bar_volatility, crosses_spread=True) -> float:
        return 0.0


class VolatilityScaledCostModel(CostModel):
    """The default. Spread and impact both scale with the bar's own volatility."""

    def __init__(
        self,
        commissions: CommissionSchedule | None = None,
        params: SlippageParams | None = None,
        multiple: float = 1.0,
    ) -> None:
        self.commissions = commissions or CommissionSchedule()
        self.params = params or SlippageParams()
        #: Scales the whole slippage estimate. The cost-sensitivity sweep varies
        #: this to answer "how wrong can my spread estimate be before the edge dies".
        self.multiple = multiple

    def with_multiple(self, multiple: float) -> "VolatilityScaledCostModel":
        return VolatilityScaledCostModel(self.commissions, self.params, multiple)

    def commission(self, instrument: InstrumentSpec, quantity: float, price: float) -> float:
        return self.commissions.charge(quantity, price, instrument.multiplier)

    def slippage_per_unit(
        self,
        instrument: InstrumentSpec,
        quantity: float,
        bar: Bar,
        bar_volatility: float,
        crosses_spread: bool = True,
    ) -> float:
        if bar_volatility < 0:
            raise CostError(f"negative volatility {bar_volatility}")
        p = self.params
        price = bar.close
        floor = p.min_half_spread_ticks * instrument.tick_size

        half_spread = max(p.spread_vol_fraction * bar_volatility * price, floor)
        cost = half_spread if crosses_spread else 0.0

        if p.impact_coefficient > 0 and abs(quantity) > 0:
            if bar.volume <= 0:
                raise CostError(
                    f"{bar.symbol} @ {bar.ts_open}: impact model needs bar volume but "
                    "volume is 0. Either the source provides no volume (see "
                    "CsvBarSource.allow_missing_volume) or the bar is bad. Refusing "
                    "to price impact as zero, which would invent free liquidity."
                )
            participation = abs(quantity) / bar.volume
            cost += p.impact_coefficient * bar_volatility * price * math.sqrt(participation)

        return cost * self.multiple


def side_crosses_spread(side: Side, is_long_gamma: bool) -> bool:
    """Bennett's rule, encoded.

    A long-gamma delta hedge buys as the market falls and sells as it rises, so it
    can sit passively on the bid and the offer. A short-gamma hedge does the
    opposite and must cross. This is a real, sign-dependent cost asymmetry, and
    charging both sides the same spread systematically flatters short-vol
    strategies - which is the direction of error we are trying to eliminate.
    """
    return not is_long_gamma
