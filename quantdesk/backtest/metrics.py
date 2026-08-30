"""Performance statistics. Stdlib only - none of this needs numpy.

Every statistic here reports its sample size alongside its value, because a
profit factor of 1.8 over 11 trades and a profit factor of 1.8 over 900 trades
are different claims about the world, and a report that renders them identically
is lying by omission.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.types import Trade

#: Below this many round trips, treat any statistic as decorative.
MIN_MEANINGFUL_TRADES = 30


@dataclass(frozen=True, slots=True)
class TradeStats:
    """Per-trade statistics. `gross` variants exclude costs, `net` include them."""

    n: int
    wins: int
    losses: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_pnl: float
    total_costs: float
    profit_factor: float
    expectancy: float
    gross_expectancy: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float

    @property
    def is_meaningful(self) -> bool:
        return self.n >= MIN_MEANINGFUL_TRADES

    @property
    def cost_per_trade(self) -> float:
        return self.total_costs / self.n if self.n else 0.0

    @property
    def edge_to_cost_ratio(self) -> float:
        """Gross edge per trade divided by cost per trade.

        The most useful single number in this whole module. Below ~2 the strategy
        is a cost-recovery exercise: a modest error in the spread estimate flips
        the sign of the result. This is the shape of the PF 0.75-0.85 wall.
        """
        cost = self.cost_per_trade
        if cost <= 0:
            return math.inf
        return self.gross_expectancy / cost

    @classmethod
    def from_trades(cls, trades: Sequence[Trade]) -> "TradeStats":
        n = len(trades)
        if n == 0:
            return cls(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        nets = [t.net_pnl for t in trades]
        wins = [p for p in nets if p > 0]
        losses = [p for p in nets if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        costs = sum(t.costs for t in trades)
        gross = sum(t.gross_pnl for t in trades)

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            #: No losing trades. Almost always too small a sample rather than a
            #: miracle, so it is reported as infinity and `is_meaningful` will be
            #: False - rather than quietly substituting a large finite number.
            profit_factor = math.inf
        else:
            profit_factor = 0.0

        return cls(
            n=n,
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / n,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_pnl=sum(nets),
            total_costs=costs,
            profit_factor=profit_factor,
            expectancy=sum(nets) / n,
            gross_expectancy=gross / n,
            average_win=gross_profit / len(wins) if wins else 0.0,
            average_loss=-gross_loss / len(losses) if losses else 0.0,
            largest_win=max(nets) if nets else 0.0,
            largest_loss=min(nets) if nets else 0.0,
        )


@dataclass(frozen=True, slots=True)
class EquityStats:
    n_periods: int
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration: int
    final_equity: float
    peak_equity: float
    min_equity: float = 0.0
    ruined: bool = False
    """Equity reached zero or below at some point.

    This flag exists because a drawdown percentage stops meaning anything once
    equity goes negative: `(value - peak) / peak` will happily report "290%
    drawdown", which reads like a bad-but-survivable number and is in fact an
    account that no longer exists. Real accounts are closed out long before
    this - a broker liquidates, a prop firm fails you - so any statistic
    computed past the ruin point is describing a fiction.

    When this is True, every other figure here is arithmetic on a hypothetical
    and should be reported as ruin rather than as performance."""


def drawdown_series(equity: Sequence[float]) -> list[float]:
    """Fractional drawdown from running peak, at each point."""
    out: list[float] = []
    peak = -math.inf
    for value in equity:
        peak = max(peak, value)
        out.append(0.0 if peak <= 0 else (value - peak) / peak)
    return out


def max_drawdown(equity: Sequence[float]) -> tuple[float, int]:
    """(worst fractional drawdown as a positive number, longest underwater run)."""
    if not equity:
        return 0.0, 0
    dd = drawdown_series(equity)
    worst = -min(dd) if dd else 0.0
    longest = current = 0
    for value in dd:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return worst, longest


def period_returns(equity: Sequence[float]) -> list[float]:
    """Simple returns between consecutive equity points."""
    out: list[float] = []
    for prev, nxt in zip(equity, equity[1:]):
        if prev == 0:
            out.append(0.0)
        else:
            out.append((nxt - prev) / abs(prev))
    return out


def sharpe_ratio(returns: Sequence[float], periods_per_year: float, risk_free: float = 0.0) -> float:
    """Annualised Sharpe.

    `periods_per_year` is required and comes from the Timeframe, not from a
    guess. Annualising 5-minute returns with sqrt(252) overstates Sharpe by
    roughly 8.8x, which is the single most common way a mediocre intraday
    strategy comes to look institutional.
    """
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free / periods_per_year for r in returns]
    mean = sum(excess) / len(excess)
    variance = sum((r - mean) ** 2 for r in excess) / (len(excess) - 1)
    if variance <= 0:
        return 0.0
    return (mean / math.sqrt(variance)) * math.sqrt(periods_per_year)


def sortino_ratio(returns: Sequence[float], periods_per_year: float, target: float = 0.0) -> float:
    """Annualised Sortino, penalising only downside deviation.

    The denominator divides by the count of ALL periods, not just the losing
    ones. Dividing by the losing count is a common implementation error that
    inflates the ratio for strategies which lose rarely but badly - exactly the
    short-vol profile this system is most likely to generate.
    """
    if len(returns) < 2:
        return 0.0
    per_period_target = target / periods_per_year
    mean = sum(returns) / len(returns) - per_period_target
    downside = [min(0.0, r - per_period_target) ** 2 for r in returns]
    dd = math.sqrt(sum(downside) / len(returns))
    if dd <= 0:
        return math.inf if mean > 0 else 0.0
    return (mean / dd) * math.sqrt(periods_per_year)


def equity_stats(equity: Sequence[float], periods_per_year: float) -> EquityStats:
    if not equity:
        return EquityStats(0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    rets = period_returns(equity)
    worst_dd, dd_duration = max_drawdown(equity)
    start = equity[0]
    return EquityStats(
        n_periods=len(equity),
        total_return=(equity[-1] - start) / abs(start) if start else 0.0,
        sharpe=sharpe_ratio(rets, periods_per_year),
        sortino=sortino_ratio(rets, periods_per_year),
        max_drawdown=worst_dd,
        max_drawdown_duration=dd_duration,
        final_equity=equity[-1],
        peak_equity=max(equity),
        min_equity=min(equity),
        ruined=min(equity) <= 0.0,
    )
