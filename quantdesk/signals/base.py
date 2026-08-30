"""The signal-module interface, and per-module performance accounting.

Every alpha family - mean reversion, momentum, GEX regime, CISD/session - is a
subclass of `SignalModule` and nothing else. The engine knows only this
interface, so modules can be added, removed, or benchmarked against each other
without the engine changing.

`ModuleLedger` is the "which of these is actually contributing" machinery. Each
module's trades are tracked separately and scored independently, so a module that
is net-negative after costs cannot hide inside a portfolio equity curve that
another module is carrying.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

from ..core.types import Direction, MarketState, Signal, Trade
from ..backtest.metrics import TradeStats


class SignalModule(ABC):
    """Common interface for every alpha module."""

    #: Bars of history required before this module will emit anything real.
    #: The engine will not call `generate_signal` before this many bars exist,
    #: which stops a module from silently computing a 20-period mean from 3 bars.
    warmup: int = 0

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def generate_signal(self, market_state: MarketState) -> Signal:
        """Return a Signal for `market_state`.

        Contract, enforced by the engine and by `backtest.lookahead`:
        * Read only from `market_state.history`, which ends at `market_state.ts`.
        * Be deterministic - same state in, same signal out.
        * Return `Direction.FLAT` with confidence 0 rather than raising when there
          is nothing to say. A module that raises on quiet markets makes the
          backtest and live paths diverge.
        """

    def reset(self) -> None:
        """Clear any internal state. Called between walk-forward windows so a
        module cannot carry fitted state from one test window into the next."""

    def flat(self, market_state: MarketState, why: str = "no setup") -> Signal:
        """Convenience for the common 'nothing to do' return."""
        return Signal(
            module=self.name,
            ts=market_state.ts,
            symbol=market_state.symbol,
            direction=Direction.FLAT,
            confidence=0.0,
            reasoning=why,
        )


@dataclass
class ModuleLedger:
    """Per-module trade accounting, so dead weight is visible."""

    trades: dict[str, list[Trade]] = field(default_factory=lambda: defaultdict(list))

    def record(self, trade: Trade) -> None:
        self.trades[trade.module].append(trade)

    def stats(self) -> dict[str, TradeStats]:
        return {name: TradeStats.from_trades(ts) for name, ts in self.trades.items()}

    def contributors(self, min_profit_factor: float = 1.0) -> dict[str, bool]:
        """Which modules clear the bar after costs.

        A module with too few trades to be meaningful returns False. Not
        "undecided" - False. The default assumption for an unproven module is
        that it does not work, and it should have to earn its way out of that.
        """
        out: dict[str, bool] = {}
        for name, stats in self.stats().items():
            out[name] = stats.is_meaningful and stats.profit_factor >= min_profit_factor
        return out

    def report_lines(self) -> list[str]:
        lines = []
        for name, s in sorted(self.stats().items()):
            flag = "" if s.is_meaningful else "  [SAMPLE TOO SMALL]"
            lines.append(
                f"{name:<28} n={s.n:>5}  PF={s.profit_factor:>6.2f}  "
                f"win={s.win_rate:>6.1%}  exp={s.expectancy:>9.2f}  "
                f"edge/cost={s.edge_to_cost_ratio:>5.2f}{flag}"
            )
        return lines
