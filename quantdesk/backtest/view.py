"""The only sanctioned way to hand market data to a strategy.

A `MarketView` owns the full bar series but will not show a strategy any bar that
had not closed by the requested instant. Strategies receive an immutable
`MarketState`; they never touch the underlying series, so they cannot index past
the present even by accident.

This is prevention. `backtest/lookahead.py` is detection - for the leaks that get
in through a feature's own arithmetic rather than through data access.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
from typing import Any, Mapping, Sequence

from ..core.clock import ensure_utc
from ..core.types import Bar, BarCompleteness, MarketState


class LookaheadError(AssertionError):
    """Raised when something tried to see data from after the current instant."""


class MarketView:
    """Causal accessor over a bar series."""

    def __init__(self, symbol: str, bars: Sequence[Bar], require_complete: bool = True) -> None:
        if require_complete:
            bad = [b for b in bars if b.completeness is not BarCompleteness.COMPLETE]
            if bad:
                raise LookaheadError(
                    f"{symbol}: {len(bad)} non-COMPLETE bar(s) passed to MarketView, "
                    f"first at {bad[0].ts_open}. Run data.bars.normalise first - a "
                    "partial bar in a backtest is lookahead with extra steps."
                )
        ordered = sorted(bars, key=lambda b: b.ts_open)
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt.ts_open < prev.ts_close:
                raise LookaheadError(
                    f"{symbol}: overlapping bars at {prev.ts_open} / {nxt.ts_open}. "
                    "Overlapping bars double-count returns."
                )
        self.symbol = symbol
        self._bars: tuple[Bar, ...] = tuple(ordered)
        self._closes: list[datetime] = [b.ts_close for b in self._bars]

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def bars(self) -> tuple[Bar, ...]:
        """The full series. For the ENGINE's use (it needs the next bar to fill
        against); strategies must never receive this."""
        return self._bars

    def visible_count(self, ts: datetime) -> int:
        """How many bars had closed at or before `ts`."""
        return bisect_right(self._closes, ensure_utc(ts))

    def history(self, ts: datetime, lookback: int | None = None) -> tuple[Bar, ...]:
        """Bars with ts_close <= ts, oldest first, optionally the last `lookback`."""
        count = self.visible_count(ts)
        window = self._bars[:count]
        if lookback is not None:
            window = window[-lookback:] if lookback > 0 else ()
        return window

    def state(
        self,
        ts: datetime,
        lookback: int | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> MarketState:
        """Build the immutable snapshot a signal module is allowed to see."""
        return MarketState(
            ts=ensure_utc(ts),
            symbol=self.symbol,
            history=self.history(ts, lookback),
            extras=dict(extras or {}),
        )

    def next_bar_after(self, ts: datetime) -> Bar | None:
        """The first bar that OPENS at or after `ts`.

        This is the engine's execution hook: a decision made at bar N's close is
        filled somewhere inside bar N+1. Returning None means the decision landed
        at the end of the series with no bar left to trade in - those orders must
        be discarded, not filled at the last close, or the backtest gets a free
        trade at a price it could not have transacted at.
        """
        target = ensure_utc(ts)
        for bar in self._bars:
            if bar.ts_open >= target:
                return bar
        return None

    def bar_at_index(self, index: int) -> Bar:
        return self._bars[index]
