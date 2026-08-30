"""The pluggable data-source interface.

Every feed - free or paid - implements `BarSource`. Nothing above this layer
knows whether bars came from Yahoo, a CSV export, or a paid tick feed, which is
what makes swapping in a real feed later a config change rather than a rewrite.

Sources return RAW bars. They do not label completeness and do not apply the
partial-bar policy; that is `data/bars.py::normalise`, so the policy is applied
identically no matter where the data came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from ..core.clock import Timeframe
from ..core.types import Bar


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """What a source can actually deliver.

    Worth stating explicitly because the limits bite: Yahoo will not give you
    1-minute bars older than 30 days, and a backtest that silently ran on 30 days
    of intraday data is not the backtest you thought you ran.
    """

    name: str
    supported_timeframes: tuple[str, ...]
    max_intraday_history_days: int | None = None
    provides_volume: bool = True
    # ASSUMPTION (data availability): free feeds are adjusted for splits but the
    # adjustment is applied retroactively to the whole series, so a backtest run
    # today sees prices that nobody could have seen at the time. This matters for
    # anything with a price-level threshold. Flagged, not fixed.
    retroactively_adjusted: bool = True


class BarSource(ABC):
    """Adapter interface for OHLCV bar data."""

    @property
    @abstractmethod
    def capabilities(self) -> SourceCapabilities:
        ...

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Return raw bars with ts_open/ts_close in UTC, ascending, unlabelled.

        `start` is inclusive on ts_open, `end` is exclusive.
        """

    def check_request(self, timeframe: Timeframe, start: datetime, end: datetime) -> None:
        """Raise if this source cannot honour the request. Call before fetching."""
        caps = self.capabilities
        if timeframe.label not in caps.supported_timeframes:
            raise ValueError(
                f"{caps.name} does not support timeframe {timeframe.label!r}; "
                f"supported: {caps.supported_timeframes}"
            )
        if caps.max_intraday_history_days is not None and timeframe.is_intraday:
            age_days = (datetime.now(start.tzinfo) - start).days
            if age_days > caps.max_intraday_history_days:
                raise ValueError(
                    f"{caps.name} only serves {caps.max_intraday_history_days} days of "
                    f"{timeframe.label} history; requested start is {age_days} days ago. "
                    "Silently truncating would make the backtest window a lie."
                )
