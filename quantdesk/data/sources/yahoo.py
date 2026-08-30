"""Yahoo Finance bar source, via yfinance.

Same feed the GEX dashboard uses, so numbers tie out between the two systems.
`yfinance` and `pandas` are imported INSIDE `fetch`, so importing this module -
or anything above it - does not require them.

Known limits of this feed, all of which will silently produce a prettier
backtest than reality if you do not account for them:

* 1-minute history is capped at ~30 days, and other intraday timeframes at ~60.
  `check_request` raises rather than truncating.
* Prices are retroactively split/dividend adjusted, so the series you backtest
  is not the series that was printed at the time.
* There is no bid/ask. Every spread in the cost model is therefore an ESTIMATE,
  not an observation - see backtest/costs.py.
* The final bar of a live fetch is the still-forming one. It is returned with a
  truthful ts_close so `data/bars.py::normalise` can drop it.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from ...core.clock import Timeframe, ensure_utc
from ...core.types import Bar
from ..source import BarSource, SourceCapabilities

# ASSUMPTION (market microstructure): US equity RTH. Override for other venues.
EXCHANGE_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)

# Map our timeframe labels onto yfinance interval strings.
_INTERVALS = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
}

#: A "daily" bar is one regular-hours session, not 24 hours. Defining it this way
#: means ts_close lands on the actual 16:00 ET close, so the bar becomes knowable
#: when it really became knowable - and periods_per_year still works out to 252.
SESSION_DAY = Timeframe("1d", seconds=23_400)


class YahooBarSource(BarSource):
    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    @property
    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name="yahoo",
            supported_timeframes=tuple(_INTERVALS),
            max_intraday_history_days=30,
            provides_volume=True,
            retroactively_adjusted=self.auto_adjust,
        )

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        self.check_request(timeframe, start, end)
        interval = _INTERVALS[timeframe.label]

        import yfinance  # lazy - keeps the core dependency-free

        frame = yfinance.download(
            tickers=symbol,
            start=ensure_utc(start).date().isoformat(),
            end=ensure_utc(end).date().isoformat(),
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=False,
            threads=False,
        )
        if frame is None or frame.empty:
            return []
        # yfinance returns a MultiIndex column frame for multi-ticker requests and
        # has flip-flopped on doing it for single tickers too. Flatten defensively.
        if hasattr(frame.columns, "nlevels") and frame.columns.nlevels > 1:
            frame = frame.droplevel(axis=1, level=-1)

        bars: list[Bar] = []
        for index, row in frame.iterrows():
            ts_open = self._bar_open(index, timeframe)
            bars.append(
                Bar(
                    symbol=symbol,
                    ts_open=ts_open,
                    ts_close=timeframe.close_of(ts_open),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0) or 0.0),
                )
            )
        bars.sort(key=lambda b: b.ts_open)
        return bars

    @staticmethod
    def _bar_open(index_value, timeframe: Timeframe) -> datetime:
        """Turn a pandas index entry into a true UTC bar-open timestamp.

        Daily rows arrive as a bare date. A date is not an instant: pinning it to
        midnight would claim the bar was knowable ~16 hours before it was. We
        localise to the session open in exchange time and convert, which also
        gets DST right - the same bar is 13:30 UTC in summer and 14:30 in winter.
        """
        stamp = index_value.to_pydatetime() if hasattr(index_value, "to_pydatetime") else index_value
        if timeframe.is_intraday and timeframe.seconds < 23_400:
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=EXCHANGE_TZ)
            return ensure_utc(stamp)
        session_date = stamp.date()
        localised = datetime.combine(session_date, SESSION_OPEN, tzinfo=EXCHANGE_TZ)
        return ensure_utc(localised)
