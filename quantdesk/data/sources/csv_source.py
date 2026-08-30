"""CSV bar source - reads TradingView exports and generic OHLCV csv.

Written against the actual exports already on disk (`AMEX_SPY, 5_*.csv`), which
are ISO-8601 with a UTC offset and sometimes carry no volume column.

Uses the stdlib `csv` module: no pandas in the ingest path, so the data layer
stays importable in a bare interpreter.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from ...core.clock import Timeframe, ensure_utc
from ...core.types import Bar
from ..source import BarSource, SourceCapabilities

# TradingView writes these; other exporters vary, hence the override hook.
DEFAULT_COLUMNS = {
    "ts": "time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


class CsvBarSource(BarSource):
    """Read bars from a CSV file.

    Timestamps carrying an offset (`2026-07-06T14:40:00-04:00`) are converted to
    UTC. Naive timestamps are REJECTED - see `core.clock.ensure_utc` for why.
    Pass `assume_tz` only if you have verified what the exporter meant.
    """

    def __init__(
        self,
        path: str | Path,
        symbol: str,
        timeframe: Timeframe,
        columns: dict[str, str] | None = None,
        assume_tz: str | None = None,
        allow_missing_volume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.symbol = symbol
        self.timeframe = timeframe
        self.columns = {**DEFAULT_COLUMNS, **(columns or {})}
        self.assume_tz = assume_tz
        self.allow_missing_volume = allow_missing_volume
        if not self.path.exists():
            raise FileNotFoundError(f"csv not found: {self.path}")

    @property
    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=f"csv:{self.path.name}",
            supported_timeframes=(self.timeframe.label,),
            max_intraday_history_days=None,
            provides_volume=not self.allow_missing_volume,
            retroactively_adjusted=False,
        )

    def _parse_ts(self, raw: str) -> datetime:
        text = raw.strip()
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            # Epoch seconds are the other common TradingView export shape.
            if text.isdigit():
                parsed = datetime.fromtimestamp(int(text), tz=ensure_utc(datetime.now()).tzinfo)
            else:
                raise ValueError(f"{self.path.name}: cannot parse timestamp {raw!r}") from None
        if parsed.tzinfo is None:
            if self.assume_tz is None:
                raise ValueError(
                    f"{self.path.name}: timestamp {raw!r} has no timezone. Pass "
                    "assume_tz explicitly once you have confirmed what the exporter meant - "
                    "guessing shifts the whole series."
                )
            from zoneinfo import ZoneInfo

            parsed = parsed.replace(tzinfo=ZoneInfo(self.assume_tz))
        return ensure_utc(parsed)

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        if timeframe.seconds != self.timeframe.seconds:
            raise ValueError(
                f"{self.path.name} holds {self.timeframe.label} bars, "
                f"asked for {timeframe.label}"
            )
        lo, hi = ensure_utc(start), ensure_utc(end)
        cols = self.columns
        bars: list[Bar] = []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            missing = [c for c in ("ts", "open", "high", "low", "close") if cols[c] not in header]
            if missing:
                raise ValueError(
                    f"{self.path.name}: missing required column(s) "
                    f"{[cols[m] for m in missing]}; header was {header}"
                )
            has_volume = cols["volume"] in header
            if not has_volume and not self.allow_missing_volume:
                raise ValueError(
                    f"{self.path.name}: no {cols['volume']!r} column. Volume-relative "
                    "fill and impact models cannot run without it. Pass "
                    "allow_missing_volume=True to ingest anyway - volume will be 0 and "
                    "those models will refuse to price fills rather than invent liquidity."
                )
            for row in reader:
                ts_open = self._parse_ts(row[cols["ts"]])
                if not (lo <= ts_open < hi):
                    continue
                raw_volume = row.get(cols["volume"], "") if has_volume else ""
                bars.append(
                    Bar(
                        symbol=symbol,
                        ts_open=ts_open,
                        ts_close=self.timeframe.close_of(ts_open),
                        open=float(row[cols["open"]]),
                        high=float(row[cols["high"]]),
                        low=float(row[cols["low"]]),
                        close=float(row[cols["close"]]),
                        volume=float(raw_volume) if raw_volume not in ("", None) else 0.0,
                    )
                )
        bars.sort(key=lambda b: b.ts_open)
        return bars
