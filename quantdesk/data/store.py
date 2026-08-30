"""Normalised bar storage.

SQLite is the default because it is in the stdlib, transactional, and queryable
from anywhere - including the Obsidian export scripts and any dashboard you point
at the file. Parquet is available for bulk analytics via `to_parquet`, with
pyarrow imported lazily.

SCHEMA NOTES
------------
* Timestamps are INTEGER epoch seconds, UTC. Not ISO strings: string comparison
  of mixed-offset timestamps sorts wrongly, and that failure is silent.
* Primary key is (symbol, timeframe, ts_open) so re-ingesting an overlapping
  window UPSERTS rather than duplicating.
* `completeness` and `source` are stored per row. A bar written as PARTIAL that
  is later re-fetched COMPLETE will be upgraded in place - which is exactly what
  should happen, and is why the ingest job can be run repeatedly without care.
* `ingested_at` is kept so you can tell when a series was last refreshed without
  guessing from file mtimes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from ..core.clock import Timeframe, ensure_utc
from ..core.types import Bar, BarCompleteness

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol       TEXT    NOT NULL,
    timeframe    TEXT    NOT NULL,
    ts_open      INTEGER NOT NULL,
    ts_close     INTEGER NOT NULL,
    open         REAL    NOT NULL,
    high         REAL    NOT NULL,
    low          REAL    NOT NULL,
    close        REAL    NOT NULL,
    volume       REAL    NOT NULL,
    completeness TEXT    NOT NULL,
    source       TEXT    NOT NULL,
    ingested_at  INTEGER NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts_open)
);
CREATE INDEX IF NOT EXISTS idx_bars_lookup ON bars (symbol, timeframe, ts_close);
"""


def _epoch(ts: datetime) -> int:
    return int(ensure_utc(ts).timestamp())


def _from_epoch(value: int) -> datetime:
    return datetime.fromtimestamp(value, tz=timezone.utc)


class BarStore:
    """SQLite-backed store for normalised bars."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BarStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write(self, bars: Iterable[Bar], timeframe: Timeframe, source: str) -> int:
        """Upsert bars. Returns the number of rows written."""
        now = int(datetime.now(tz=timezone.utc).timestamp())
        rows = [
            (
                b.symbol,
                timeframe.label,
                _epoch(b.ts_open),
                _epoch(b.ts_close),
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                b.completeness.value,
                source,
                now,
            )
            for b in bars
        ]
        if not rows:
            return 0
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO bars (symbol, timeframe, ts_open, ts_close, open, high, low,
                                  close, volume, completeness, source, ingested_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol, timeframe, ts_open) DO UPDATE SET
                    ts_close=excluded.ts_close, open=excluded.open, high=excluded.high,
                    low=excluded.low, close=excluded.close, volume=excluded.volume,
                    completeness=excluded.completeness, source=excluded.source,
                    ingested_at=excluded.ingested_at
                """,
                rows,
            )
        return len(rows)

    def read(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        complete_only: bool = True,
    ) -> list[Bar]:
        """Read bars ascending by ts_open.

        `complete_only` defaults to True. Reading partial bars into a backtest is
        the failure this whole layer exists to prevent, so you have to ask.
        """
        clauses = ["symbol = ?", "timeframe = ?"]
        params: list[object] = [symbol, timeframe.label]
        if start is not None:
            clauses.append("ts_open >= ?")
            params.append(_epoch(start))
        if end is not None:
            clauses.append("ts_open < ?")
            params.append(_epoch(end))
        if complete_only:
            clauses.append("completeness = ?")
            params.append(BarCompleteness.COMPLETE.value)
        sql = f"SELECT * FROM bars WHERE {' AND '.join(clauses)} ORDER BY ts_open ASC"
        cursor = self._conn.execute(sql, params)
        return [
            Bar(
                symbol=row["symbol"],
                ts_open=_from_epoch(row["ts_open"]),
                ts_close=_from_epoch(row["ts_close"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                completeness=BarCompleteness(row["completeness"]),
            )
            for row in cursor
        ]

    def coverage(self, symbol: str, timeframe: Timeframe) -> tuple[datetime, datetime, int] | None:
        """(first ts_open, last ts_close, row count) or None if empty.

        Call this before a backtest and print it. A run that silently covered
        three months when you believed it covered three years is the cheapest
        possible mistake to catch and one of the easiest to miss.
        """
        row = self._conn.execute(
            """
            SELECT MIN(ts_open) AS lo, MAX(ts_close) AS hi, COUNT(*) AS n
            FROM bars WHERE symbol = ? AND timeframe = ?
            """,
            (symbol, timeframe.label),
        ).fetchone()
        if row is None or row["n"] == 0:
            return None
        return _from_epoch(row["lo"]), _from_epoch(row["hi"]), int(row["n"])

    def symbols(self) -> list[tuple[str, str]]:
        cursor = self._conn.execute(
            "SELECT DISTINCT symbol, timeframe FROM bars ORDER BY symbol, timeframe"
        )
        return [(r["symbol"], r["timeframe"]) for r in cursor]

    def to_parquet(self, symbol: str, timeframe: Timeframe, path: str | Path) -> Path:
        """Export one series to Parquet for bulk analytics.

        pandas/pyarrow are imported here and nowhere else in the data layer.
        """
        import pandas  # lazy

        bars = self.read(symbol, timeframe, complete_only=False)
        frame = pandas.DataFrame(
            [
                {
                    "ts_open": b.ts_open,
                    "ts_close": b.ts_close,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "completeness": b.completeness.value,
                }
                for b in bars
            ]
        )
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
        return out


def ingest(
    source,
    store: BarStore,
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    as_of: datetime | None = None,
    policy=None,
) -> dict[str, object]:
    """Fetch -> normalise -> store, returning a report you should actually read.

    The report exists so ingest is not a silent operation. Gap counts and dropped
    partial-bar counts are the two numbers that tell you whether the series you
    just stored is the series you think you stored.
    """
    from .bars import PartialBarPolicy, find_gaps, normalise

    policy = policy or PartialBarPolicy.DROP
    stamp = ensure_utc(as_of) if as_of else datetime.now(tz=timezone.utc)

    raw = source.fetch(symbol, timeframe, start, end)
    clean = normalise(raw, timeframe, stamp, policy)
    written = store.write(clean, timeframe, source.capabilities.name)
    gaps = find_gaps(clean, timeframe)
    return {
        "symbol": symbol,
        "timeframe": timeframe.label,
        "source": source.capabilities.name,
        "fetched": len(raw),
        "stored": written,
        "dropped_incomplete": len(raw) - len(clean),
        "gaps": len(gaps),
        "first": clean[0].ts_open.isoformat() if clean else None,
        "last": clean[-1].ts_close.isoformat() if clean else None,
    }
