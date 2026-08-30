"""Persistent storage for options-chain snapshots.

WHY THIS EXISTS
---------------
Historical options positioning cannot be bought retroactively. Vendors sell you
*today's* chain; nobody sells you what open interest looked like on an arbitrary
past Tuesday, and the vendors that do are priced for institutions. The only way
to have that history is to have recorded it.

So this is the highest-value process in the system, and it does not depend on any
strategy existing. Every session not recorded is permanently gone.

WHAT IT ENABLES THAT A SINGLE SNAPSHOT CANNOT
---------------------------------------------
* **oi_change** - day-over-day open interest change per contract. This was a paid
  Unusual Whales field; differencing consecutive snapshots computes it for free.
  It is the difference between "there is size at 760" and "size arrived at 760
  yesterday", which is the part that carries information.
* IV rank and IV percentile, which are definitionally historical.
* Any backtest of a GEX or positioning signal at all.

SCHEMA NOTES
------------
* `session_date` is the TRADING date (see OptionChain.as_of_date), stored as an
  ISO string so it sorts correctly. `fetched_at` is epoch seconds UTC.
* `source` is recorded per snapshot. If the feed ever changes - as it just did,
  from Unusual Whales to CBOE - any study spanning the boundary is comparing two
  different instruments and needs to know that.
* Contracts with zero open interest AND zero volume are dropped. They carry no
  positioning information and are roughly a quarter of the book.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .options import OptionChain, OptionContract

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chain_snapshots (
    underlying    TEXT    NOT NULL,
    session_date  TEXT    NOT NULL,
    fetched_at    INTEGER NOT NULL,
    source        TEXT    NOT NULL,
    spot          REAL    NOT NULL,
    spot_bid      REAL,
    spot_ask      REAL,
    n_contracts   INTEGER NOT NULL,
    n_stored      INTEGER NOT NULL,
    total_oi      REAL    NOT NULL,
    parity_breaks INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (underlying, session_date, fetched_at)
);

CREATE TABLE IF NOT EXISTS chain_contracts (
    underlying     TEXT    NOT NULL,
    session_date   TEXT    NOT NULL,
    fetched_at     INTEGER NOT NULL,
    symbol         TEXT    NOT NULL,
    expiry         TEXT    NOT NULL,
    strike         REAL    NOT NULL,
    right          TEXT    NOT NULL,
    bid            REAL,
    ask            REAL,
    last           REAL,
    volume         REAL,
    open_interest  REAL,
    iv             REAL,
    vendor_delta   REAL,
    vendor_gamma   REAL,
    PRIMARY KEY (underlying, fetched_at, symbol)
);

CREATE INDEX IF NOT EXISTS idx_chain_session
    ON chain_contracts (underlying, session_date);
CREATE INDEX IF NOT EXISTS idx_chain_strike
    ON chain_contracts (underlying, session_date, expiry, strike);
"""


class ChainStore:
    """SQLite storage for chain snapshots."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ChainStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write_chain(self, chain: OptionChain, keep_empty: bool = False) -> dict[str, Any]:
        """Store one snapshot. Returns a report you should log.

        Idempotent on (underlying, fetched_at, symbol), so re-running the
        recorder cannot duplicate a session.
        """
        session = chain.as_of_date.isoformat()
        fetched = int(chain.fetched_at.timestamp())

        keep: list[OptionContract] = [
            c for c in chain.contracts
            if keep_empty or c.open_interest > 0 or c.volume > 0
        ]
        rows = [
            (
                chain.underlying, session, fetched, c.symbol, c.expiry.isoformat(),
                c.strike, c.right, c.bid, c.ask, c.last, c.volume,
                c.open_interest, c.iv, c.vendor_delta, c.vendor_gamma,
            )
            for c in keep
        ]

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO chain_snapshots
                    (underlying, session_date, fetched_at, source, spot, spot_bid,
                     spot_ask, n_contracts, n_stored, total_oi, parity_breaks)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(underlying, session_date, fetched_at) DO UPDATE SET
                    source=excluded.source, spot=excluded.spot,
                    n_contracts=excluded.n_contracts, n_stored=excluded.n_stored,
                    total_oi=excluded.total_oi, parity_breaks=excluded.parity_breaks
                """,
                (
                    chain.underlying, session, fetched, chain.source, chain.spot,
                    chain.spot_bid, chain.spot_ask, len(chain.contracts), len(keep),
                    chain.total_open_interest(), len(chain.parity_breaks()),
                ),
            )
            self._conn.executemany(
                """
                INSERT INTO chain_contracts
                    (underlying, session_date, fetched_at, symbol, expiry, strike,
                     right, bid, ask, last, volume, open_interest, iv,
                     vendor_delta, vendor_gamma)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(underlying, fetched_at, symbol) DO UPDATE SET
                    bid=excluded.bid, ask=excluded.ask, last=excluded.last,
                    volume=excluded.volume, open_interest=excluded.open_interest,
                    iv=excluded.iv, vendor_delta=excluded.vendor_delta,
                    vendor_gamma=excluded.vendor_gamma
                """,
                rows,
            )

        return {
            "underlying": chain.underlying,
            "session_date": session,
            "source": chain.source,
            "spot": chain.spot,
            "fetched": len(chain.contracts),
            "stored": len(keep),
            "dropped_empty": len(chain.contracts) - len(keep),
            "total_oi": chain.total_open_interest(),
            "parity_breaks": len(chain.parity_breaks()),
        }

    def read_chain(self, underlying: str, session_date: str | date,
                   fetched_at: int | None = None) -> OptionChain | None:
        """Load one stored snapshot back as an OptionChain.

        The store could write but not read, which meant every recorded session
        was write-only and no backtest could reach it. Recording history you
        cannot load is the same as not recording it.

        A session may hold several snapshots (premarket, midday, close). With
        `fetched_at` unset this returns the LAST one of the session, which is
        the end-of-day book and the one open-interest studies want. Pass an
        explicit `fetched_at` from `snapshots_for()` to pick another.

        Returns None when the session was never recorded - callers backtesting
        over a date range must expect gaps, because the recorder does stop.
        """
        session = (session_date.isoformat()
                   if isinstance(session_date, date) else session_date)

        if fetched_at is None:
            row = self._conn.execute(
                """
                SELECT fetched_at FROM chain_snapshots
                 WHERE underlying=? AND session_date=?
                 ORDER BY fetched_at DESC LIMIT 1
                """, (underlying, session)).fetchone()
            if row is None:
                return None
            fetched_at = int(row[0])

        head = self._conn.execute(
            """
            SELECT source, spot, spot_bid, spot_ask FROM chain_snapshots
             WHERE underlying=? AND session_date=? AND fetched_at=?
            """, (underlying, session, fetched_at)).fetchone()
        if head is None:
            return None
        source, spot, spot_bid, spot_ask = head

        contracts = tuple(
            OptionContract(
                symbol=r[0], underlying=underlying,
                expiry=date.fromisoformat(r[1]), strike=r[2], right=r[3],
                bid=r[4], ask=r[5], last=r[6], volume=r[7],
                open_interest=r[8], iv=r[9],
                vendor_delta=r[10], vendor_gamma=r[11],
            )
            for r in self._conn.execute(
                """
                SELECT symbol, expiry, strike, right, bid, ask, last, volume,
                       open_interest, iv, vendor_delta, vendor_gamma
                  FROM chain_contracts
                 WHERE underlying=? AND session_date=? AND fetched_at=?
                """, (underlying, session, fetched_at))
        )
        if not contracts:
            return None

        return OptionChain(
            underlying=underlying, spot=spot,
            fetched_at=datetime.fromtimestamp(fetched_at, tz=timezone.utc),
            contracts=contracts, source=source,
            session_date=date.fromisoformat(session),
            spot_bid=spot_bid, spot_ask=spot_ask,
        )

    def snapshots_for(self, underlying: str,
                      session_date: str | date) -> list[int]:
        """Every `fetched_at` recorded for one session, ascending."""
        session = (session_date.isoformat()
                   if isinstance(session_date, date) else session_date)
        return [int(r[0]) for r in self._conn.execute(
            """
            SELECT fetched_at FROM chain_snapshots
             WHERE underlying=? AND session_date=? ORDER BY fetched_at
            """, (underlying, session))]

    def sessions(self, underlying: str) -> list[str]:
        """Recorded session dates, ascending."""
        cursor = self._conn.execute(
            "SELECT DISTINCT session_date FROM chain_snapshots WHERE underlying = ?"
            " ORDER BY session_date",
            (underlying,),
        )
        return [r["session_date"] for r in cursor]

    def coverage(self, underlying: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n, MIN(session_date) AS lo, MAX(session_date) AS hi,
                   COUNT(DISTINCT source) AS sources
            FROM chain_snapshots WHERE underlying = ?
            """,
            (underlying,),
        ).fetchone()
        if row is None or row["n"] == 0:
            return None
        contracts = self._conn.execute(
            "SELECT COUNT(*) AS n FROM chain_contracts WHERE underlying = ?", (underlying,)
        ).fetchone()["n"]
        return {
            "snapshots": row["n"],
            "first_session": row["lo"],
            "last_session": row["hi"],
            "distinct_sources": row["sources"],
            "contract_rows": contracts,
        }

    def missing_sessions(self, underlying: str, since: date | None = None) -> list[str]:
        """Weekdays with no snapshot, between the first recorded session and today.

        Approximate - it does not know market holidays, so US holidays appear as
        gaps. That is the correct bias: an over-reported gap prompts a look, an
        under-reported one hides a dead recorder. This is the check that would
        have caught the capture pipeline dying on 2026-08-12.
        """
        recorded = set(self.sessions(underlying))
        if not recorded:
            return []
        start = since or date.fromisoformat(min(recorded))
        today = datetime.now(tz=timezone.utc).date()
        missing: list[str] = []
        current = start
        while current <= today:
            if current.weekday() < 5 and current.isoformat() not in recorded:
                missing.append(current.isoformat())
            current = date.fromordinal(current.toordinal() + 1)
        return missing

    def open_interest_change(
        self, underlying: str, session: str, previous: str | None = None
    ) -> list[dict[str, Any]]:
        """Day-over-day open-interest change per contract.

        The paid `oi_change` field, computed from our own recordings. Returns
        contracts present in `session`; `oi_prev` is None where the contract did
        not exist in the previous snapshot, which is different from an OI of
        zero and must not be conflated with it.
        """
        sessions = self.sessions(underlying)
        if session not in sessions:
            return []
        if previous is None:
            index = sessions.index(session)
            if index == 0:
                return []
            previous = sessions[index - 1]

        cursor = self._conn.execute(
            """
            SELECT cur.symbol, cur.expiry, cur.strike, cur.right,
                   cur.open_interest AS oi, prev.open_interest AS oi_prev
            FROM chain_contracts cur
            LEFT JOIN chain_contracts prev
              ON prev.underlying = cur.underlying
             AND prev.symbol = cur.symbol
             AND prev.session_date = ?
            WHERE cur.underlying = ? AND cur.session_date = ?
            """,
            (previous, underlying, session),
        )
        out: list[dict[str, Any]] = []
        for row in cursor:
            prev_oi = row["oi_prev"]
            out.append({
                "symbol": row["symbol"],
                "expiry": row["expiry"],
                "strike": row["strike"],
                "right": row["right"],
                "oi": row["oi"],
                "oi_prev": prev_oi,
                "oi_change": None if prev_oi is None else row["oi"] - prev_oi,
                "is_new": prev_oi is None,
            })
        return out
