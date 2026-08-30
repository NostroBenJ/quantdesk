"""Databento OPRA `statistics` files -> OptionChain.

Everything in this module was learned by decoding a real order rather than
from the docs, and each rule below is one silent corruption avoided.

THE FIVE RULES
--------------
1. OPEN INTEREST IS IN `quantity`, NOT `price`. stat_type 9 carries the OI
   in the quantity field and leaves price set to the undefined sentinel.
   Reading `price` gives you INT64_MAX, and reading it as a float gives you
   9.2e18 open interest, which is not obviously wrong until it is.

2. OPEN INTEREST DESCRIBES THE PREVIOUS SESSION. Every OI record in the
   file dated D is stamped 06:30 ET on D -- before that session opens. It
   is the book as of D-1's close. That makes it causally valid for
   predicting session D (you hold it before the bell), but it is NOT the
   book at D's close, and treating it as such shifts a study by a day.

3. THE FILE MIXES TWO SESSIONS. Prices in the same file are stamped 16:15
   and 17:45 ET on D, so a naive per-file read pairs D-1's positioning with
   D's prices. `load_session` returns both, labelled, rather than pretending
   they are contemporaneous.

4. THE LATE PUBLICATION IS A REVISION. Price stats appear twice, at 16:15
   and 17:45. 82.8% are identical; of the rest, almost all are 0 -> a real
   value. Take the LATEST per (instrument, stat_type) or lose ~17% of
   prices to zeros.

5. PRICES ARE 1e-9 FIXED POINT, and both INT64_MAX and UINT64_MAX appear as
   "undefined". Check for both.

WHAT YOU GET, AND WHAT YOU STILL NEED
-------------------------------------
The statistics schema alone carries open interest AND a bid/offer for
essentially every contract (LOWEST_OFFER 100% populated, HIGHEST_BID 98.1%
on the session measured; CLOSE_PRICE only 14.8%, since only traded
contracts have one). So no quote or OHLCV schema is needed.

ASSUMPTION (microstructure): HIGHEST_BID and LOWEST_OFFER are session
EXTREMES, not a simultaneous quote. Their midpoint is a reasonable central
estimate but it is not a real mid, and on a volatile session the highest
bid can exceed the lowest offer. `crossed` marks those rows rather than
hiding them.

Records key on `instrument_id`, so a Definition order is required to map
them to strike/expiry/right. Without it this module yields OI by
instrument and nothing that can be priced.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

#: Both appear in real files as "no value".
UNDEFINED = (9223372036854775807, 18446744073709551615, -9223372036854775808)

#: DBN fixed-point scale for prices.
PRICE_SCALE = 1e-9

#: stat_type codes we consume.
STAT_OPEN_INTEREST = 9
STAT_HIGHEST_BID = 8
STAT_LOWEST_OFFER = 7
STAT_CLOSE_PRICE = 11

_ET = dt.timezone(dt.timedelta(hours=-4))


def _defined(value: int) -> bool:
    return value not in UNDEFINED


def _price(raw: int) -> float | None:
    return raw * PRICE_SCALE if _defined(raw) and raw != 0 else None


@dataclass(frozen=True)
class InstrumentStats:
    """One instrument's daily statistics, with the two clocks kept apart."""

    instrument_id: int
    open_interest: float | None = None
    #: The session the OPEN INTEREST describes (D-1), not the file's date.
    oi_session: dt.date | None = None
    bid: float | None = None
    offer: float | None = None
    close: float | None = None
    #: The session the PRICES describe (D).
    price_session: dt.date | None = None

    @property
    def mid(self) -> float | None:
        """Midpoint of the session's extreme bid and offer. See the
        ASSUMPTION above -- this is not a simultaneous quote."""
        if self.bid is None or self.offer is None:
            return None
        return (self.bid + self.offer) / 2.0

    @property
    def crossed(self) -> bool:
        """Highest bid above lowest offer. Not an error -- it means the
        market moved through the day -- but the mid is meaningless there."""
        return (self.bid is not None and self.offer is not None
                and self.bid > self.offer)


def read_statistics(path: str | Path) -> Iterator:
    """Yield raw StatMsg records. `databento` is imported here, not at
    module scope, so quantdesk still imports in a bare interpreter."""
    import databento as db                      # presentation-layer import
    for record in db.DBNStore.from_file(str(path)):
        yield record


def load_session(path: str | Path) -> dict[int, InstrumentStats]:
    """One day's file -> {instrument_id: InstrumentStats}.

    Applies rules 1-5. The returned objects carry BOTH session dates so a
    caller cannot accidentally treat the open interest and the prices as
    belonging to the same day.
    """
    oi: dict[int, tuple[float, dt.date]] = {}
    # (instrument, stat_type) -> (ts_event, price). Latest publication wins.
    latest: dict[tuple[int, int], tuple[int, int]] = {}
    price_session: dt.date | None = None
    # Every instrument the file mentioned, whether or not anything usable
    # came with it. An instrument that appears with an undefined OI and no
    # quote is a real observation of "nothing here"; one that never appears
    # is a gap in the data. Collapsing those two makes the counts lie.
    seen: set[int] = set()

    for r in read_statistics(path):
        stat = int(r.stat_type)
        instrument = int(r.instrument_id)
        seen.add(instrument)
        ts = int(r.ts_event)
        when = dt.datetime.fromtimestamp(ts / 1e9, dt.timezone.utc).astimezone(_ET)

        if stat == STAT_OPEN_INTEREST:
            qty = int(r.quantity)
            if _defined(qty) and qty > 0:
                # Rule 2: published pre-open, describes the prior session.
                oi[instrument] = (float(qty), _previous_session(when.date()))
            continue

        if stat in (STAT_HIGHEST_BID, STAT_LOWEST_OFFER, STAT_CLOSE_PRICE):
            key = (instrument, stat)
            prev = latest.get(key)
            if prev is None or ts >= prev[0]:      # Rule 4: latest wins
                latest[key] = (ts, int(r.price))
            if price_session is None:
                price_session = when.date()

    out: dict[int, InstrumentStats] = {}
    for instrument in sorted(seen):
        oi_value, oi_date = oi.get(instrument, (None, None))
        out[instrument] = InstrumentStats(
            instrument_id=instrument,
            open_interest=oi_value,
            oi_session=oi_date,
            bid=_price(latest.get((instrument, STAT_HIGHEST_BID), (0, 0))[1]),
            offer=_price(latest.get((instrument, STAT_LOWEST_OFFER), (0, 0))[1]),
            close=_price(latest.get((instrument, STAT_CLOSE_PRICE), (0, 0))[1]),
            price_session=price_session,
        )
    return out


def _previous_session(day: dt.date) -> dt.date:
    """The trading day before `day`, ignoring holidays.

    ASSUMPTION (data availability): weekends only. A holiday will attribute
    open interest to a day the market was shut, which is visible as a gap
    rather than as a wrong number -- the store simply has no session there.
    Replace with an exchange calendar before this feeds anything live.
    """
    step = {0: 3, 6: 2}.get(day.weekday(), 1)    # Mon -> Fri, Sun -> Fri
    return day - dt.timedelta(days=step)


def session_summary(stats: dict[int, InstrumentStats]) -> dict:
    """Counts worth logging after every ingest."""
    with_oi = [s for s in stats.values() if s.open_interest]
    with_bid = [s for s in stats.values() if s.bid is not None]
    with_offer = [s for s in stats.values() if s.offer is not None]
    crossed = [s for s in stats.values() if s.crossed]
    oi_sessions = {s.oi_session for s in with_oi if s.oi_session}
    px_sessions = {s.price_session for s in stats.values() if s.price_session}
    return {
        "instruments": len(stats),
        "with_open_interest": len(with_oi),
        "total_open_interest": sum(s.open_interest for s in with_oi),
        "with_bid": len(with_bid),
        "with_offer": len(with_offer),
        "crossed": len(crossed),
        "oi_session": sorted(oi_sessions),
        "price_session": sorted(px_sessions),
    }


# ---------------------------------------------------------- definitions


@dataclass(frozen=True)
class Instrument:
    """What an instrument_id actually refers to."""

    instrument_id: int
    raw_symbol: str
    asset: str
    strike: float
    expiration: dt.date
    right: str          # "C" or "P"


def load_definitions(path: str | Path) -> dict[int, Instrument]:
    """One day's definition file -> {instrument_id: Instrument}.

    Strikes are 1e-9 fixed point like every other price. `instrument_class`
    is a single char C/P for vanilla options; anything else (spreads, legs)
    is skipped rather than guessed at.
    """
    import databento as db                      # presentation-layer import

    out: dict[int, Instrument] = {}
    for r in db.DBNStore.from_file(str(path)):
        klass = str(getattr(r, "instrument_class", ""))
        right = klass[-1].upper() if klass else ""
        if right not in ("C", "P"):
            continue
        strike_raw = int(r.strike_price)
        exp_raw = int(r.expiration)
        if not _defined(strike_raw) or not _defined(exp_raw):
            continue
        out[int(r.instrument_id)] = Instrument(
            instrument_id=int(r.instrument_id),
            raw_symbol=str(r.raw_symbol),
            asset=str(r.asset),
            strike=strike_raw * PRICE_SCALE,
            expiration=dt.datetime.fromtimestamp(
                exp_raw / 1e9, dt.timezone.utc).date(),
            right=right,
        )
    return out


def implied_spot(stats: dict[int, InstrumentStats],
                 defs: dict[int, Instrument],
                 as_of: dt.date,
                 r: float = 0.042,
                 max_dte: int = 60) -> float | None:
    """Recover the underlying from put-call parity.

    SPX definitions carry no index level, and Cboe charges $1k/month for a
    distinct index quote. Parity needs neither: for one expiry and strike,

        C - P = S*exp(-qT) - K*exp(-rT)

    so S is implied by the call/put price difference. For an index this is
    arguably the better number anyway -- it is the FORWARD the options are
    actually priced off, not the spot index print, and the two differ by
    carry.

    Estimated at the strike where |C - P| is smallest (nearest the forward,
    where both legs are liquid and the parity residual is least sensitive
    to a stale quote), using the nearest expiry with enough strikes.
    """
    import math

    by_expiry: dict[dt.date, dict[float, dict[str, float]]] = {}
    for iid, st in stats.items():
        d = defs.get(iid)
        mid = st.mid
        if d is None or mid is None or st.crossed:
            continue
        dte = (d.expiration - as_of).days
        if dte <= 0 or dte > max_dte:
            continue
        by_expiry.setdefault(d.expiration, {}).setdefault(
            d.strike, {})[d.right] = mid

    best: tuple[float, float] | None = None      # (abs(C-P), implied spot)
    for expiry, strikes in by_expiry.items():
        pairs = {k: v for k, v in strikes.items() if "C" in v and "P" in v}
        if len(pairs) < 3:
            continue
        T = max((expiry - as_of).days, 1) / 365.0
        for strike, legs in pairs.items():
            diff = legs["C"] - legs["P"]
            spot = diff + strike * math.exp(-r * T)
            if spot <= 0:
                continue
            if best is None or abs(diff) < best[0]:
                best = (abs(diff), spot)
    return best[1] if best else None


def build_chain(stats: dict[int, InstrumentStats],
                defs: dict[int, Instrument],
                session: dt.date,
                spot: float | None = None,
                source: str = "databento-opra"):
    """Join statistics and definitions into an OptionChain.

    NOTE ON WHICH SESSION THIS IS. Open interest in a file dated D belongs
    to D-1 (see rule 2), so `session` should be the OI's session, and the
    prices carried alongside are D's. They are one day apart by
    construction. This is recorded in `source` so a study can see it.
    """
    from ..options import OptionChain, OptionContract

    if spot is None:
        spot = implied_spot(stats, defs, session)
    if not spot or spot <= 0:
        return None

    contracts = []
    for iid, st in stats.items():
        d = defs.get(iid)
        if d is None or not st.open_interest:
            continue
        contracts.append(OptionContract(
            symbol=d.raw_symbol.strip() or str(iid),
            underlying=d.asset,
            expiry=d.expiration,
            strike=d.strike,
            right=d.right,
            bid=st.bid or 0.0,
            ask=st.offer or 0.0,
            last=st.close or 0.0,
            volume=0.0,
            open_interest=st.open_interest,
            # No vendor IV or greeks in this feed -- we solve and compute
            # our own, which is the design anyway.
            iv=0.0,
        ))
    if not contracts:
        return None

    return OptionChain(
        underlying=contracts[0].underlying,
        spot=spot,
        fetched_at=dt.datetime.combine(
            session, dt.time(16, 15), tzinfo=dt.timezone.utc),
        contracts=tuple(contracts),
        source=source,
        session_date=session,
    )
