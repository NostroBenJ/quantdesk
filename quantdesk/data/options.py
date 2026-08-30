"""Normalised options-chain schema, and the quality filters that keep it honest.

Feeds a GEX/positioning layer without any paid subscription. The schema is
vendor-neutral: CBOE today, a paid feed later, same shape either way.

TWO DESIGN RULES THAT MATTER
----------------------------
1. **Vendor greeks are stored separately from ours.** `vendor_gamma` is what the
   feed published; our own gamma comes from `black_scholes.py`. Keeping both is
   what makes the Level Check disagreement-detector possible - the same role the
   Unusual Whales `gex-levels` endpoint used to play. Two computations that
   agree is evidence. One silently overwriting the other is not, so nothing in
   this module ever copies a vendor greek into our field.

2. **Unusable contracts are labelled, not silently dropped.** A chain where
   10,482 of 14,100 contracts carry open interest is normal. Which 3,618 got
   excluded, and why, is information - and if a filter starts excluding half the
   book, that is a data incident you want to see rather than a quieter number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Flag, auto
from typing import Iterator, Sequence
from zoneinfo import ZoneInfo

#: OCC 21-character option symbol: root(6, padded) + YYMMDD + C/P + strike(8, /1000)
_OCC = re.compile(r"^(?P<root>[A-Z]+)(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")

#: ASSUMPTION (market microstructure): US listed options, regular hours.
EXCHANGE_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)

#: Calendar-day year. `T` is in years throughout; 1 calendar day = 1/365.
DAYS_PER_YEAR = 365.0

#: Moneyness beyond which a contract is treated as deep ITM and its own IV is
#: not trusted. 1% of spot is roughly one strike on SPY.
ITM_THRESHOLD = 0.01

#: Relative gap between a call's and a put's vendor gamma at the same strike
#: above which the feed is contradicting itself. Gamma is identical for the two
#: by parity, so anything beyond measurement noise is a data fault.
PARITY_TOLERANCE = 0.05


class ChainError(ValueError):
    """Raised when chain data violates an invariant we refuse to compute on."""


class Quality(Flag):
    """Why a contract is or is not usable. Flags combine."""

    OK = 0
    NO_BID = auto()
    """bid <= 0. The contract cannot be sold, and any IV inverted from a
    half-cent mid is noise, not a volatility."""

    NO_OPEN_INTEREST = auto()
    """No OI. Carries no positioning information, so it contributes nothing to
    GEX - but it can still be traded, so this is not the same as untradeable."""

    NO_IV = auto()
    """iv <= 0 or absent. Nothing that needs a volatility can use this row."""

    EXPIRED = auto()
    """Expiry is in the past relative to the chain's observation time."""

    CROSSED = auto()
    """bid > ask. A real and surprisingly common artifact in delayed composite
    feeds; it means the two sides came from different moments."""

    WIDE = auto()
    """Spread exceeds half the mid. Priceable in principle, but any cost model
    built on it is quoting an estimate of an estimate."""

    DEEP_ITM = auto()
    """In the money by more than one strike's worth of moneyness.

    ITM options carry nearly all intrinsic value and trade with wide spreads, so
    an IV inverted from their mid is dominated by the spread rather than by the
    market's view of volatility. Standard practice is to build a surface from
    the OTM wing at every strike - see `OptionChain.otm_iv_by_strike`."""

    PARITY_BREAK = auto()
    """The call and the put at this strike/expiry report inconsistent vendor
    gamma. Gamma is identical for a call and a put at the same strike and
    expiry - that is an identity, not an approximation. When a feed's own two
    numbers disagree, at least one of its IVs is unreliable and anything derived
    from it is too.

    Observed live in the CBOE SPY chain on 2026-08-21: at K=779 the put reported
    gamma 0.00010 against the call's 0.00510, a factor of 51, because the put's
    IV was inverted from a 15.37/18.10 quote. At the money the same feed agreed
    to within 1%."""


def parse_occ(symbol: str) -> tuple[str, date, str, float]:
    """Return (root, expiry, right, strike) from an OCC symbol.

    Raises rather than returning None: an unparseable symbol means the feed
    changed shape, and continuing past that produces a chain that is quietly
    missing whichever contracts stopped matching.
    """
    match = _OCC.match(symbol.strip().replace(" ", ""))
    if not match:
        raise ChainError(f"unparseable OCC symbol: {symbol!r}")
    raw = match.group("expiry")
    # ASSUMPTION (data availability): OCC YY is 2000-2099. Listed options do not
    # predate 2000 in any feed we use, and this breaks in 2100.
    expiry = date(2000 + int(raw[:2]), int(raw[2:4]), int(raw[4:6]))
    return (
        match.group("root"),
        expiry,
        match.group("right"),
        int(match.group("strike")) / 1000.0,
    )


@dataclass(frozen=True, slots=True)
class OptionContract:
    """One listed contract, normalised."""

    symbol: str
    underlying: str
    expiry: date
    strike: float
    right: str  # "C" or "P"
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: float = 0.0
    open_interest: float = 0.0
    iv: float = 0.0

    # Vendor-published greeks. NEVER read these as if they were ours - they exist
    # so `levels.compare()` can disagree with us out loud.
    vendor_delta: float | None = None
    vendor_gamma: float | None = None
    vendor_theta: float | None = None
    vendor_vega: float | None = None

    def __post_init__(self) -> None:
        if self.right not in ("C", "P"):
            raise ChainError(f"{self.symbol}: right must be C or P, got {self.right!r}")
        if self.strike <= 0:
            raise ChainError(f"{self.symbol}: non-positive strike {self.strike}")

    @property
    def is_call(self) -> bool:
        return self.right == "C"

    @property
    def mid(self) -> float:
        """Mid price. Falls back to `last` only when one side is missing, and
        returns 0.0 rather than inventing a price when neither is available."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last if self.last > 0 else 0.0

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)

    @property
    def half_spread(self) -> float:
        """Half the quoted spread, in price units.

        This is the number that turns the backtest cost model from an estimate
        into an observation. `SlippageParams.spread_vol_fraction` exists only
        because free bar data has no bid/ask; where a real quote exists, use it.
        """
        return self.spread / 2.0

    @property
    def relative_spread(self) -> float:
        m = self.mid
        return self.spread / m if m > 0 else float("inf")

    def time_to_expiry(self, as_of: date) -> float:
        """`T` in YEARS, on a 365 calendar-day basis.

        Expiry day itself returns 0.0, not a negative number - a contract at its
        own expiry has no time value left, and a negative T silently produces
        NaN greeks downstream.
        """
        return max(0, (self.expiry - as_of).days) / DAYS_PER_YEAR

    def is_itm(self, spot: float) -> bool:
        return self.strike < spot if self.is_call else self.strike > spot

    def moneyness(self, spot: float) -> float:
        """Signed distance from spot as a fraction, positive when in the money."""
        if spot <= 0:
            return 0.0
        raw = (spot - self.strike) / spot
        return raw if self.is_call else -raw

    def quality(self, as_of: date, spot: float | None = None) -> Quality:
        flags = Quality.OK
        if self.bid <= 0:
            flags |= Quality.NO_BID
        if self.open_interest <= 0:
            flags |= Quality.NO_OPEN_INTEREST
        if self.iv <= 0:
            flags |= Quality.NO_IV
        if self.expiry < as_of:
            flags |= Quality.EXPIRED
        if self.bid > 0 and self.ask > 0 and self.bid > self.ask:
            flags |= Quality.CROSSED
        if self.mid > 0 and self.relative_spread > 1.0:
            flags |= Quality.WIDE
        if spot is not None and self.moneyness(spot) > ITM_THRESHOLD:
            flags |= Quality.DEEP_ITM
        return flags

    def usable_for_gex(self, as_of: date) -> bool:
        """Contributes to dealer gamma exposure.

        Requires open interest (GEX is an OI-weighted quantity) and a live
        expiry. A no-bid contract still counts: worthless OTM puts carry real
        open interest and real dealer gamma, and excluding them would understate
        the wings - which is where the gamma that matters actually sits.
        """
        flags = self.quality(as_of)
        return not (flags & (Quality.NO_OPEN_INTEREST | Quality.EXPIRED))

    def usable_for_pricing(self, as_of: date, spot: float | None = None) -> bool:
        """Safe to invert for implied vol or fit a surface to.

        Stricter than `usable_for_gex`: needs a two-sided quote, a sane spread,
        and - when `spot` is supplied - it must not be deep in the money. The
        0.00 bid / 0.01 ask expiring puts in a real CBOE chain carry an "IV"
        inverted from a half-cent mid; the 15.37/18.10 ITM puts carry one
        dominated by a $2.73 spread. Both produce a smile with a vertical wing.

        Pass `spot`. It is optional only so that callers without a spot can still
        do the quote-quality checks, and omitting it silently keeps deep-ITM
        contracts in the pricing set.
        """
        flags = self.quality(as_of, spot)
        blocking = (
            Quality.NO_BID | Quality.NO_IV | Quality.EXPIRED
            | Quality.CROSSED | Quality.WIDE | Quality.DEEP_ITM
        )
        return not (flags & blocking)


@dataclass(frozen=True, slots=True)
class OptionChain:
    """A full chain snapshot at one instant."""

    underlying: str
    spot: float
    fetched_at: datetime
    contracts: tuple[OptionContract, ...]
    source: str = "unknown"

    #: The vendor's own timestamp, verbatim and UNPARSED.
    #: ASSUMPTION (data availability): CBOE's `timestamp` field carries no
    #: timezone and the docs do not state one. Rather than guess - which would
    #: shift every snapshot by hours and look like a real intraday effect - we
    #: record ours (`fetched_at`, unambiguous UTC) as the observation time and
    #: keep theirs as an opaque string until somebody confirms the zone.
    vendor_timestamp_raw: str | None = None

    #: Explicit trading date for these quotes. Set it when the exact session
    #: matters; otherwise `as_of_date` derives it in exchange time.
    session_date: date | None = None

    #: Underlying quote when the feed supplies one.
    spot_bid: float | None = None
    spot_ask: float | None = None

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ChainError(f"{self.underlying}: non-positive spot {self.spot}")
        if self.fetched_at.tzinfo is None:
            raise ChainError("fetched_at must be timezone-aware UTC")

    @property
    def as_of_date(self) -> date:
        """The TRADING date these quotes belong to. NOT the UTC calendar date.

        This distinction cost a real error and is worth spelling out. An
        end-of-day CBOE chain fetched at 04:29 UTC is 00:29 in New York: the UTC
        date has rolled over, but the quotes are the previous session's. Using
        the UTC date there understates every contract's DTE by one, and for a
        3-DTE option that is a 33% error in `T` - which showed up as our gamma
        coming in ~3x below CBOE's on short-dated OTM calls.

        Rule: convert to exchange time; if it is before the opening bell, the
        quotes belong to the previous session.

        ASSUMPTION (market calendar): the rollback subtracts one calendar day,
        so a pre-open Monday fetch resolves to Sunday rather than to Friday, and
        holidays are not handled. Set `session_date` explicitly when the exact
        trading date matters - which it does for anything under about a week to
        expiry.
        """
        if self.session_date is not None:
            return self.session_date
        local = self.fetched_at.astimezone(EXCHANGE_TZ)
        if local.time() < SESSION_OPEN:
            return local.date() - timedelta(days=1)
        return local.date()

    def __len__(self) -> int:
        return len(self.contracts)

    def __iter__(self) -> Iterator[OptionContract]:
        return iter(self.contracts)

    @property
    def expiries(self) -> list[date]:
        return sorted({c.expiry for c in self.contracts})

    def by_expiry(self, expiry: date) -> list[OptionContract]:
        return [c for c in self.contracts if c.expiry == expiry]

    def within_dte(self, max_dte: int) -> list[OptionContract]:
        """Contracts expiring within `max_dte` calendar days, expiry inclusive."""
        today = self.as_of_date
        return [c for c in self.contracts if 0 <= (c.expiry - today).days <= max_dte]

    def for_gex(self, max_dte: int | None = None) -> list[OptionContract]:
        today = self.as_of_date
        pool = self.within_dte(max_dte) if max_dte is not None else list(self.contracts)
        return [c for c in pool if c.usable_for_gex(today)]

    def for_pricing(self, max_dte: int | None = None) -> list[OptionContract]:
        today = self.as_of_date
        pool = self.within_dte(max_dte) if max_dte is not None else list(self.contracts)
        return [c for c in pool if c.usable_for_pricing(today, self.spot)]

    def otm_iv_by_strike(self, max_dte: int | None = None) -> dict[tuple[date, float], float]:
        """The OTM wing's implied vol at each (expiry, strike).

        THE RULE: below spot take the put, above spot take the call. An ITM
        option is nearly all intrinsic value, so its IV is an artifact of the
        bid/ask spread rather than a statement about volatility. Every surface
        worth fitting is built from the OTM wing for exactly this reason.

        This is not a refinement. In the live CBOE SPY chain the ITM put at
        K=779 reported IV 0.1343 against the OTM call's 0.0912 at the same
        strike and expiry - a 47% difference in the input to every greek.
        """
        out: dict[tuple[date, float], float] = {}
        today = self.as_of_date
        pool = self.within_dte(max_dte) if max_dte is not None else list(self.contracts)
        for contract in pool:
            if contract.iv <= 0 or contract.expiry < today:
                continue
            if contract.is_itm(self.spot):
                continue
            key = (contract.expiry, contract.strike)
            # At the money both wings qualify; prefer the tighter quote.
            existing = out.get(key)
            if existing is None:
                out[key] = contract.iv
            else:
                out[key] = min(existing, contract.iv) if existing > 0 else contract.iv
        return out

    def parity_breaks(
        self, max_dte: int | None = None, tolerance: float = PARITY_TOLERANCE
    ) -> list[tuple[date, float, float, float]]:
        """Strikes where the feed's own call and put gamma disagree.

        Returns (expiry, strike, call_gamma, put_gamma). Gamma is identical for
        a call and a put at the same strike and expiry - that is an identity, so
        every row returned here is a fault in the data, not in the market.

        Cheap, needs no external reference, and catches exactly the pocket of
        bad IV that would otherwise corrupt a GEX profile at those strikes.
        """
        pool = self.within_dte(max_dte) if max_dte is not None else list(self.contracts)
        by_strike: dict[tuple[date, float], dict[str, float]] = {}
        for contract in pool:
            if contract.vendor_gamma is None or contract.vendor_gamma <= 0:
                continue
            by_strike.setdefault((contract.expiry, contract.strike), {})[
                contract.right
            ] = contract.vendor_gamma

        breaks: list[tuple[date, float, float, float]] = []
        for (expiry, strike), sides in sorted(by_strike.items()):
            call, put = sides.get("C"), sides.get("P")
            if call is None or put is None:
                continue
            scale = max(call, put)
            if scale > 0 and abs(call - put) / scale > tolerance:
                breaks.append((expiry, strike, call, put))
        return breaks

    def quality_report(self) -> dict[str, int]:
        """Counts per exclusion reason. Print this on every ingest.

        A chain that suddenly loses half its open interest is a data incident,
        and the only way to notice is to have been looking at the number when it
        was normal.
        """
        today = self.as_of_date
        counts = {
            "total": len(self.contracts),
            "usable_for_gex": 0,
            "usable_for_pricing": 0,
        }
        for flag in Quality:
            if flag is not Quality.OK:
                counts[flag.name.lower()] = 0
        for contract in self.contracts:
            flags = contract.quality(today, self.spot)
            for flag in Quality:
                if flag is not Quality.OK and flags & flag:
                    counts[flag.name.lower()] += 1
            if contract.usable_for_gex(today):
                counts["usable_for_gex"] += 1
            if contract.usable_for_pricing(today, self.spot):
                counts["usable_for_pricing"] += 1
        counts["parity_break"] = len(self.parity_breaks())
        return counts

    def total_open_interest(self) -> float:
        return sum(c.open_interest for c in self.contracts)

    def observed_half_spread(self, max_dte: int = 7, width: float = 0.02) -> float | None:
        """Median half-spread of near-the-money contracts, in price units.

        The point of this function: `backtest/costs.py` estimates the spread from
        volatility because bar data has no quotes. When a chain IS available, the
        spread can be measured instead. Returns None rather than a default when
        no contract qualifies - a missing observation must not silently become a
        plausible-looking number.
        """
        lo, hi = self.spot * (1 - width), self.spot * (1 + width)
        halves = sorted(
            c.half_spread
            for c in self.for_pricing(max_dte)
            if lo <= c.strike <= hi and c.half_spread > 0
        )
        if not halves:
            return None
        mid = len(halves) // 2
        if len(halves) % 2:
            return halves[mid]
        return (halves[mid - 1] + halves[mid]) / 2.0


class OptionChainSource:
    """Adapter interface for chain data. Mirrors `data/source.py::BarSource`."""

    @property
    def name(self) -> str:
        return type(self).__name__

    def fetch_chain(self, underlying: str) -> OptionChain:
        raise NotImplementedError
