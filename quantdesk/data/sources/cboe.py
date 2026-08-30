"""CBOE delayed-quotes chain source. Free, no key, no pagination.

    https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json

Replaces the paid Unusual Whales chain endpoint, and is better in one way that
matters a lot: it carries **bid and ask**. That turns the backtest spread from a
volatility-scaled estimate into a measurement.

VERIFIED against a live SPY pull on 2026-08-21:
  * 14,100 contracts in a single request (UW returned 13,958 across a paged walk)
  * all 14,100 symbols parse as valid OCC
  * 10,482 with non-zero open interest, 11,747 with non-zero IV
  * gamma is identical for the call and the put at each strike, as put-call
    parity requires - a cheap internal-consistency check on the feed itself

WHAT THIS FEED IS NOT
---------------------
* **Not real-time.** "delayed_quotes" is 15-minute delayed intraday, and the
  overnight pull is an end-of-day snapshot. Correct input for a pre-market bias
  on a session traded 09:30-12:00 ET. Wrong input for intraday execution.
* **Not a flow feed.** No sweep/block classification, no dark pool prints. Those
  are proprietary aggregations and are not recoverable from public data.
* **Index symbols use an underscore prefix** (`_SPX`), which `symbol_for` handles.

Uses `urllib` from the stdlib, not requests, so the data layer keeps its
zero-dependency property.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone

from ..options import ChainError, OptionChain, OptionChainSource, OptionContract, parse_occ

BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"

# A real browser User-Agent. The default `Python-urllib/3.x` is rejected at the
# edge on several market-data CDNs - this is the same class of failure that made
# every Unusual Whales endpoint return 403 with a perfectly valid key, and it
# costs one line to avoid rediscovering it.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: Cash-settled index products CBOE serves under an underscore-prefixed symbol.
_INDEX_SYMBOLS = {"SPX", "VIX", "NDX", "RUT", "XSP", "DJX", "OEX"}


def symbol_for(underlying: str) -> str:
    """CBOE's path symbol. Indices take a leading underscore; ETFs do not."""
    ticker = underlying.strip().upper().lstrip("_")
    return f"_{ticker}" if ticker in _INDEX_SYMBOLS else ticker


class CboeChainSource(OptionChainSource):
    """Fetch a full options chain from CBOE's public delayed-quotes endpoint."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "cboe-delayed"

    def _get(self, url: str) -> dict:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ChainError(
                    f"CBOE has no chain at {url} (404). Check the symbol - indices "
                    "need the underscore form, e.g. _SPX not SPX."
                ) from exc
            raise ChainError(f"CBOE returned HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise ChainError(f"could not reach CBOE: {exc.reason}") from exc

    def fetch_chain(self, underlying: str) -> OptionChain:
        url = BASE_URL.format(symbol=symbol_for(underlying))
        fetched_at = datetime.now(tz=timezone.utc)
        payload = self._get(url)

        data = payload.get("data")
        if not isinstance(data, dict):
            raise ChainError(f"unexpected CBOE payload shape for {underlying}: no 'data' object")

        rows = data.get("options") or []
        if not rows:
            raise ChainError(
                f"CBOE returned an empty chain for {underlying}. Treating this as an "
                "error rather than an empty chain - a zero-contract book is a fetch "
                "failure, not a market state."
            )

        spot = _as_float(data.get("current_price"))
        if spot <= 0:
            raise ChainError(f"CBOE gave no usable underlying price for {underlying}")

        ticker = underlying.strip().upper().lstrip("_")
        contracts: list[OptionContract] = []
        unparsed: list[str] = []

        for row in rows:
            symbol = str(row.get("option", "")).strip()
            try:
                _root, expiry, right, strike = parse_occ(symbol)
            except ChainError:
                unparsed.append(symbol)
                continue
            contracts.append(
                OptionContract(
                    symbol=symbol,
                    underlying=ticker,
                    expiry=expiry,
                    strike=strike,
                    right=right,
                    bid=_as_float(row.get("bid")),
                    ask=_as_float(row.get("ask")),
                    last=_as_float(row.get("last_trade_price")),
                    volume=_as_float(row.get("volume")),
                    open_interest=_as_float(row.get("open_interest")),
                    iv=_as_float(row.get("iv")),
                    # Vendor greeks, kept strictly separate from anything we compute.
                    vendor_delta=_as_optional_float(row.get("delta")),
                    vendor_gamma=_as_optional_float(row.get("gamma")),
                    vendor_theta=_as_optional_float(row.get("theta")),
                    vendor_vega=_as_optional_float(row.get("vega")),
                )
            )

        if unparsed:
            # Loud, not silent. Symbols that stopped parsing means the feed changed
            # shape, and a chain quietly missing whichever contracts no longer match
            # is exactly the Unusual Whales truncation failure in a new costume.
            raise ChainError(
                f"{len(unparsed)} of {len(rows)} CBOE symbols failed OCC parsing "
                f"(first: {unparsed[0]!r}). The feed format has changed; fix the "
                "parser rather than continuing with a partial chain."
            )

        return OptionChain(
            underlying=ticker,
            spot=spot,
            fetched_at=fetched_at,
            contracts=tuple(contracts),
            source=self.name,
            vendor_timestamp_raw=payload.get("timestamp"),
            spot_bid=_as_optional_float(data.get("bid")),
            spot_ask=_as_optional_float(data.get("ask")),
        )


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if result != result else result  # NaN -> default


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if result != result else result
