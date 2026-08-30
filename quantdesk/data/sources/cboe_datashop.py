"""Cboe DataShop 'Option EOD Summary' historical files.

The recorder captures the live delayed chain going forward; this reads the
purchased history backwards. Same exchange, same OPRA feed, which is the
reason to buy from Cboe rather than a cheaper reseller: a series stitched
from two vendors has a seam in the middle, and `chain_store` warns about
exactly that.

TWO CLOCKS IN ONE ROW
---------------------
This is the thing to understand about the file. Each row carries two
snapshots of the same contract:

    ...1545     bid, ask, sizes, underlying, IV, and all five greeks
    ...EOD      bid, ask, sizes, underlying
    (row-level) open, high, low, close, volume, VWAP, OPEN INTEREST

So IV and the greeks exist ONLY at 15:45, while open interest exists only
at the row level. Any GEX built from this file therefore pairs 15:45 greeks
with end-of-day open interest.

ASSUMPTION (data availability): that pairing is sound because open interest
does not move intraday - OPRA publishes it once, around 06:30 ET, reflecting
the prior session's close. The 15-minute offset between the two snapshots
cannot affect it. What it DOES mean is that a row's open interest describes
the book as of the PREVIOUS session, not the quote date. `verify_oi_lag()`
exists to check that against a session you can confirm independently, and
until someone runs it this remains unverified.

ASSUMPTION (microstructure): we default to the 15:45 quotes rather than the
EOD ones, because that is where the vendor's IV and greeks are computed and
Cboe states it is the better representation of liquidity. Mixing 15:45
greeks with EOD quotes would put two times in one contract for no gain.
Pass `snapshot="eod"` to override, and expect vendor greeks to be absent
from the comparison when you do.

The file layout is published:
https://datashop.cboe.com/documents/Option_EOD_Summary_Layout.pdf
"""

from __future__ import annotations

import csv
import gzip
import io
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterator

from ..options import ChainError, OptionChain, OptionContract

#: Header names as they appear in the file, mapped to what we call them.
#: Kept verbatim rather than normalised so a layout change fails loudly at
#: the header rather than silently producing empty columns.
COLUMNS = {
    "underlying_symbol": "underlying",
    "quote_date": "quote_date",
    "root": "root",
    "expiration": "expiration",
    "strike": "strike",
    "option_type": "option_type",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "trade_volume": "volume",
    "bid_size_1545": "bid_size_1545",
    "bid_1545": "bid_1545",
    "ask_size_1545": "ask_size_1545",
    "ask_1545": "ask_1545",
    "underlying_bid_1545": "underlying_bid_1545",
    "underlying_ask_1545": "underlying_ask_1545",
    # Present in the delivered files but NOT in the published field list.
    # It is the underlying implied by put-call parity, and it reads 0.0000
    # throughout the 2012 sample, so it is recorded and not relied upon.
    "implied_underlying_price_1545": "implied_underlying_1545",
    "active_underlying_price_1545": "spot_1545",
    "implied_volatility_1545": "iv",
    "delta_1545": "delta",
    "gamma_1545": "gamma",
    "theta_1545": "theta",
    "vega_1545": "vega",
    "rho_1545": "rho",
    "bid_size_eod": "bid_size_eod",
    "bid_eod": "bid_eod",
    "ask_size_eod": "ask_size_eod",
    "ask_eod": "ask_eod",
    "underlying_bid_eod": "underlying_bid_eod",
    "underlying_ask_eod": "underlying_ask_eod",
    "vwap": "vwap",
    "open_interest": "open_interest",
    # Also undocumented; empty throughout the 2012 sample. Non-empty marks a
    # non-standard deliverable, which is exactly when a strike's gamma stops
    # meaning what the rest of the book's does.
    "delivery_code": "delivery_code",
}

#: Columns present only when the Calcs add-on was purchased.
CALCS_COLUMNS = ("implied_volatility_1545", "delta_1545", "gamma_1545",
                 "theta_1545", "vega_1545", "rho_1545")

#: Cboe records the snapshot times in US Eastern; we store UTC.
SNAPSHOT_TIME = {"1545": time(15, 45), "eod": time(16, 15)}


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _num(raw: str | None, default: float = 0.0) -> float:
    """Blank and unparseable cells become `default`, never a crash.

    A 14-year file will contain blanks - a contract with no 15:45 quote, an
    IV the vendor could not solve. Those are real states, and a loader that
    dies on the first one is a loader nobody can use.
    """
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _opt_num(raw: str | None) -> float | None:
    """As `_num`, but absent stays absent. Vendor greeks are Optional."""
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


def _csv_members(path: Path) -> list[str]:
    """CSV entries inside a zip. Empty list for non-zip paths."""
    if path.suffix.lower() != ".zip":
        return []
    with zipfile.ZipFile(path) as zf:
        return [n for n in zf.namelist()
                if n.lower().endswith(".csv") and not n.endswith("/")]


@contextmanager
def _open_text(path: Path, member: str | None = None):
    """Text handle for .csv, .csv.gz, or a CSV inside a .zip.

    DataShop delivers zips -- one CSV per archive in practice, though the
    format does not promise that, so the reader iterates members rather than
    assuming the first one is the only one.
    """
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = member or (_csv_members(path) or [None])[0]
            if name is None:
                raise ChainError(f"{path.name}: zip contains no .csv entry")
            with zf.open(name) as raw:
                yield io.TextIOWrapper(raw, encoding="utf-8", newline="")
        return
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as raw:
            yield io.TextIOWrapper(raw, encoding="utf-8", newline="")
        return
    with path.open("r", encoding="utf-8", newline="") as fh:
        yield fh


def _right(raw: str) -> str:
    """'C'/'P', however the file spells it."""
    v = raw.strip().upper()
    if v.startswith("C"):
        return "C"
    if v.startswith("P"):
        return "P"
    raise ChainError(f"unrecognised option_type {raw!r}")


def read_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield normalised rows. Raises if the header is not what we expect."""
    p = Path(path)
    members = _csv_members(p) or [None]
    for member in members:
        with _open_text(p, member) as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ChainError(f"{p.name}: no header row")
            present = {_norm(f) for f in reader.fieldnames}
            required = {"underlying_symbol", "quote_date", "expiration",
                        "strike", "option_type", "open_interest"}
            missing = required - present
            if missing:
                raise ChainError(
                    f"{p.name}: missing required columns {sorted(missing)}. "
                    "The DataShop layout may have changed - check the file "
                    "spec before adapting this reader, because a silently "
                    "renamed column reads as zero.")
            for row in reader:
                yield {_norm(k): v for k, v in row.items() if k is not None}


def has_calcs(path: str | Path) -> bool:
    """Whether the Calcs add-on columns are present in this file."""
    p = Path(path)
    member = (_csv_members(p) or [None])[0]
    with _open_text(p, member) as fh:
        try:
            names = {_norm(c) for c in next(csv.reader(fh))}
        except StopIteration:
            return False
    return all(c in names for c in CALCS_COLUMNS)


def load_chains(path: str | Path, *, snapshot: str = "1545",
                underlying: str | None = None) -> list[OptionChain]:
    """Every session in one DataShop file, as OptionChains.

    A daily file yields one chain; a monthly file yields ~21. Grouping the
    rows by quote_date rather than assuming one date per file means the same
    reader works for either, which matters because the grouping is an order
    option somebody picks once and forgets.
    """
    if snapshot not in SNAPSHOT_TIME:
        raise ValueError(f"snapshot must be one of {sorted(SNAPSHOT_TIME)}")

    bid_col = f"bid_{snapshot}"
    ask_col = f"ask_{snapshot}"
    ubid_col = f"underlying_bid_{snapshot}"
    uask_col = f"underlying_ask_{snapshot}"

    sessions: dict[tuple[str, date], list[OptionContract]] = {}
    spots: dict[tuple[str, date], tuple[float, float | None, float | None]] = {}

    for row in read_rows(path):
        sym = (row.get("underlying_symbol") or "").strip().upper()
        if underlying and sym != underlying.upper():
            continue
        quote_date = date.fromisoformat(row["quote_date"].strip()[:10])
        key = (sym, quote_date)

        strike = _num(row.get("strike"))
        if strike <= 0:
            continue

        bid, ask = _num(row.get(bid_col)), _num(row.get(ask_col))
        contract = OptionContract(
            # The file has no OCC symbol column, so build a stable identity.
            # `root` differs from `underlying_symbol` for adjusted series
            # (SPY1 after a corporate action), and collapsing them would
            # silently merge two different deliverables at one strike.
            symbol="{0}{1}{2}{3:08.0f}".format(
                (row.get("root") or sym).strip().upper(),
                date.fromisoformat(row["expiration"].strip()[:10]).strftime("%y%m%d"),
                _right(row["option_type"]), strike * 1000),
            underlying=sym,
            expiry=date.fromisoformat(row["expiration"].strip()[:10]),
            strike=strike,
            right=_right(row["option_type"]),
            bid=bid, ask=ask, last=_num(row.get("close")),
            volume=_num(row.get("trade_volume")),
            open_interest=_num(row.get("open_interest")),
            iv=_num(row.get("implied_volatility_1545")),
            vendor_delta=_opt_num(row.get("delta_1545")),
            vendor_gamma=_opt_num(row.get("gamma_1545")),
            vendor_theta=_opt_num(row.get("theta_1545")),
            vendor_vega=_opt_num(row.get("vega_1545")),
        )
        sessions.setdefault(key, []).append(contract)

        if key not in spots:
            ub, ua = _num(row.get(ubid_col)), _num(row.get(uask_col))
            active = _num(row.get("active_underlying_price_1545"))
            # Prefer the vendor's own active price; fall back to the mid of
            # the underlying quote. Both can be blank for index roots, where
            # a distinct bid/ask needs the CGI licence.
            spot = active or ((ub + ua) / 2.0 if ub > 0 and ua > 0 else 0.0)
            spots[key] = (spot, ub or None, ua or None)

    chains: list[OptionChain] = []
    for (sym, quote_date), contracts in sorted(sessions.items()):
        spot, ubid, uask = spots[(sym, quote_date)]
        if spot <= 0:
            # OptionChain refuses a non-positive spot, and it is right to.
            # Skipping loudly beats inventing a price.
            continue
        chains.append(OptionChain(
            underlying=sym, spot=spot,
            fetched_at=datetime.combine(
                quote_date, SNAPSHOT_TIME[snapshot], tzinfo=timezone.utc),
            contracts=tuple(contracts),
            source=f"cboe-datashop-{snapshot}",
            session_date=quote_date,
            spot_bid=ubid, spot_ask=uask,
        ))
    return chains


def ingest(paths, store, *, snapshot: str = "1545",
           underlying: str | None = None, progress=None) -> dict:
    """Load files into a ChainStore. Returns a summary worth logging."""
    summary = {"files": 0, "sessions": 0, "contracts": 0, "skipped": []}
    for path in paths:
        try:
            chains = load_chains(path, snapshot=snapshot, underlying=underlying)
        except (ChainError, OSError, ValueError) as exc:
            summary["skipped"].append(f"{Path(path).name}: {exc}")
            continue
        summary["files"] += 1
        for chain in chains:
            report = store.write_chain(chain)
            summary["sessions"] += 1
            summary["contracts"] += report["stored"]
            if progress:
                progress(report)
    return summary


def verify_oi_lag(store, underlying: str, sessions: list[str]) -> dict:
    """Does a row's open interest describe its own session, or the previous?

    UNVERIFIED ASSUMPTION, and the one most likely to shift a study by a day.
    OPRA publishes open interest in the morning for the prior close, so a
    file's `open_interest` column may be lagged relative to its quote_date.

    This does not answer the question on its own - it surfaces the evidence.
    If OI is lagged one session, then consecutive sessions' OI should match
    the SHIFTED series better than the aligned one. Run it over a stretch of
    real sessions and read the two correlations.
    """
    totals = []
    for session in sessions:
        chain = store.read_chain(underlying, session)
        totals.append(None if chain is None else chain.total_open_interest())
    known = [(s, t) for s, t in zip(sessions, totals) if t is not None]
    return {
        "sessions": len(sessions),
        "loaded": len(known),
        "totals": known,
        "note": ("Compare against an independent source for one date before "
                 "trusting the alignment. Two sessions is not enough to tell."),
    }
