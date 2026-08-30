"""The DataShop historical reader.

No network, no purchased data. Fixtures are written to the published field
spec, so these tests fail if the reader drifts from the layout rather than
if Cboe changes their prices.

The tests worth reading are the ones about the two clocks: the file carries
15:45 quotes AND end-of-day quotes on the same row, and picking the wrong
pair silently gives you greeks from one time and prices from another.
"""

from __future__ import annotations

import gzip
from datetime import date

import pytest

from quantdesk.data.chain_store import ChainStore
from quantdesk.data.levels import gamma_profile, zero_gamma
from quantdesk.data.options import ChainError
from quantdesk.data.sources.cboe_datashop import (
    has_calcs,
    ingest,
    load_chains,
    read_rows,
)

HEADER = (
    "underlying_symbol,quote_date,root,expiration,strike,option_type,"
    "open,high,low,close,trade_volume,"
    "bid_size_1545,bid_1545,ask_size_1545,ask_1545,"
    "underlying_bid_1545,underlying_ask_1545,active_underlying_price_1545,"
    "implied_volatility_1545,delta_1545,gamma_1545,theta_1545,vega_1545,rho_1545,"
    "bid_size_eod,bid_eod,ask_size_eod,ask_eod,"
    "underlying_bid_eod,underlying_ask_eod,vwap,open_interest"
)


def row(strike=770.0, right="C", quote="2026-08-20", expiry="2026-09-18",
        oi=1000.0, iv=0.18, gamma=0.0123, bid1545=5.00, ask1545=5.10,
        bid_eod=4.00, ask_eod=4.10, spot1545=762.60, root="SPY",
        underlying="SPY", volume=250.0):
    return (
        f"{underlying},{quote},{root},{expiry},{strike},{right},"
        f"5.5,5.9,4.8,5.05,{volume},"
        f"10,{bid1545},12,{ask1545},762.55,762.65,{spot1545},"
        f"{iv},0.42,{gamma},-0.25,0.31,0.11,"
        f"8,{bid_eod},9,{ask_eod},761.90,762.00,5.02,{oi}"
    )


def write(tmp_path, rows, name="spy.csv", header=HEADER, gz=False):
    text = header + "\n" + "\n".join(rows) + "\n"
    path = tmp_path / (name + (".gz" if gz else ""))
    if gz:
        path.write_bytes(gzip.compress(text.encode("utf-8")))
    else:
        path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ reading


def test_reads_a_basic_file(tmp_path):
    path = write(tmp_path, [row()])
    chains = load_chains(path)
    assert len(chains) == 1
    chain = chains[0]
    assert chain.underlying == "SPY"
    assert chain.as_of_date == date(2026, 8, 20)
    assert chain.spot == pytest.approx(762.60)
    assert len(chain) == 1
    c = chain.contracts[0]
    assert c.strike == pytest.approx(770.0)
    assert c.right == "C"
    assert c.open_interest == pytest.approx(1000.0)
    assert c.iv == pytest.approx(0.18)
    assert c.vendor_gamma == pytest.approx(0.0123)


def test_gzip_is_transparent(tmp_path):
    plain = load_chains(write(tmp_path, [row()], name="a.csv"))
    zipped = load_chains(write(tmp_path, [row()], name="b.csv", gz=True))
    assert plain[0].spot == zipped[0].spot
    assert len(plain[0]) == len(zipped[0])


def test_monthly_file_splits_into_one_chain_per_session(tmp_path):
    rows = [row(quote="2026-08-20", strike=770.0),
            row(quote="2026-08-20", strike=775.0),
            row(quote="2026-08-21", strike=770.0)]
    chains = load_chains(write(tmp_path, rows))
    assert [c.as_of_date for c in chains] == [date(2026, 8, 20),
                                              date(2026, 8, 21)]
    assert len(chains[0]) == 2
    assert len(chains[1]) == 1


def test_a_missing_required_column_raises_rather_than_reading_zeros(tmp_path):
    broken = HEADER.replace("open_interest", "oi_renamed_by_vendor")
    path = write(tmp_path, [row()], header=broken)
    with pytest.raises(ChainError, match="missing required columns"):
        list(read_rows(path))


def test_blank_cells_do_not_crash_the_loader(tmp_path):
    """A 14-year file will contain blanks; they are a real state."""
    r = row()
    parts = r.split(",")
    parts[18] = ""          # implied_volatility_1545
    parts[20] = ""          # gamma_1545
    chains = load_chains(write(tmp_path, [",".join(parts)]))
    c = chains[0].contracts[0]
    assert c.iv == 0.0
    assert c.vendor_gamma is None       # absent stays absent, never 0.0


def test_other_underlyings_can_be_filtered_out(tmp_path):
    rows = [row(underlying="SPY"), row(underlying="QQQ", root="QQQ")]
    assert len(load_chains(write(tmp_path, rows), underlying="SPY")) == 1
    assert len(load_chains(write(tmp_path, rows))) == 2


def test_adjusted_roots_do_not_collide_at_the_same_strike(tmp_path):
    """SPY1 after a corporate action is a different deliverable from SPY."""
    rows = [row(root="SPY"), row(root="SPY1")]
    chain = load_chains(write(tmp_path, rows))[0]
    assert len({c.symbol for c in chain}) == 2


def test_session_with_no_usable_spot_is_skipped_not_invented(tmp_path):
    r = row().split(",")
    r[17] = "0"     # active_underlying_price_1545
    r[15] = "0"     # underlying_bid_1545
    r[16] = "0"     # underlying_ask_1545
    assert load_chains(write(tmp_path, [",".join(r)])) == []


# ------------------------------------------------------------- the two clocks


def test_snapshot_selects_which_quotes_are_used(tmp_path):
    path = write(tmp_path, [row(bid1545=5.00, ask1545=5.10,
                                bid_eod=4.00, ask_eod=4.10)])
    at_1545 = load_chains(path, snapshot="1545")[0].contracts[0]
    at_eod = load_chains(path, snapshot="eod")[0].contracts[0]
    assert at_1545.bid == pytest.approx(5.00)
    assert at_1545.ask == pytest.approx(5.10)
    assert at_eod.bid == pytest.approx(4.00)
    assert at_eod.ask == pytest.approx(4.10)


def test_open_interest_is_identical_across_snapshots(tmp_path):
    """OI is a row-level field. Neither snapshot owns it, and it must not
    change with the choice - if it ever does, the reader is wrong."""
    path = write(tmp_path, [row(oi=4242.0)])
    for snap in ("1545", "eod"):
        assert load_chains(path, snapshot=snap)[0].contracts[0].open_interest \
            == pytest.approx(4242.0)


def test_source_records_which_snapshot_was_read(tmp_path):
    """A study spanning two snapshot choices must be able to see that."""
    path = write(tmp_path, [row()])
    assert load_chains(path, snapshot="1545")[0].source.endswith("1545")
    assert load_chains(path, snapshot="eod")[0].source.endswith("eod")


def test_unknown_snapshot_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_chains(write(tmp_path, [row()]), snapshot="noon")


def test_has_calcs_detects_the_addon(tmp_path):
    assert has_calcs(write(tmp_path, [row()], name="with.csv"))
    no_calcs_header = ",".join(
        c for c in HEADER.split(",")
        if not c.endswith("_1545") or c.startswith(("bid", "ask", "underlying")))
    thin = write(tmp_path, [], name="without.csv", header=no_calcs_header)
    assert not has_calcs(thin)


# ------------------------------------------------------- end to end into GEX


def test_a_loaded_file_produces_a_gamma_profile(tmp_path):
    """The point of all of this: purchased history reaching levels.py."""
    rows = [row(strike=k, right="C", oi=5000.0) for k in (770.0, 775.0, 780.0)]
    rows += [row(strike=k, right="P", oi=5000.0) for k in (745.0, 750.0, 755.0)]
    chain = load_chains(write(tmp_path, rows))[0]

    prof = gamma_profile(chain)
    assert prof.contracts_used == 6
    assert len(prof.strikes) == 6
    assert all(s.gex > 0 for s in prof.strikes if s.strike > chain.spot)
    assert all(s.gex < 0 for s in prof.strikes if s.strike < chain.spot)
    assert zero_gamma(chain) is not None


def test_ingest_writes_sessions_into_the_store(tmp_path):
    rows = [row(quote="2026-08-20"), row(quote="2026-08-21")]
    path = write(tmp_path, rows)
    with ChainStore(tmp_path / "chains.sqlite") as store:
        summary = ingest([path], store, underlying="SPY")
        assert summary["files"] == 1
        assert summary["sessions"] == 2
        assert store.sessions("SPY") == ["2026-08-20", "2026-08-21"]
        back = store.read_chain("SPY", "2026-08-20")
        assert back is not None
        assert back.contracts[0].vendor_gamma == pytest.approx(0.0123)


def test_ingest_records_bad_files_instead_of_dying(tmp_path):
    good = write(tmp_path, [row()], name="good.csv")
    bad = write(tmp_path, [row()], name="bad.csv",
                header=HEADER.replace("strike", "strike_renamed"))
    with ChainStore(tmp_path / "chains.sqlite") as store:
        summary = ingest([good, bad], store)
    assert summary["files"] == 1
    assert len(summary["skipped"]) == 1
    assert "bad.csv" in summary["skipped"][0]
