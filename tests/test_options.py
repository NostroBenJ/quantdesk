"""Options chain schema, quality filters, and the CBOE symbol mapping.

No network. The CBOE adapter's live behaviour is smoke-checked by
`scripts/check_chain.py`; what is tested here is the parsing and filtering
logic, which is where the silent-wrong-answer bugs live.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from quantdesk.data.options import (
    ChainError,
    OptionChain,
    OptionContract,
    Quality,
    parse_occ,
)
from quantdesk.data.sources.cboe import symbol_for


def contract(
    strike: float,
    right: str = "C",
    expiry: date = date(2026, 9, 18),
    bid: float = 1.00,
    ask: float = 1.10,
    oi: float = 100.0,
    iv: float = 0.20,
    volume: float = 10.0,
) -> OptionContract:
    yy = expiry.strftime("%y%m%d")
    symbol = f"SPY{yy}{right}{int(strike * 1000):08d}"
    return OptionContract(
        symbol=symbol,
        underlying="SPY",
        expiry=expiry,
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        volume=volume,
        open_interest=oi,
        iv=iv,
    )


def chain(contracts, spot: float = 760.0, when: date = date(2026, 8, 21)) -> OptionChain:
    """A chain whose session date is `when`, stated rather than inferred.

    `session_date` is set explicitly because 12:00 UTC is 08:00 in New York -
    before the opening bell - so the derived date would legitimately roll back
    to the previous session and every DTE in these tests would shift by one.
    That inference is correct behaviour (see the session-date tests below); a
    fixture just should not depend on it.
    """
    return OptionChain(
        underlying="SPY",
        spot=spot,
        fetched_at=datetime(when.year, when.month, when.day, 12, 0, tzinfo=timezone.utc),
        contracts=tuple(contracts),
        source="test",
        session_date=when,
    )


# ------------------------------------------------------------------ OCC parsing


def test_parse_occ_roundtrip():
    root, expiry, right, strike = parse_occ("SPY260918C00760000")
    assert root == "SPY"
    assert expiry == date(2026, 9, 18)
    assert right == "C"
    assert strike == pytest.approx(760.0)


def test_parse_occ_handles_fractional_strikes():
    _, _, _, strike = parse_occ("SPY260918P00762500")
    assert strike == pytest.approx(762.5)


def test_parse_occ_raises_rather_than_returning_none():
    """A silently-skipped symbol is a silently-truncated chain."""
    with pytest.raises(ChainError, match="unparseable"):
        parse_occ("NOT-AN-OCC-SYMBOL")


# ------------------------------------------------------------------- contracts


def test_mid_prefers_two_sided_quote():
    c = contract(760, bid=2.00, ask=2.50)
    assert c.mid == pytest.approx(2.25)
    assert c.spread == pytest.approx(0.50)
    assert c.half_spread == pytest.approx(0.25)


def test_mid_returns_zero_rather_than_inventing_a_price():
    c = OptionContract(
        symbol="SPY260918C00760000", underlying="SPY", expiry=date(2026, 9, 18),
        strike=760.0, right="C", bid=0.0, ask=0.0, last=0.0,
    )
    assert c.mid == 0.0
    assert c.relative_spread == float("inf")


def test_time_to_expiry_is_years_on_365_basis():
    c = contract(760, expiry=date(2026, 9, 18))
    t = c.time_to_expiry(date(2026, 8, 21))
    assert t == pytest.approx(28 / 365.0)


def test_time_to_expiry_never_goes_negative():
    """A negative T produces NaN greeks downstream, silently."""
    c = contract(760, expiry=date(2026, 8, 1))
    assert c.time_to_expiry(date(2026, 8, 21)) == 0.0


def test_expiry_day_itself_is_zero_not_negative():
    c = contract(760, expiry=date(2026, 8, 21))
    assert c.time_to_expiry(date(2026, 8, 21)) == 0.0


def test_right_must_be_call_or_put():
    with pytest.raises(ChainError, match="right must be"):
        OptionContract(
            symbol="X", underlying="SPY", expiry=date(2026, 9, 18),
            strike=760.0, right="X",
        )


# --------------------------------------------------------------------- quality


def test_worthless_expiring_put_is_excluded_from_pricing_but_kept_for_gex():
    """The exact shape seen in the real CBOE chain: 0.00 bid / 0.01 ask.

    Its 'IV' is inverted from a half-cent mid and is noise - so it must not feed
    a surface fit. But it carries 13,814 contracts of real open interest and
    real dealer gamma, so excluding it from GEX would understate the wings,
    which is where the gamma that matters actually sits.
    """
    today = date(2026, 8, 21)
    c = contract(760, right="P", expiry=date(2026, 8, 21), bid=0.0, ask=0.01, oi=13_814, iv=0.26)
    flags = c.quality(today)

    assert flags & Quality.NO_BID
    assert not (flags & Quality.NO_OPEN_INTEREST)
    assert c.usable_for_gex(today) is True
    assert c.usable_for_pricing(today) is False


def test_zero_open_interest_is_excluded_from_gex_but_still_tradeable():
    today = date(2026, 8, 21)
    c = contract(760, oi=0.0)
    assert c.quality(today) & Quality.NO_OPEN_INTEREST
    assert c.usable_for_gex(today) is False
    assert c.usable_for_pricing(today) is True


def test_crossed_quote_is_flagged():
    today = date(2026, 8, 21)
    c = contract(760, bid=2.00, ask=1.50)
    assert c.quality(today) & Quality.CROSSED
    assert c.usable_for_pricing(today) is False


def test_expired_contract_is_excluded_everywhere():
    today = date(2026, 8, 21)
    c = contract(760, expiry=date(2026, 8, 20))
    assert c.quality(today) & Quality.EXPIRED
    assert c.usable_for_gex(today) is False
    assert c.usable_for_pricing(today) is False


def test_wide_spread_is_flagged():
    today = date(2026, 8, 21)
    c = contract(760, bid=0.10, ask=1.00)  # spread 0.90 on a mid of 0.55
    assert c.quality(today) & Quality.WIDE
    assert c.usable_for_pricing(today) is False


# ----------------------------------------------------------------------- chain


def test_chain_rejects_naive_timestamp():
    with pytest.raises(ChainError, match="timezone-aware"):
        OptionChain(
            underlying="SPY", spot=760.0,
            fetched_at=datetime(2026, 8, 21, 12, 0),
            contracts=(contract(760),),
        )


def test_chain_rejects_nonpositive_spot():
    with pytest.raises(ChainError, match="spot"):
        OptionChain(
            underlying="SPY", spot=0.0,
            fetched_at=datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc),
            contracts=(contract(760),),
        )


def test_within_dte_is_inclusive_of_both_ends():
    c = chain([
        contract(760, expiry=date(2026, 8, 21)),   # 0 DTE
        contract(761, expiry=date(2026, 8, 28)),   # 7 DTE
        contract(762, expiry=date(2026, 9, 18)),   # 28 DTE
    ])
    assert len(c.within_dte(0)) == 1
    assert len(c.within_dte(7)) == 2
    assert len(c.within_dte(28)) == 3


def test_quality_report_counts_every_reason():
    c = chain([
        contract(760),                                        # clean
        contract(761, oi=0.0),                                # no OI
        contract(762, bid=0.0, ask=0.01),                     # no bid
        contract(763, expiry=date(2026, 8, 1)),               # expired
    ])
    report = c.quality_report()
    assert report["total"] == 4
    assert report["no_open_interest"] == 1
    assert report["no_bid"] == 1
    assert report["expired"] == 1
    assert report["usable_for_gex"] == 2   # clean + no-bid (OI intact, live)
    assert report["usable_for_pricing"] == 2  # clean + zero-OI


def test_observed_half_spread_measures_rather_than_estimates():
    """The whole point: with quotes present, the spread is measured, not modelled."""
    c = chain([
        contract(755, bid=2.00, ask=2.20, expiry=date(2026, 8, 24)),  # half 0.10
        contract(760, bid=3.00, ask=3.40, expiry=date(2026, 8, 24)),  # half 0.20
        contract(765, bid=1.00, ask=1.60, expiry=date(2026, 8, 24)),  # half 0.30
    ], spot=760.0)
    assert c.observed_half_spread(max_dte=7, width=0.02) == pytest.approx(0.20)


def test_observed_half_spread_returns_none_rather_than_a_default():
    """A missing observation must never become a plausible-looking number."""
    c = chain([contract(400, expiry=date(2026, 8, 24))], spot=760.0)
    assert c.observed_half_spread(max_dte=7) is None


def test_expiries_are_sorted_and_deduplicated():
    c = chain([
        contract(760, expiry=date(2026, 9, 18)),
        contract(761, expiry=date(2026, 8, 28)),
        contract(762, expiry=date(2026, 9, 18)),
    ])
    assert c.expiries == [date(2026, 8, 28), date(2026, 9, 18)]


# ------------------------------------------------------------------ CBOE symbol


@pytest.mark.parametrize(
    "given,expected",
    [("SPY", "SPY"), ("spy", "SPY"), ("SPX", "_SPX"), ("_SPX", "_SPX"), ("VIX", "_VIX")],
)
def test_index_symbols_get_the_underscore_prefix(given, expected):
    assert symbol_for(given) == expected


# ------------------------------------------------- session date and OTM wing
# These four tests exist because the gamma cross-check against CBOE's published
# greeks found two real bugs. Both are now pinned.


def test_session_date_rolls_back_for_an_overnight_fetch():
    """00:29 ET on the 21st is still the 20th session.

    Taking the UTC date here understates DTE by one for every contract. On a
    3-DTE option that is a 33% error in T, and it showed up as our gamma coming
    in ~3x below CBOE's on short-dated OTM calls.
    """
    c = contract(760)
    overnight = OptionChain(
        underlying="SPY", spot=762.6,
        fetched_at=datetime(2026, 8, 21, 4, 29, tzinfo=timezone.utc),  # 00:29 ET
        contracts=(c,),
    )
    assert overnight.as_of_date == date(2026, 8, 20)

    intraday = OptionChain(
        underlying="SPY", spot=762.6,
        fetched_at=datetime(2026, 8, 21, 18, 0, tzinfo=timezone.utc),  # 14:00 ET
        contracts=(c,),
    )
    assert intraday.as_of_date == date(2026, 8, 21)


def test_explicit_session_date_overrides_the_derived_one():
    c = contract(760)
    ch = OptionChain(
        underlying="SPY", spot=762.6,
        fetched_at=datetime(2026, 8, 24, 4, 29, tzinfo=timezone.utc),
        contracts=(c,),
        session_date=date(2026, 8, 21),  # Friday, not the derived Sunday
    )
    assert ch.as_of_date == date(2026, 8, 21)


def test_otm_iv_by_strike_takes_the_out_of_the_money_wing():
    """The live case: ITM put IV 0.1343 vs OTM call IV 0.0912 at the same strike.

    The put is ITM (K=779 > spot 762.60) and its IV is an artifact of a
    15.37/18.10 spread. The call's is the one to use.
    """
    expiry = date(2026, 8, 24)
    itm_put = contract(779, right="P", expiry=expiry, bid=15.37, ask=18.10, iv=0.1343)
    otm_call = contract(779, right="C", expiry=expiry, bid=0.03, ask=0.04, iv=0.0912)
    ch = chain([itm_put, otm_call], spot=762.60, when=date(2026, 8, 21))

    ivs = ch.otm_iv_by_strike()
    assert ivs[(expiry, 779.0)] == pytest.approx(0.0912)

    # And below spot the roles swap: the put is the OTM wing.
    itm_call = contract(745, right="C", expiry=expiry, iv=0.30)
    otm_put = contract(745, right="P", expiry=expiry, iv=0.14)
    ch2 = chain([itm_call, otm_put], spot=762.60, when=date(2026, 8, 21))
    assert ch2.otm_iv_by_strike()[(expiry, 745.0)] == pytest.approx(0.14)


def test_parity_breaks_catches_the_feed_contradicting_itself():
    """Gamma is identical for a call and a put at one strike. It is an identity.

    Real CBOE data on 2026-08-21 reported 0.00510 for the call and 0.00010 for
    the put at K=779 - a factor of 51. Every row this returns is a data fault.
    """
    expiry = date(2026, 8, 24)
    broken_call = OptionContract(
        symbol="SPY260824C00779000", underlying="SPY", expiry=expiry, strike=779.0,
        right="C", bid=0.03, ask=0.04, iv=0.0912, open_interest=1179, vendor_gamma=0.00510,
    )
    broken_put = OptionContract(
        symbol="SPY260824P00779000", underlying="SPY", expiry=expiry, strike=779.0,
        right="P", bid=15.37, ask=18.10, iv=0.1343, open_interest=18, vendor_gamma=0.00010,
    )
    consistent_call = OptionContract(
        symbol="SPY260824C00762000", underlying="SPY", expiry=expiry, strike=762.0,
        right="C", bid=3.56, ask=3.58, iv=0.0992, open_interest=138, vendor_gamma=0.04970,
    )
    consistent_put = OptionContract(
        symbol="SPY260824P00762000", underlying="SPY", expiry=expiry, strike=762.0,
        right="P", bid=2.80, ask=2.84, iv=0.0993, open_interest=1095, vendor_gamma=0.05020,
    )
    ch = chain(
        [broken_call, broken_put, consistent_call, consistent_put],
        spot=762.60, when=date(2026, 8, 21),
    )
    breaks = ch.parity_breaks()

    assert len(breaks) == 1
    expiry_out, strike, call_g, put_g = breaks[0]
    assert strike == pytest.approx(779.0)
    assert call_g == pytest.approx(0.00510)
    assert put_g == pytest.approx(0.00010)
    # The at-the-money pair agrees to 1% and must NOT be flagged.
    assert all(s != pytest.approx(762.0) for _, s, _, _ in breaks)


def test_deep_itm_is_excluded_from_pricing_but_only_when_spot_is_known():
    today = date(2026, 8, 21)
    itm_put = contract(779, right="P", expiry=date(2026, 8, 24), bid=15.37, ask=18.10, iv=0.1343)

    assert itm_put.quality(today, spot=762.60) & Quality.DEEP_ITM
    assert itm_put.usable_for_pricing(today, spot=762.60) is False
    # Without a spot the ITM test cannot run, so the contract survives. That is
    # why for_pricing() always passes the chain's spot.
    assert itm_put.usable_for_pricing(today) is True
    # It still carries open interest, so GEX keeps it.
    assert itm_put.usable_for_gex(today) is True
