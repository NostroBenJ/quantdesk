"""Dealer gamma, the flip level, and the store round-trip.

No network. The maths is checked against finite differences rather than
against a number someone once printed, which is the standing rule for
anything in this codebase that claims to be a derivative.

The store round-trip matters more than it looks: `read_chain` was added
because the recorder could write sessions and nothing could load them, so a
test that a written chain comes back identical is a test that the recorded
history is worth having.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from quantdesk.data.chain_store import ChainStore
from quantdesk.data.levels import (
    Convention,
    bs_gamma,
    compare,
    dealer_sign,
    gamma_profile,
    gex_dollars,
    net_gex_at,
    verify,
    zero_gamma,
)
from quantdesk.data.options import OptionChain, OptionContract

TODAY = date(2026, 8, 29)
EXPIRY = TODAY + timedelta(days=30)
FETCHED = datetime(2026, 8, 29, 20, 0, tzinfo=timezone.utc)


def contract(right: str, strike: float, oi: float = 5000.0,
             gamma: float | None = None) -> OptionContract:
    return OptionContract(
        symbol=f"X{right}{strike:.0f}", underlying="X", expiry=EXPIRY,
        strike=strike, right=right, bid=1.00, ask=1.10, last=1.05,
        volume=10.0, open_interest=oi, iv=0.20, vendor_gamma=gamma,
    )


def book(contracts, spot: float = 100.0) -> OptionChain:
    return OptionChain(
        underlying="X", spot=spot, fetched_at=FETCHED,
        contracts=tuple(contracts), source="synthetic", session_date=TODAY,
    )


@pytest.fixture
def two_sided() -> OptionChain:
    """Calls stacked above, puts stacked below - a book with a real flip."""
    return book(
        [contract("C", k) for k in (105.0, 110.0, 115.0)]
        + [contract("P", k) for k in (85.0, 90.0, 95.0)]
    )


# ------------------------------------------------------------------- maths


def test_module_self_test_passes():
    assert verify()


def test_gamma_is_the_derivative_of_delta():
    S, K, T, r, sigma, q = 100.0, 100.0, 0.25, 0.042, 0.20, 0.012

    def delta_call(s: float) -> float:
        d1 = ((math.log(s / K) + (r - q + 0.5 * sigma ** 2) * T)
              / (sigma * math.sqrt(T)))
        return math.exp(-q * T) * 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))

    h = 1e-4
    fd = (delta_call(S + h) - delta_call(S - h)) / (2 * h)
    assert bs_gamma(S, K, T, r, sigma, q) == pytest.approx(fd, abs=1e-9)


@pytest.mark.parametrize("args", [
    (100.0, 100.0, 0.0, 0.04, 0.2, 0.0),      # expired
    (100.0, 100.0, 0.25, 0.04, 0.0, 0.0),     # zero vol
    (0.0, 100.0, 0.25, 0.04, 0.2, 0.0),       # zero spot
    (100.0, 100.0, -1.0, 0.04, 0.2, 0.0),     # negative T
])
def test_degenerate_gamma_is_zero_not_nan(args):
    g = bs_gamma(*args)
    assert g == 0.0
    assert not math.isnan(g)


def test_gex_scales_with_oi_and_spot_squared():
    base = gex_dollars(0.02, 1000, 100.0)
    assert base == pytest.approx(200_000.0)
    assert gex_dollars(0.02, 2000, 100.0) == pytest.approx(2 * base)
    assert gex_dollars(0.02, 1000, 200.0) == pytest.approx(4 * base)


# -------------------------------------------------------------- convention


def test_conventions_are_exact_mirrors():
    for c in (contract("C", 100.0), contract("P", 100.0)):
        a = dealer_sign(c, Convention.DEALER_LONG_CALLS)
        b = dealer_sign(c, Convention.DEALER_SHORT_CALLS)
        assert a == -b


def test_mirroring_the_convention_negates_net_gex(two_sided):
    a = gamma_profile(two_sided, convention=Convention.DEALER_LONG_CALLS)
    b = gamma_profile(two_sided, convention=Convention.DEALER_SHORT_CALLS)
    assert b.net_gex == pytest.approx(-a.net_gex)


def test_flip_level_does_not_depend_on_the_convention(two_sided):
    """The regime LABEL is an assumption; the boundary is not.

    Negating a function does not move its zero, so the two dealer conventions
    disagree about which side is long gamma while agreeing exactly on where
    the line sits. Worth pinning: it means the level is a more trustworthy
    object than the label attached to either side of it.
    """
    a = zero_gamma(two_sided, convention=Convention.DEALER_LONG_CALLS)
    b = zero_gamma(two_sided, convention=Convention.DEALER_SHORT_CALLS)
    assert a is not None and b is not None
    assert a == pytest.approx(b, abs=0.02)


def test_all_long_has_no_flip(two_sided):
    assert zero_gamma(two_sided, convention=Convention.ALL_LONG) is None


# ------------------------------------------------------------ flip and walls


def test_flip_sits_between_the_clusters_and_gamma_changes_sign(two_sided):
    flip = zero_gamma(two_sided)
    assert flip is not None
    assert 95.0 < flip < 105.0
    below = net_gex_at(two_sided, flip - 2.0)
    above = net_gex_at(two_sided, flip + 2.0)
    assert (below > 0) != (above > 0)
    assert abs(net_gex_at(two_sided, flip)) < abs(below) * 1e-2


def test_one_sided_book_reports_none_not_zero():
    calls_only = book([contract("C", k) for k in (105.0, 110.0)])
    assert zero_gamma(calls_only) is None


def test_walls_find_the_heaviest_strike():
    heavy = book([contract("C", 105.0, oi=50_000.0),
                  contract("C", 110.0, oi=100.0),
                  contract("P", 95.0, oi=100.0)])
    prof = gamma_profile(heavy)
    assert prof.largest_walls(1)[0].strike == 105.0
    assert prof.nearest_wall(above=True).strike == 105.0
    assert prof.nearest_wall(above=False).strike == 95.0


def test_strike_oi_is_split_by_right():
    b = book([contract("C", 100.0, oi=300.0), contract("P", 100.0, oi=700.0)])
    row = gamma_profile(b).strikes[0]
    assert row.call_oi == 300.0
    assert row.put_oi == 700.0
    assert row.total_oi == 1000.0


def test_max_dte_excludes_far_expiries():
    near = contract("C", 105.0)
    far = OptionContract(
        symbol="XCFAR", underlying="X", expiry=TODAY + timedelta(days=200),
        strike=105.0, right="C", bid=1.0, ask=1.1, open_interest=5000.0,
        iv=0.20)
    b = book([near, far])
    assert gamma_profile(b, max_dte=45).contracts_used == 1
    assert gamma_profile(b).contracts_used == 2


# -------------------------------------------------------------- vs vendor


def test_compare_flags_a_wrong_vendor_gamma():
    """A vendor gamma off by 50% should show up as roughly a 50% error."""
    S, K, T = 100.0, 100.0, 30 / 365.0
    truth = bs_gamma(S, K, T, 0.042, 0.20, 0.012)
    b = book([contract("C", 100.0, gamma=truth * 1.5),
              contract("P", 100.0, gamma=truth * 1.5)])
    result = compare(b, max_dte=45)
    assert result is not None
    assert result.median_rel_error == pytest.approx(1 / 3, abs=0.02)


def test_compare_returns_none_without_vendor_greeks(two_sided):
    assert compare(two_sided) is None


# ------------------------------------------------------- store round-trip


def test_write_then_read_returns_an_equivalent_chain(tmp_path, two_sided):
    with ChainStore(tmp_path / "chains.sqlite") as store:
        store.write_chain(two_sided)
        back = store.read_chain("X", TODAY)

    assert back is not None
    assert back.underlying == two_sided.underlying
    assert back.spot == pytest.approx(two_sided.spot)
    assert back.as_of_date == two_sided.as_of_date
    assert len(back) == len(two_sided)

    before = {c.symbol: c for c in two_sided}
    for c in back:
        original = before[c.symbol]
        assert c.strike == pytest.approx(original.strike)
        assert c.right == original.right
        assert c.expiry == original.expiry
        assert c.open_interest == pytest.approx(original.open_interest)
        assert c.iv == pytest.approx(original.iv)


def test_gamma_survives_the_round_trip(tmp_path, two_sided):
    """The number the study uses must be the number that was recorded."""
    with ChainStore(tmp_path / "chains.sqlite") as store:
        store.write_chain(two_sided)
        back = store.read_chain("X", TODAY)
    assert gamma_profile(back).net_gex == pytest.approx(
        gamma_profile(two_sided).net_gex)
    assert zero_gamma(back) == pytest.approx(zero_gamma(two_sided), abs=0.02)


def test_missing_session_reads_as_none(tmp_path, two_sided):
    with ChainStore(tmp_path / "chains.sqlite") as store:
        store.write_chain(two_sided)
        assert store.read_chain("X", date(2020, 1, 1)) is None
        assert store.read_chain("NOPE", TODAY) is None


def test_last_snapshot_of_a_session_wins(tmp_path, two_sided):
    """Sessions hold several snapshots; the default is the end-of-day book."""
    later = OptionChain(
        underlying="X", spot=111.0,
        fetched_at=FETCHED + timedelta(hours=3),
        contracts=two_sided.contracts, source="synthetic", session_date=TODAY)
    with ChainStore(tmp_path / "chains.sqlite") as store:
        store.write_chain(two_sided)
        store.write_chain(later)
        stamps = store.snapshots_for("X", TODAY)
        assert len(stamps) == 2
        assert store.read_chain("X", TODAY).spot == pytest.approx(111.0)
        assert store.read_chain(
            "X", TODAY, fetched_at=stamps[0]).spot == pytest.approx(100.0)
