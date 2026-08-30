"""Cash-account mechanics, chain storage, and the recorder's sanity checks.

The account tests use the real numbers: a $750 Level 2 cash account trading SPY
options. The point of most of them is to demonstrate a constraint that a naive
backtest would have silently violated.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from quantdesk.data.chain_store import ChainStore
from quantdesk.data.options import OptionChain, OptionContract
from quantdesk.data.recorder import sanity_check
from quantdesk.risk.account import (
    CONTRACT_MULTIPLIER,
    CashAccount,
    ContractSizer,
    InsufficientSettledCash,
    OptionLevel,
)

SESSION = date(2026, 8, 21)


def account(balance: float = 750.0) -> CashAccount:
    return CashAccount(starting_balance=balance, option_level=OptionLevel.LONG)


def contract(strike: float, right: str = "C", oi: float = 100.0, expiry=date(2026, 9, 18)):
    return OptionContract(
        symbol=f"SPY{expiry:%y%m%d}{right}{int(strike * 1000):08d}",
        underlying="SPY", expiry=expiry, strike=strike, right=right,
        bid=1.00, ask=1.10, last=1.05, volume=10.0, open_interest=oi, iv=0.20,
        vendor_gamma=0.01,
    )


# --------------------------------------------------------------- granularity


def test_one_contract_is_a_third_of_a_small_account():
    """The number that makes vol-targeted sizing inapplicable."""
    acct = account(750.0)
    premium = 2.50  # $250 per contract
    feas = acct.feasibility(premium)

    assert acct.cost_of(premium, 1) == pytest.approx(250.0)
    assert feas.max_contracts == 3
    assert feas.position_fraction == pytest.approx(1 / 3, rel=1e-3)
    assert feas.tradeable
    assert any("all-or-nothing" in r for r in feas.reasons)


def test_contract_that_costs_more_than_the_account_is_not_tradeable():
    acct = account(750.0)
    feas = acct.feasibility(premium=9.00)  # $900 per contract
    assert feas.max_contracts == 0
    assert feas.tradeable is False
    assert any("not tradeable" in r for r in feas.reasons)


def test_position_cap_reduces_max_contracts():
    acct = account(750.0)
    assert acct.max_contracts(2.50, max_fraction=1.0) == 3
    assert acct.max_contracts(2.50, max_fraction=0.5) == 1
    assert acct.max_contracts(2.50, max_fraction=0.25) == 0


def test_long_only_level_is_reported_as_a_constraint():
    """The VRP research points at short premium, which this account cannot trade."""
    acct = account()
    assert acct.option_level.allows_short_premium is False
    assert any("long-only" in r for r in acct.feasibility(1.00).reasons)
    assert OptionLevel.SPREADS.allows_short_premium is True


# ---------------------------------------------------------------- settlement


def test_proceeds_are_not_spendable_until_they_settle():
    """The constraint a naive engine violates on every same-day re-entry."""
    acct = account(500.0)
    acct.buy(premium=2.50, contracts=2, session_date=SESSION)  # spends 500
    assert acct.settled_cash == pytest.approx(0.0)

    acct.sell(premium=3.00, contracts=2, session_date=SESSION)  # 600 proceeds
    assert acct.unsettled_cash == pytest.approx(600.0)
    assert acct.settled_cash == pytest.approx(0.0)
    assert acct.total_equity == pytest.approx(600.0)

    # Same session: the money exists but is not spendable.
    with pytest.raises(InsufficientSettledCash, match="settle"):
        acct.buy(premium=2.50, contracts=1, session_date=SESSION)

    # Next session it is.
    acct.settle(date(2026, 8, 22))
    assert acct.settled_cash == pytest.approx(600.0)
    acct.buy(premium=2.50, contracts=2, session_date=date(2026, 8, 22))
    assert acct.settled_cash == pytest.approx(100.0)


def test_settlement_releases_only_what_is_due():
    acct = account(0.001)
    acct.sell(premium=1.00, contracts=1, session_date=date(2026, 8, 21))  # due 8/22
    acct.sell(premium=1.00, contracts=1, session_date=date(2026, 8, 25))  # due 8/26

    released = acct.settle(date(2026, 8, 22))
    assert released == pytest.approx(100.0)
    assert acct.unsettled_cash == pytest.approx(100.0)


def test_cash_account_cannot_go_negative():
    """An engine that allows this is modelling margin the account does not have."""
    acct = account(100.0)
    with pytest.raises(InsufficientSettledCash):
        acct.buy(premium=2.50, contracts=1, session_date=SESSION)
    assert acct.settled_cash == pytest.approx(100.0)  # unchanged


# --------------------------------------------------------------------- sizer


def test_contract_sizer_returns_whole_contracts_only():
    acct = account(750.0)
    sizer = ContractSizer(max_account_fraction=1.0)
    assert sizer.contracts(acct, premium=2.50) == 3
    assert isinstance(sizer.contracts(acct, premium=2.50), int)


def test_confidence_scaling_collapses_to_all_or_nothing_when_small():
    """At $750 there is no such thing as a half-size position.

    This is the honest behaviour and the test pins it: a 0.5-confidence signal
    does not produce a smaller risk, it produces one fewer contract - and at the
    bottom it produces none at all.
    """
    acct = account(750.0)
    sizer = ContractSizer(max_account_fraction=1.0)
    assert sizer.contracts(acct, 2.50, confidence=1.0) == 3
    assert sizer.contracts(acct, 2.50, confidence=0.5) == 1
    assert sizer.contracts(acct, 2.50, confidence=0.3) == 0


def test_sizer_respects_the_hard_contract_cap():
    acct = account(1_000_000.0)
    sizer = ContractSizer(max_account_fraction=1.0, max_contracts=10)
    assert sizer.contracts(acct, premium=1.00) == 10


# --------------------------------------------------------------- chain store


def _chain(session: date, contracts, spot: float = 760.0, source: str = "cboe-delayed"):
    return OptionChain(
        underlying="SPY", spot=spot,
        fetched_at=datetime(session.year, session.month, session.day, 20, 0, tzinfo=timezone.utc),
        contracts=tuple(contracts), source=source, session_date=session,
    )


def test_chain_store_roundtrip_and_idempotence(tmp_path):
    with ChainStore(tmp_path / "chains.sqlite") as store:
        chain = _chain(date(2026, 8, 20), [contract(760), contract(765, "P")])
        report = store.write_chain(chain)
        assert report["stored"] == 2

        store.write_chain(chain)  # re-running must not duplicate
        coverage = store.coverage("SPY")
        assert coverage["snapshots"] == 1
        assert coverage["contract_rows"] == 2


def test_contracts_with_no_interest_are_dropped():
    """Roughly a quarter of a real chain carries neither OI nor volume."""
    live = contract(760, oi=500.0)
    dead = OptionContract(
        symbol="SPY260918C00900000", underlying="SPY", expiry=date(2026, 9, 18),
        strike=900.0, right="C", open_interest=0.0, volume=0.0,
    )
    chain = _chain(date(2026, 8, 20), [live, dead])
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        with ChainStore(f"{tmp}/c.sqlite") as store:
            report = store.write_chain(chain)
    assert report["stored"] == 1
    assert report["dropped_empty"] == 1


def test_open_interest_change_is_computed_from_our_own_snapshots(tmp_path):
    """The paid `oi_change` field, reconstructed for free from two recordings."""
    with ChainStore(tmp_path / "chains.sqlite") as store:
        store.write_chain(_chain(date(2026, 8, 20), [contract(760, oi=1000.0)]))
        store.write_chain(
            _chain(date(2026, 8, 21), [contract(760, oi=1750.0), contract(770, oi=42.0)])
        )
        changes = {c["strike"]: c for c in store.open_interest_change("SPY", "2026-08-21")}

    assert changes[760.0]["oi_change"] == pytest.approx(750.0)
    assert changes[760.0]["is_new"] is False
    # A contract that did not exist yesterday has no change - which is NOT the
    # same as a change of zero, and must not be conflated with one.
    assert changes[770.0]["oi_change"] is None
    assert changes[770.0]["is_new"] is True


def test_missing_sessions_detects_a_dead_recorder(tmp_path):
    """The check that would have caught the capture dying on 2026-08-12."""
    with ChainStore(tmp_path / "chains.sqlite") as store:
        store.write_chain(_chain(date(2026, 8, 17), [contract(760)]))  # Monday
        store.write_chain(_chain(date(2026, 8, 18), [contract(760)]))  # Tuesday
        # nothing after
        missing = store.missing_sessions("SPY")

    assert "2026-08-19" in missing  # Wednesday
    assert "2026-08-20" in missing  # Thursday
    assert "2026-08-22" not in missing  # Saturday is not a session


def test_source_changes_are_visible_in_coverage(tmp_path):
    """A study spanning a feed change is comparing two instruments."""
    with ChainStore(tmp_path / "chains.sqlite") as store:
        store.write_chain(_chain(date(2026, 8, 11), [contract(760)], source="unusual-whales"))
        store.write_chain(_chain(date(2026, 8, 20), [contract(760)], source="cboe-delayed"))
        assert store.coverage("SPY")["distinct_sources"] == 2


# ------------------------------------------------------------------ recorder


def test_sanity_check_catches_a_truncated_chain():
    report = {"fetched": 500, "stored": 500, "total_oi": 20_000_000.0, "spot": 760.0,
              "parity_breaks": 0}
    problems = sanity_check(report)
    assert any("truncated" in p for p in problems)


def test_sanity_check_catches_collapsed_open_interest():
    report = {"fetched": 14_000, "stored": 10_000, "total_oi": 1_000.0, "spot": 760.0,
              "parity_breaks": 0}
    assert any("open interest" in p for p in sanity_check(report))


def test_sanity_check_passes_a_healthy_snapshot():
    """The real 2026-08-20 numbers."""
    report = {"fetched": 14_100, "stored": 10_944, "total_oi": 22_074_847.0,
              "spot": 762.60, "parity_breaks": 460}
    assert sanity_check(report) == []
