"""Databento OPRA statistics reader.

No network, no purchased data, no `databento` dependency — the reader's
record source is patched with synthetic StatMsgs.

Every test here corresponds to a trap found by decoding a real order. They
are worth reading as documentation of what the file actually does, because
none of it is in the schema description.
"""

from __future__ import annotations

import datetime as dt

import pytest

from quantdesk.data.sources import databento_opra as dbo
from quantdesk.data.sources.databento_opra import (
    STAT_CLOSE_PRICE,
    STAT_HIGHEST_BID,
    STAT_LOWEST_OFFER,
    STAT_OPEN_INTEREST,
    UNDEFINED,
    load_session,
    session_summary,
)

ET = dt.timezone(dt.timedelta(hours=-4))


class FakeStat:
    """Stands in for a DBN StatMsg."""

    def __init__(self, instrument_id, stat_type, ts_event,
                 price=UNDEFINED[0], quantity=UNDEFINED[0]):
        self.instrument_id = instrument_id
        self.stat_type = stat_type
        self.ts_event = ts_event
        self.price = price
        self.quantity = quantity


def at(hour, minute, day=29):
    """Nanosecond timestamp for a given ET wall-clock time in Aug 2025."""
    moment = dt.datetime(2025, 8, day, hour, minute, tzinfo=ET)
    return int(moment.timestamp() * 1e9)


def patch_records(monkeypatch, records):
    monkeypatch.setattr(dbo, "read_statistics", lambda path: iter(records))


# ----------------------------------------------------- rule 1: OI in quantity


def test_open_interest_is_read_from_quantity_not_price(monkeypatch):
    """price is the undefined sentinel on OI records.

    Reading it would give 9.2e18 open interest -- not obviously wrong until
    it silently dominates every weighted sum it touches.
    """
    patch_records(monkeypatch, [
        FakeStat(1, STAT_OPEN_INTEREST, at(6, 30), quantity=4242),
    ])
    stats = load_session("x")
    assert stats[1].open_interest == 4242
    assert stats[1].open_interest < 1e9


def test_undefined_open_interest_is_dropped(monkeypatch):
    for sentinel in UNDEFINED:
        patch_records(monkeypatch, [
            FakeStat(1, STAT_OPEN_INTEREST, at(6, 30), quantity=sentinel),
        ])
        assert load_session("x")[1].open_interest is None


def test_zero_open_interest_is_dropped(monkeypatch):
    patch_records(monkeypatch, [
        FakeStat(1, STAT_OPEN_INTEREST, at(6, 30), quantity=0),
    ])
    assert load_session("x")[1].open_interest is None


# -------------------------------------- rule 2: OI describes the prior session


def test_open_interest_is_attributed_to_the_previous_session(monkeypatch):
    """Published 06:30 ET Friday -> it is Thursday's closing book."""
    patch_records(monkeypatch, [
        FakeStat(1, STAT_OPEN_INTEREST, at(6, 30, day=29), quantity=100),
    ])
    assert load_session("x")[1].oi_session == dt.date(2025, 8, 28)


def test_monday_open_interest_reaches_back_to_friday(monkeypatch):
    """2025-09-01 is a Monday; the prior session is Friday the 29th."""
    moment = dt.datetime(2025, 9, 1, 6, 30, tzinfo=ET)
    patch_records(monkeypatch, [
        FakeStat(1, STAT_OPEN_INTEREST, int(moment.timestamp() * 1e9),
                 quantity=100),
    ])
    assert load_session("x")[1].oi_session == dt.date(2025, 8, 29)


# ------------------------------------------- rule 3: the file mixes two days


def test_the_two_sessions_are_reported_separately(monkeypatch):
    """The whole point: a caller must not be able to conflate them."""
    patch_records(monkeypatch, [
        FakeStat(1, STAT_OPEN_INTEREST, at(6, 30), quantity=500),
        FakeStat(1, STAT_HIGHEST_BID, at(16, 15), price=1_000_000_000),
        FakeStat(1, STAT_LOWEST_OFFER, at(16, 15), price=1_200_000_000),
    ])
    s = load_session("x")[1]
    assert s.oi_session == dt.date(2025, 8, 28)
    assert s.price_session == dt.date(2025, 8, 29)
    assert s.oi_session != s.price_session


def test_summary_surfaces_both_sessions(monkeypatch):
    patch_records(monkeypatch, [
        FakeStat(1, STAT_OPEN_INTEREST, at(6, 30), quantity=500),
        FakeStat(1, STAT_LOWEST_OFFER, at(16, 15), price=1_200_000_000),
    ])
    summary = session_summary(load_session("x"))
    assert summary["oi_session"] == [dt.date(2025, 8, 28)]
    assert summary["price_session"] == [dt.date(2025, 8, 29)]


# --------------------------------------------- rule 4: the late one is a fix


def test_the_later_publication_wins(monkeypatch):
    """17:45 revises 16:15, and the revision is usually 0 -> a real value.

    Taking the first would leave ~17% of prices at zero, which reads as
    "no quote" and quietly drops those contracts from any study.
    """
    patch_records(monkeypatch, [
        FakeStat(1, STAT_CLOSE_PRICE, at(16, 15), price=0),
        FakeStat(1, STAT_CLOSE_PRICE, at(17, 45), price=177_900_000_000),
    ])
    assert load_session("x")[1].close == pytest.approx(177.90)


def test_order_in_the_file_does_not_matter(monkeypatch):
    """Records are not guaranteed sorted; the timestamp decides."""
    patch_records(monkeypatch, [
        FakeStat(1, STAT_CLOSE_PRICE, at(17, 45), price=177_900_000_000),
        FakeStat(1, STAT_CLOSE_PRICE, at(16, 15), price=0),
    ])
    assert load_session("x")[1].close == pytest.approx(177.90)


# --------------------------------------------------- rule 5: fixed point


def test_prices_are_scaled_from_fixed_point(monkeypatch):
    patch_records(monkeypatch, [
        FakeStat(1, STAT_HIGHEST_BID, at(16, 15), price=69_000_000_000),
        FakeStat(1, STAT_LOWEST_OFFER, at(16, 15), price=69_700_000_000),
    ])
    s = load_session("x")[1]
    assert s.bid == pytest.approx(69.00)
    assert s.offer == pytest.approx(69.70)
    assert s.mid == pytest.approx(69.35)


@pytest.mark.parametrize("sentinel", UNDEFINED)
def test_sentinel_prices_become_none(monkeypatch, sentinel):
    patch_records(monkeypatch, [
        FakeStat(1, STAT_HIGHEST_BID, at(16, 15), price=sentinel),
    ])
    assert load_session("x")[1].bid is None


def test_mid_is_none_without_both_sides(monkeypatch):
    patch_records(monkeypatch, [
        FakeStat(1, STAT_HIGHEST_BID, at(16, 15), price=69_000_000_000),
    ])
    assert load_session("x")[1].mid is None


# -------------------------------------------------------------- crossed rows


def test_crossed_rows_are_flagged_not_hidden(monkeypatch):
    """Session extremes, not a simultaneous quote: on a volatile day the
    highest bid can exceed the lowest offer, and the mid is then junk."""
    patch_records(monkeypatch, [
        FakeStat(1, STAT_HIGHEST_BID, at(16, 15), price=80_000_000_000),
        FakeStat(1, STAT_LOWEST_OFFER, at(16, 15), price=70_000_000_000),
    ])
    s = load_session("x")[1]
    assert s.crossed
    assert session_summary({1: s})["crossed"] == 1


def test_normal_rows_are_not_flagged_crossed(monkeypatch):
    patch_records(monkeypatch, [
        FakeStat(1, STAT_HIGHEST_BID, at(16, 15), price=69_000_000_000),
        FakeStat(1, STAT_LOWEST_OFFER, at(16, 15), price=69_700_000_000),
    ])
    assert not load_session("x")[1].crossed


# ------------------------------------------------------------------ summary


def test_summary_counts(monkeypatch):
    patch_records(monkeypatch, [
        FakeStat(1, STAT_OPEN_INTEREST, at(6, 30), quantity=100),
        FakeStat(2, STAT_OPEN_INTEREST, at(6, 30), quantity=250),
        FakeStat(3, STAT_OPEN_INTEREST, at(6, 30), quantity=0),
        FakeStat(1, STAT_LOWEST_OFFER, at(16, 15), price=1_000_000_000),
    ])
    s = session_summary(load_session("x"))
    assert s["instruments"] == 3
    assert s["with_open_interest"] == 2
    assert s["total_open_interest"] == 350
    assert s["with_offer"] == 1
