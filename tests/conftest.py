"""Shared fixtures. Synthetic data only - tests must not hit the network."""

from __future__ import annotations

import math
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantdesk.core.clock import Timeframe, utc  # noqa: E402
from quantdesk.core.types import AssetClass, Bar, BarCompleteness, InstrumentSpec  # noqa: E402


@pytest.fixture
def tf_5m() -> Timeframe:
    return Timeframe.parse("5m")


@pytest.fixture
def spy() -> InstrumentSpec:
    return InstrumentSpec(symbol="SPY", asset_class=AssetClass.ETF, multiplier=1.0, tick_size=0.01)


@pytest.fixture
def mnq() -> InstrumentSpec:
    """Micro E-mini Nasdaq. tick 0.25 index points, $2/point => $0.50 per tick."""
    return InstrumentSpec(
        symbol="MNQ", asset_class=AssetClass.FUTURE, multiplier=2.0, tick_size=0.25
    )


def make_bars(
    n: int,
    timeframe: Timeframe,
    start_price: float = 100.0,
    drift: float = 0.0,
    wiggle: float = 0.0,
    volume: float = 10_000.0,
    symbol: str = "SPY",
    complete: bool = True,
) -> list[Bar]:
    """Deterministic synthetic bars. `drift` is per-bar log drift."""
    bars: list[Bar] = []
    ts = utc(2026, 1, 5, 14, 30)
    price = start_price
    for i in range(n):
        open_price = price
        close_price = price * math.exp(drift + (wiggle if i % 2 == 0 else -wiggle))
        high = max(open_price, close_price) * 1.001
        low = min(open_price, close_price) * 0.999
        bars.append(
            Bar(
                symbol=symbol,
                ts_open=ts,
                ts_close=ts + timedelta(seconds=timeframe.seconds),
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
                completeness=BarCompleteness.COMPLETE if complete else BarCompleteness.UNKNOWN,
            )
        )
        ts = ts + timedelta(seconds=timeframe.seconds)
        price = close_price
    return bars


@pytest.fixture
def bars_flat(tf_5m: Timeframe) -> list[Bar]:
    return make_bars(200, tf_5m)


@pytest.fixture
def bars_trending(tf_5m: Timeframe) -> list[Bar]:
    return make_bars(200, tf_5m, drift=0.0005, wiggle=0.0002)
