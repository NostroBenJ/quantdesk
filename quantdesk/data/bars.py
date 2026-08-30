"""Bar hygiene: completeness policy, gap detection, and vol estimators.

This module is where the documented partial-bar policy lives, because that policy
is the difference between a backtest and a fiction.

THE PARTIAL BAR PROBLEM
-----------------------
Every bar vendor - Yahoo included - will hand you the currently-forming bar as if
it were finished. Its `close` is not the period's close, it is the last trade.
Three things go wrong if you keep it:

1.  A strategy reading `close` sees a price from before the period ended, and
    then "predicts" the rest of that period. That is lookahead.
2.  Its `volume` is a fraction of a real bar's, which corrupts any
    volume-relative fill or impact model into being far too optimistic.
3.  Re-running the same backtest an hour later gives a different answer, because
    the partial bar has grown. Irreproducibility is the tell.

Default policy is DROP. It is the only policy that is safe by construction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import Enum

from ..core.clock import Timeframe, ensure_utc
from ..core.types import Bar, BarCompleteness


class PartialBarPolicy(Enum):
    """What to do with a bar that does not cover its full window."""

    DROP = "drop"
    """Discard it. The default, and the only one safe by construction."""

    KEEP_LABELLED = "keep_labelled"
    """Keep it but mark it PARTIAL. Live trading needs this - you cannot wait for
    the 16:00 close to decide something at 15:58 - but the backtest engine still
    refuses to compute features from it, so behaviour stays consistent."""

    RAISE = "raise"
    """Blow up. Use in tests and data-ingest jobs where a partial bar means the
    upstream fetch was mis-specified."""


class DataQualityError(ValueError):
    """Raised when bar data violates an invariant we refuse to trade on."""


def label_completeness(
    bars: Iterable[Bar],
    timeframe: Timeframe,
    as_of: datetime,
) -> list[Bar]:
    """Return `bars` with `completeness` set by comparing ts_close to `as_of`.

    A bar is COMPLETE only when its close time has actually passed. Anything
    whose window extends past `as_of` is PARTIAL, no matter what the vendor said.
    """
    now = ensure_utc(as_of)
    labelled: list[Bar] = []
    for bar in bars:
        expected_close = timeframe.close_of(bar.ts_open)
        if bar.ts_close != expected_close:
            # Vendor bar does not span a full period - e.g. a session-truncated
            # final bar on a half-day. Treat as partial: it is not comparable to
            # its neighbours for any per-bar statistic.
            state = BarCompleteness.PARTIAL
        elif bar.ts_close > now:
            state = BarCompleteness.PARTIAL
        else:
            state = BarCompleteness.COMPLETE
        labelled.append(
            Bar(
                symbol=bar.symbol,
                ts_open=bar.ts_open,
                ts_close=bar.ts_close,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                completeness=state,
            )
        )
    return labelled


def apply_partial_policy(
    bars: Sequence[Bar],
    policy: PartialBarPolicy = PartialBarPolicy.DROP,
) -> list[Bar]:
    """Enforce `policy` over already-labelled bars."""
    if policy is PartialBarPolicy.RAISE:
        bad = [b for b in bars if b.completeness is not BarCompleteness.COMPLETE]
        if bad:
            raise DataQualityError(
                f"{len(bad)} non-complete bar(s) present, first at {bad[0].ts_open}; "
                "policy is RAISE"
            )
        return list(bars)
    if policy is PartialBarPolicy.DROP:
        return [b for b in bars if b.completeness is BarCompleteness.COMPLETE]
    return list(bars)


def normalise(
    bars: Iterable[Bar],
    timeframe: Timeframe,
    as_of: datetime,
    policy: PartialBarPolicy = PartialBarPolicy.DROP,
) -> list[Bar]:
    """Full ingest pipeline: sort, de-duplicate, label, apply policy.

    De-duplication keeps the LAST occurrence of a given ts_open, on the reasoning
    that a re-fetch supersedes an earlier one (vendors revise bars, especially
    volume). Duplicates are common when appending an overlapping fetch window to
    an existing store.
    """
    by_open: dict[datetime, Bar] = {}
    for bar in bars:
        by_open[ensure_utc(bar.ts_open)] = bar
    ordered = [by_open[k] for k in sorted(by_open)]
    labelled = label_completeness(ordered, timeframe, as_of)
    return apply_partial_policy(labelled, policy)


def find_gaps(bars: Sequence[Bar], timeframe: Timeframe) -> list[tuple[datetime, datetime]]:
    """Return (previous_close, next_open) pairs where bars are missing.

    Gaps are expected and fine across sessions (overnight, weekends). This does
    not judge them - it reports them so ingest can log the count and you can
    eyeball whether a "gap" is a holiday or a vendor outage that will quietly
    bias a lookback window.
    """
    gaps: list[tuple[datetime, datetime]] = []
    for prev, nxt in zip(bars, bars[1:]):
        if nxt.ts_open > prev.ts_close:
            gaps.append((prev.ts_close, nxt.ts_open))
    return gaps


def log_returns(bars: Sequence[Bar]) -> list[float]:
    """Close-to-close log returns. Length is len(bars) - 1."""
    out: list[float] = []
    for prev, nxt in zip(bars, bars[1:]):
        if prev.close <= 0 or nxt.close <= 0:
            raise DataQualityError(f"non-positive close at {nxt.ts_open}")
        out.append(math.log(nxt.close / prev.close))
    return out


def realized_volatility(
    bars: Sequence[Bar],
    timeframe: Timeframe,
    annualise: bool = True,
) -> float:
    """Close-to-close realized volatility, as a decimal (0.16 == 16 vol).

    The timeframe is a REQUIRED argument, not a default. Bergomi (Ch.1, fn.1)
    makes the point that realized volatility is only defined relative to the time
    scale of the returns used to measure it, and that for a hedged book the
    relevant scale is the rehedge frequency. Defaulting to "daily" here would
    silently produce a number that means something different from what the caller
    thinks it means.

    Uses the zero-mean estimator (sum of squared returns / n), not the
    sample-variance estimator. Over short windows the estimated mean is almost
    entirely noise, and subtracting it biases volatility downward - which in a
    vol-targeted sizer means positions that are systematically too large.
    """
    rets = log_returns(bars)
    if len(rets) < 2:
        raise DataQualityError(
            f"need at least 3 bars to estimate volatility, got {len(bars)}"
        )
    variance = sum(r * r for r in rets) / len(rets)
    vol = math.sqrt(variance)
    return vol * timeframe.annualisation_factor if annualise else vol


def parkinson_volatility(
    bars: Sequence[Bar],
    timeframe: Timeframe,
    annualise: bool = True,
) -> float:
    """High-low range volatility (Parkinson 1980).

    Roughly 5x more efficient than close-to-close for the same sample size, which
    matters when the lookback is short. It does assume continuous observation and
    no drift, so it UNDERSTATES vol when a market gaps - the gap happens while the
    range is not being observed. Use it alongside, not instead of, the
    close-to-close estimator; a large divergence between the two is itself a
    signal that the period was gap-driven.
    """
    if not bars:
        raise DataQualityError("no bars supplied")
    scale = 1.0 / (4.0 * math.log(2.0))
    total = 0.0
    for bar in bars:
        if bar.low <= 0:
            raise DataQualityError(f"non-positive low at {bar.ts_open}")
        total += math.log(bar.high / bar.low) ** 2
    vol = math.sqrt(scale * total / len(bars))
    return vol * timeframe.annualisation_factor if annualise else vol


def true_ranges(bars: Sequence[Bar]) -> list[float]:
    """Gap-aware true range. Length is len(bars) - 1 (first bar has no prior close)."""
    out: list[float] = []
    for prev, nxt in zip(bars, bars[1:]):
        out.append(
            max(
                nxt.high - nxt.low,
                abs(nxt.high - prev.close),
                abs(nxt.low - prev.close),
            )
        )
    return out


def average_true_range(bars: Sequence[Bar], period: int) -> float:
    """Simple-average ATR over the last `period` true ranges."""
    trs = true_ranges(bars)
    if len(trs) < period:
        raise DataQualityError(f"need {period} true ranges, have {len(trs)}")
    window = trs[-period:]
    return sum(window) / len(window)
