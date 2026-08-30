"""Numerical lookahead detection.

CLAUDE.md already carries the right instinct for pricing code: do not assert that
a formula is correct, verify it numerically against an independent computation.
This module applies the same discipline to causality.

The claim "this feature does not use future data" is testable, and the test is
cheap. Compute the feature at time T twice:

    1. TRUNCATION - with the series cut off at T.
    2. PERTURBATION - with every bar after T replaced by shocked prices.

If either result differs from the feature computed on the full series, the
feature read the future. Truncation alone is not enough: a function that
normalises by `max(all_closes)` or `len(bars)` survives truncation in some
implementations but fails perturbation. Perturbation alone is not enough either:
a feature that happens to be insensitive to the particular shock passes. Running
both is what makes this a check rather than a gesture.

This is the direct analogue of the central-difference test:

    fd = (f(x+h) - f(x-h)) / (2h);  assert abs(analytic - fd) < tol

Same shape, same reason. It cost three lines and it is the difference between
code that looks causal and code that is causal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..core.types import Bar
from .view import LookaheadError

#: A feature function takes a bar series and an index, and returns a number
#: computed as of bars[index]. It must never look past `index`.
FeatureFn = Callable[[Sequence[Bar], int], float]


def _shock_bar(bar: Bar, factor: float) -> Bar:
    """Return `bar` with all prices scaled. Volume is shocked too, since
    volume-dependent features are just as capable of peeking."""
    return Bar(
        symbol=bar.symbol,
        ts_open=bar.ts_open,
        ts_close=bar.ts_close,
        open=bar.open * factor,
        high=bar.high * factor,
        low=bar.low * factor,
        close=bar.close * factor,
        volume=bar.volume * factor,
        completeness=bar.completeness,
    )


def perturb_future(bars: Sequence[Bar], index: int, factor: float = 1.5) -> list[Bar]:
    """Copy `bars` with everything strictly after `index` scaled by `factor`."""
    return [b if i <= index else _shock_bar(b, factor) for i, b in enumerate(bars)]


@dataclass(frozen=True, slots=True)
class CausalityReport:
    feature: str
    checked: int
    truncation_failures: tuple[int, ...]
    perturbation_failures: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return not self.truncation_failures and not self.perturbation_failures

    def summary(self) -> str:
        if self.passed:
            return f"{self.feature}: causal over {self.checked} index(es)"
        return (
            f"{self.feature}: LOOKAHEAD - "
            f"{len(self.truncation_failures)} truncation, "
            f"{len(self.perturbation_failures)} perturbation failure(s); "
            f"first at index "
            f"{min([*self.truncation_failures, *self.perturbation_failures])}"
        )


def check_causality(
    feature: FeatureFn,
    bars: Sequence[Bar],
    indices: Sequence[int] | None = None,
    name: str | None = None,
    tolerance: float = 1e-12,
    shock: float = 1.5,
) -> CausalityReport:
    """Test `feature` for future-data dependence. Does not raise; returns a report.

    `indices` defaults to a spread of positions across the series. Checking every
    index is O(n^2) and rarely tells you anything the sample did not.
    """
    label = name or getattr(feature, "__name__", "feature")
    n = len(bars)
    if n < 3:
        raise ValueError("need at least 3 bars to test causality")
    if indices is None:
        # Skip the first few - most features need a warmup and legitimately raise.
        start = max(2, n // 10)
        indices = sorted({start, n // 3, n // 2, (2 * n) // 3, n - 2})
        indices = [i for i in indices if 0 <= i < n - 1]

    trunc_fail: list[int] = []
    perturb_fail: list[int] = []

    for i in indices:
        try:
            baseline = feature(bars, i)
        except Exception:
            # A feature that cannot compute at this index (insufficient warmup)
            # is not a causality failure. Skip it.
            continue

        truncated = feature(list(bars[: i + 1]), i)
        if not _close(baseline, truncated, tolerance):
            trunc_fail.append(i)

        perturbed = feature(perturb_future(bars, i, shock), i)
        if not _close(baseline, perturbed, tolerance):
            perturb_fail.append(i)

    return CausalityReport(
        feature=label,
        checked=len(indices),
        truncation_failures=tuple(trunc_fail),
        perturbation_failures=tuple(perturb_fail),
    )


def assert_causal(
    feature: FeatureFn,
    bars: Sequence[Bar],
    indices: Sequence[int] | None = None,
    name: str | None = None,
    tolerance: float = 1e-12,
    shock: float = 1.5,
) -> CausalityReport:
    """`check_causality`, but raises `LookaheadError` on failure.

    Call this in the test for every feature. It is three lines and it is the
    difference between a backtest and a story.
    """
    report = check_causality(feature, bars, indices, name, tolerance, shock)
    if not report.passed:
        raise LookaheadError(report.summary())
    return report


def _close(a: float, b: float, tol: float) -> bool:
    if a != a or b != b:  # NaN on either side is not a pass
        return False
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tol * scale
