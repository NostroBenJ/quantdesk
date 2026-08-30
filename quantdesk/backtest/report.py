"""Reporting built to surface the failure mode, not bury it.

The specific wall this is designed against: a strategy that shows a profit factor
of 1.2 gross and 0.75-0.85 net, where the aggregate equity curve still slopes
upward for long enough stretches to look survivable. An equity chart is very good
at hiding that. A cost-sensitivity table is not.

THE THREE QUESTIONS
-------------------
1. Does the edge survive realistic costs?          -> net vs gross profit factor
2. How wrong can the cost estimate be first?       -> breakeven cost multiple
3. Does it survive out-of-sample?                  -> walk-forward degradation

A strategy has to answer all three. Passing one and failing another is the normal
case, not an edge case.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from ..core.types import Trade
from .metrics import MIN_MEANINGFUL_TRADES, TradeStats
from .walkforward import WalkForwardResult

#: Costs are estimates. This is the factor by which the estimate must be allowed
#: to be wrong while the strategy still makes money. 1.5 is not conservative -
#: with no bid/ask in the data, a 50% error in the spread estimate is ordinary.
REQUIRED_COST_HEADROOM = 1.5

#: Gross edge per trade must exceed cost per trade by this factor.
REQUIRED_EDGE_TO_COST = 2.0

#: Out-of-sample profit factor must retain this fraction of in-sample.
REQUIRED_DEGRADATION = 0.5


class Grade(Enum):
    PASS = "PASS"
    MARGINAL = "MARGINAL"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"


@dataclass(frozen=True, slots=True)
class CostSensitivity:
    """Profit factor as a function of how wrong the cost estimate is."""

    multiples: tuple[float, ...]
    stats: tuple[TradeStats, ...]

    @property
    def breakeven_multiple(self) -> float:
        """Cost multiple at which profit factor crosses 1.0.

        Linearly interpolated between the two bracketing sweep points. Returns
        inf if the strategy is profitable across the whole sweep, and 0.0 if it
        never is. This is the headline robustness number: below 1.0 means the
        strategy is already losing at your own cost estimate.
        """
        points = [(m, s.profit_factor) for m, s in zip(self.multiples, self.stats)]
        points.sort(key=lambda p: p[0])
        if not points or points[0][1] < 1.0:
            return 0.0
        for (m_lo, pf_lo), (m_hi, pf_hi) in zip(points, points[1:]):
            if pf_lo >= 1.0 > pf_hi:
                span = pf_lo - pf_hi
                if span <= 0:
                    return m_lo
                return m_lo + (m_hi - m_lo) * (pf_lo - 1.0) / span
        return math.inf

    def report_lines(self) -> list[str]:
        lines = [
            "the multiple scales SLIPPAGE only - commission is known exactly, so the",
            "0.00 row still carries commission and is 'gross of slippage', not free.",
            f"{'cost x':>8} {'PF':>8} {'net P&L':>12} {'expectancy':>12} {'costs':>12}",
        ]
        for m, s in zip(self.multiples, self.stats):
            pf = "inf" if s.profit_factor == math.inf else f"{s.profit_factor:.2f}"
            lines.append(
                f"{m:>8.2f} {pf:>8} {s.net_pnl:>12.2f} {s.expectancy:>12.2f} {s.total_costs:>12.2f}"
            )
        be = self.breakeven_multiple
        be_text = "never profitable" if be == 0.0 else (
            "profitable across the whole sweep" if be == math.inf else f"{be:.2f}x"
        )
        lines.append(f"breakeven cost multiple: {be_text}")
        return lines


def cost_sensitivity(
    run_at_multiple: Callable[[float], Sequence[Trade]],
    multiples: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
) -> CostSensitivity:
    """Re-run the same strategy at several cost levels.

    `run_at_multiple(0.0)` is the gross curve - the one that looks good. It is
    included on purpose: the gap between the 0.0 row and the 1.0 row IS the
    finding, and printing them adjacent makes it impossible to miss.
    """
    stats = tuple(TradeStats.from_trades(list(run_at_multiple(m))) for m in multiples)
    return CostSensitivity(multiples=tuple(multiples), stats=stats)


@dataclass
class StrategyVerdict:
    """A graded judgement, with reasons attached."""

    name: str
    grade: Grade
    reasons: list[str] = field(default_factory=list)
    net: TradeStats | None = None
    gross: TradeStats | None = None
    sensitivity: CostSensitivity | None = None
    walk_forward: WalkForwardResult | None = None

    @property
    def may_paper_trade(self) -> bool:
        """Gate from backtest to paper. PASS only.

        MARGINAL does not graduate. The point of grading is that the marginal
        cases are the ones that feel promising and are not.
        """
        return self.grade is Grade.PASS

    def report_lines(self) -> list[str]:
        lines = [
            "=" * 72,
            f"VERDICT: {self.grade.value} - {self.name}",
            "=" * 72,
        ]
        for reason in self.reasons:
            lines.append(f"  * {reason}")
        if self.gross and self.net:
            lines += [
                "",
                "gross vs net",
                "-" * 72,
                f"  profit factor   gross {self.gross.profit_factor:>8.2f}"
                f"   net {self.net.profit_factor:>8.2f}",
                f"  expectancy      gross {self.gross.expectancy:>8.2f}"
                f"   net {self.net.expectancy:>8.2f}",
                f"  edge/cost ratio {self.net.edge_to_cost_ratio:>14.2f}"
                f"   (need >= {REQUIRED_EDGE_TO_COST})",
                f"  trades          {self.net.n:>14}   (need >= {MIN_MEANINGFUL_TRADES})",
            ]
        if self.sensitivity:
            lines += ["", "cost sensitivity", "-" * 72]
            lines += [f"  {line}" for line in self.sensitivity.report_lines()]
        if self.walk_forward:
            lines += ["", "walk-forward", "-" * 72]
            lines += [f"  {line}" for line in self.walk_forward.report_lines()]
        lines.append("=" * 72)
        return lines

    def render(self) -> str:
        return "\n".join(self.report_lines())


def grade_strategy(
    name: str,
    net: TradeStats,
    gross: TradeStats | None = None,
    sensitivity: CostSensitivity | None = None,
    walk_forward: WalkForwardResult | None = None,
) -> StrategyVerdict:
    """Apply the graduation criteria and explain the result.

    Deliberately harsh, and deliberately explicit about which test failed. A
    verdict you disagree with is useful; a verdict you cannot interrogate is not.
    """
    reasons: list[str] = []
    fails = 0
    marginals = 0

    if net.n < MIN_MEANINGFUL_TRADES:
        reasons.append(
            f"only {net.n} trades - below the {MIN_MEANINGFUL_TRADES} minimum. "
            "No conclusion of any kind is available from this sample."
        )
        return StrategyVerdict(
            name, Grade.INSUFFICIENT_DATA, reasons, net, gross, sensitivity, walk_forward
        )

    if net.profit_factor < 1.0:
        fails += 1
        reasons.append(
            f"net profit factor {net.profit_factor:.2f} is below 1.0 - this loses money "
            "after costs."
        )
        if gross and gross.profit_factor >= 1.0:
            reasons.append(
                f"the gross profit factor is {gross.profit_factor:.2f}, so there IS a raw "
                "signal here; costs are eating it. Either trade it less often, capture "
                "more per trade, or trade a cheaper instrument - do not re-optimise the "
                "entry rule, which will only fit the sample."
            )
    else:
        reasons.append(f"net profit factor {net.profit_factor:.2f} clears 1.0.")

    ratio = net.edge_to_cost_ratio
    if ratio < REQUIRED_EDGE_TO_COST:
        marginals += 1
        reasons.append(
            f"gross edge per trade is only {ratio:.2f}x the cost per trade "
            f"(want >= {REQUIRED_EDGE_TO_COST}). The result depends on the slippage "
            "estimate being close to right, and it is an estimate."
        )

    if sensitivity is not None:
        be = sensitivity.breakeven_multiple
        if be < 1.0:
            fails += 1
            reasons.append(f"edge disappears below your own cost estimate (breakeven {be:.2f}x).")
        elif be < REQUIRED_COST_HEADROOM:
            marginals += 1
            reasons.append(
                f"breakeven cost multiple is {be:.2f}x - under the {REQUIRED_COST_HEADROOM}x "
                "headroom needed when the spread is estimated rather than observed."
            )
        else:
            reasons.append(f"survives costs up to {be:.2f}x the estimate.")

    if walk_forward is not None:
        deg = walk_forward.degradation
        if deg == deg and deg < REQUIRED_DEGRADATION:  # deg == deg filters NaN
            fails += 1
            reasons.append(
                f"out-of-sample profit factor is {deg:.2f}x in-sample - most of the "
                "in-sample result was curve fit."
            )
        elif deg == deg:
            reasons.append(f"out-of-sample retains {deg:.2f}x of in-sample profit factor.")
        total = len(walk_forward.out_of_sample)
        good = walk_forward.consistent_windows
        if total and good / total < 0.5:
            fails += 1
            reasons.append(
                f"only {good} of {total} out-of-sample windows were profitable - the "
                "result is carried by a minority of periods, not by a persistent edge."
            )

    if fails:
        grade = Grade.FAIL
    elif marginals:
        grade = Grade.MARGINAL
    else:
        grade = Grade.PASS
    return StrategyVerdict(name, grade, reasons, net, gross, sensitivity, walk_forward)
