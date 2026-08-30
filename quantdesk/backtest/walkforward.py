"""Walk-forward validation. The default mode, not an optional extra.

A single in-sample backtest answers "could this parameter set have made money on
data I already have?", which is a question with no bearing on tomorrow. Bennett
makes the practitioner version of the point: the optimal call-overwriting strike
in a backtest is mostly a function of whether the sample window happened to have
a positive or negative index return. The parameter is fitting the sample's drift.

WINDOW LAYOUT
-------------
    |<---- train ---->|<-embargo->|<- test ->|
                       ^ purged

THE EMBARGO
-----------
The gap between train and test is not decoration. If a feature reads L bars of
history, then the first L bars of the test window are computed partly from bars
that were in the training set - so "out-of-sample" performance is contaminated by
in-sample data at the seam.

This is the same overlap problem already documented in `vrp_study.py`: forward_rv
looks 21 days ahead, adjacent rows share 20 of 21 days, and the fix there was to
count independent observations rather than rows. Here the fix is to discard
`embargo` bars at the boundary. Default embargo is the feature lookback, which is
the smallest value that fully decorrelates the seam.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..core.types import Bar, Trade
from .metrics import TradeStats


@dataclass(frozen=True, slots=True)
class Window:
    """Index ranges into a bar series. Half-open: [start, end)."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start

    @property
    def embargo_size(self) -> int:
        return self.test_start - self.train_end

    def describe(self, bars: Sequence[Bar]) -> str:
        return (
            f"window {self.index}: train {bars[self.train_start].ts_open:%Y-%m-%d}"
            f" -> {bars[self.train_end - 1].ts_close:%Y-%m-%d}"
            f" ({self.train_size} bars) | embargo {self.embargo_size}"
            f" | test {bars[self.test_start].ts_open:%Y-%m-%d}"
            f" -> {bars[self.test_end - 1].ts_close:%Y-%m-%d} ({self.test_size} bars)"
        )


def rolling_windows(
    n_bars: int,
    train_size: int,
    test_size: int,
    embargo: int = 0,
    step: int | None = None,
    anchored: bool = False,
) -> list[Window]:
    """Generate walk-forward windows.

    `anchored=True` keeps train_start at 0 and grows the training set (expanding
    window); the default rolls a fixed-length one. Rolling adapts faster to
    regime change; anchored uses more data. Neither is universally right - run
    both and if the answers disagree materially, the strategy is regime-fragile
    and you have learned something important.

    `step` defaults to `test_size`, giving non-overlapping test windows. That is
    the honest default: overlapping test windows reuse the same out-of-sample
    bars in several windows, which makes the aggregate OOS statistics look more
    significant than the data supports.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if embargo < 0:
        raise ValueError("embargo cannot be negative")
    stride = step if step is not None else test_size
    if stride <= 0:
        raise ValueError("step must be positive")

    windows: list[Window] = []
    train_start = 0
    index = 0
    while True:
        train_end = train_start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > n_bars:
            break
        windows.append(
            Window(
                index=index,
                train_start=0 if anchored else train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        index += 1
        train_start += stride
    return windows


@dataclass
class WalkForwardResult:
    """In-sample vs out-of-sample, side by side. The comparison is the product."""

    windows: list[Window]
    in_sample: list[TradeStats]
    out_of_sample: list[TradeStats]
    oos_trades: list[Trade]

    @property
    def combined_oos(self) -> TradeStats:
        return TradeStats.from_trades(self.oos_trades)

    @property
    def combined_is(self) -> TradeStats:
        """Trade-count-weighted average IS profit factor, for the degradation ratio."""
        total_profit = sum(s.gross_profit for s in self.in_sample)
        total_loss = sum(s.gross_loss for s in self.in_sample)
        n = sum(s.n for s in self.in_sample)
        pf = total_profit / total_loss if total_loss else (float("inf") if total_profit else 0.0)
        net = sum(s.net_pnl for s in self.in_sample)
        return TradeStats(
            n=n,
            wins=sum(s.wins for s in self.in_sample),
            losses=sum(s.losses for s in self.in_sample),
            win_rate=sum(s.wins for s in self.in_sample) / n if n else 0.0,
            gross_profit=total_profit,
            gross_loss=total_loss,
            net_pnl=net,
            total_costs=sum(s.total_costs for s in self.in_sample),
            profit_factor=pf,
            expectancy=net / n if n else 0.0,
            gross_expectancy=sum(s.gross_expectancy * s.n for s in self.in_sample) / n if n else 0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=max((s.largest_win for s in self.in_sample), default=0.0),
            largest_loss=min((s.largest_loss for s in self.in_sample), default=0.0),
        )

    @property
    def degradation(self) -> float:
        """OOS profit factor divided by IS profit factor.

        Around 1.0 the strategy generalises. Below ~0.5 the in-sample result was
        mostly curve fit. Above ~1.2 is not good news either - it usually means
        the test windows caught a favourable regime, and you should check whether
        the OOS sample is dominated by one period.
        """
        is_pf = self.combined_is.profit_factor
        oos_pf = self.combined_oos.profit_factor
        if is_pf in (0.0, float("inf")) or oos_pf == float("inf"):
            return float("nan")
        return oos_pf / is_pf

    @property
    def consistent_windows(self) -> int:
        """Test windows that were profitable after costs.

        A strategy carried by one window out of nine is not a strategy; it is a
        single event you have back-fitted around. This count is more informative
        than the aggregate.
        """
        return sum(1 for s in self.out_of_sample if s.net_pnl > 0)

    def report_lines(self) -> list[str]:
        lines = [
            f"walk-forward: {len(self.windows)} window(s), "
            f"{self.consistent_windows}/{len(self.out_of_sample)} profitable out-of-sample",
            "",
            f"{'win':>4} {'IS PF':>8} {'IS n':>6} {'OOS PF':>8} {'OOS n':>6} {'OOS net':>12}",
        ]
        for w, is_s, oos in zip(self.windows, self.in_sample, self.out_of_sample):
            lines.append(
                f"{w.index:>4} {is_s.profit_factor:>8.2f} {is_s.n:>6} "
                f"{oos.profit_factor:>8.2f} {oos.n:>6} {oos.net_pnl:>12.2f}"
            )
        combined = self.combined_oos
        lines += [
            "",
            f"combined OOS: n={combined.n}  PF={combined.profit_factor:.2f}  "
            f"expectancy={combined.expectancy:.2f}  net={combined.net_pnl:.2f}",
            f"degradation (OOS PF / IS PF): {self.degradation:.2f}",
        ]
        if not combined.is_meaningful:
            lines.append(
                f"WARNING: {combined.n} out-of-sample trades is below the "
                f"{TradeStats.__module__.split('.')[-1]} minimum of 30. "
                "Nothing above is statistically meaningful yet."
            )
        return lines


def run_walk_forward(
    bars: Sequence[Bar],
    windows: Sequence[Window],
    run_slice: Callable[[Sequence[Bar]], list[Trade]],
    fit: Callable[[Sequence[Bar]], None] | None = None,
) -> WalkForwardResult:
    """Execute a walk-forward pass.

    `fit` is called with the training slice before each test window, and is where
    parameter optimisation belongs. It is a separate callable precisely so that it
    is impossible to accidentally fit on the test slice: this function never hands
    the test bars to `fit`.
    """
    is_stats: list[TradeStats] = []
    oos_stats: list[TradeStats] = []
    all_oos: list[Trade] = []

    for window in windows:
        train = bars[window.train_start : window.train_end]
        test = bars[window.test_start : window.test_end]
        if fit is not None:
            fit(train)
        is_stats.append(TradeStats.from_trades(run_slice(train)))
        test_trades = run_slice(test)
        oos_stats.append(TradeStats.from_trades(test_trades))
        all_oos.extend(test_trades)

    return WalkForwardResult(
        windows=list(windows),
        in_sample=is_stats,
        out_of_sample=oos_stats,
        oos_trades=all_oos,
    )
