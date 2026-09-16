"""Produce run files for the dashboard from real SPY data.

    python scripts/make_demo_run.py

Writes two runs, each pinning a different guard:

  1. CoinFlip   - zero edge by construction, trading 10,004 times so costs
                  dominate. Grades FAIL at PF 0.19. Proves the report can detect
                  the absence of edge.
  2. BuyAndHold - the opposite extreme: one trade over the whole sample, so it
                  posts an INFINITE profit factor (no losing trades) on n=1.
                  Grades INSUFFICIENT DATA. Proves the sample-size guard fires
                  before the profit-factor headline, which is the case most
                  likely to fool you - an infinite PF looks like a discovery.

Neither is a strategy. They are the two calibration points that make the
dashboard's colours mean something, and deliberately neither is a PASS: a PASS
example would have to be constructed to pass, which is the exact habit this
system exists to break.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.backtest.costs import CommissionSchedule, SlippageParams, VolatilityScaledCostModel
from quantdesk.backtest.engine import BacktestEngine, EngineConfig, FixedSizer
from quantdesk.backtest.report import cost_sensitivity, grade_strategy
from quantdesk.backtest.view import MarketView
from quantdesk.backtest.walkforward import rolling_windows, run_walk_forward
from quantdesk.core.clock import Timeframe, utc
from quantdesk.core.types import AssetClass, Direction, InstrumentSpec, MarketState, Signal
from quantdesk.dash.results import build_payload, save_run
from quantdesk.data.bars import PartialBarPolicy, normalise
from quantdesk.data.sources.csv_source import CsvBarSource
from quantdesk.signals.base import SignalModule

CSV = Path((str(Path.home()) + r"\Downloads\AMEX_SPY, 5_3d354.csv"))
TF = Timeframe.parse("5m")
SPY = InstrumentSpec("SPY", AssetClass.ETF, multiplier=1.0, tick_size=0.01)
CONFIG = EngineConfig(initial_equity=100_000.0, volatility_lookback=20, signal_lookback=100)


class CoinFlip(SignalModule):
    """Zero edge by construction, deterministic so two runs agree exactly."""

    warmup = 20

    def generate_signal(self, market_state: MarketState) -> Signal:
        digest = hashlib.md5(str(market_state.ts.timestamp()).encode()).digest()
        return Signal(
            module=self.name,
            ts=market_state.ts,
            symbol=market_state.symbol,
            direction=Direction.LONG if digest[0] % 2 else Direction.SHORT,
            confidence=1.0,
            reasoning="control: deterministic coin flip, no edge by construction",
        )


class BuyAndHold(SignalModule):
    """Always long. Trades once, so costs are amortised over the whole sample -
    the opposite failure mode to CoinFlip, which trades 10,000 times."""

    warmup = 20

    def generate_signal(self, market_state: MarketState) -> Signal:
        return Signal(
            module=self.name,
            ts=market_state.ts,
            symbol=market_state.symbol,
            direction=Direction.LONG,
            confidence=1.0,
            reasoning="benchmark: unconditionally long",
        )


def cost_model(multiple: float = 1.0) -> VolatilityScaledCostModel:
    return VolatilityScaledCostModel(
        commissions=CommissionSchedule(per_unit=0.005),
        # This CSV carries no volume column, so the market-impact term cannot be
        # priced honestly and is switched off rather than defaulted to zero cost.
        params=SlippageParams(impact_coefficient=0.0),
        multiple=multiple,
    )


def evaluate(name: str, module_factory, bars, view) -> Path:
    def run_at_multiple(multiple: float):
        engine = BacktestEngine(
            SPY, TF, cost_model=cost_model(multiple), sizer=FixedSizer(100.0), config=CONFIG
        )
        return engine.run(view, module_factory()).trades

    engine = BacktestEngine(
        SPY, TF, cost_model=cost_model(), sizer=FixedSizer(100.0), config=CONFIG
    )
    result = engine.run(view, module_factory())

    sens = cost_sensitivity(run_at_multiple, multiples=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0))
    gross, net = sens.stats[0], sens.stats[2]

    def run_slice(bar_slice):
        if len(bar_slice) < 60:
            return []
        sub = BacktestEngine(
            SPY, TF, cost_model=cost_model(), sizer=FixedSizer(100.0), config=CONFIG
        )
        return sub.run(MarketView("SPY", list(bar_slice)), module_factory()).trades

    windows = rolling_windows(n_bars=len(bars), train_size=2000, test_size=500, embargo=100)
    wf = run_walk_forward(bars, windows, run_slice)

    verdict = grade_strategy(name, net=net, gross=gross, sensitivity=sens, walk_forward=wf)
    payload = build_payload(name, result, verdict, sens, wf)
    path = save_run(payload, runs_dir=ROOT / "runs")
    print(f"  {name:<18} {verdict.grade.value:<18} PF {net.profit_factor:>6.2f}  n={net.n:<6} -> {path.name}")
    return path


def main() -> int:
    if not CSV.exists():
        print(f"demo data not found: {CSV}")
        return 1

    source = CsvBarSource(CSV, "SPY", TF, allow_missing_volume=True)
    raw = source.fetch("SPY", TF, utc(2000, 1, 1), utc(2030, 1, 1))
    bars = normalise(raw, TF, utc(2030, 1, 1), PartialBarPolicy.DROP)
    view = MarketView("SPY", bars)
    print(f"loaded {len(bars):,} bars  {bars[0].ts_open.date()} -> {bars[-1].ts_close.date()}\n")

    evaluate("CoinFlip control", CoinFlip, bars, view)
    evaluate("BuyAndHold benchmark", BuyAndHold, bars, view)

    print("\nrun: python -m quantdesk.dash.app   ->  http://127.0.0.1:8010")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
