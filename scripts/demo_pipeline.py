"""End-to-end demonstration on real data, with no alpha claim anywhere in it.

Run:  python scripts/demo_pipeline.py

What it does, in order:

  1. Ingests a real TradingView SPY 5-minute export into SQLite, and reports what
     it dropped and why.
  2. Audits two candidate features for lookahead - one honest, one that cheats in
     the realistic way - and shows the guard catching the cheat.
  3. Backtests a CONTROL strategy that has zero edge by construction, and shows
     the cost-sensitivity table and verdict correctly calling it FAIL.
  4. Walk-forwards the same control and reports the degradation.

Step 3 is the point. Before trusting a report that says a strategy works, you
want evidence the report can say a strategy does not. A coin flip is the only
strategy whose true edge you know in advance, so it is the only one that can
validate the measuring instrument.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.backtest.costs import CommissionSchedule, SlippageParams, VolatilityScaledCostModel
from quantdesk.backtest.engine import BacktestEngine, EngineConfig, FixedSizer
from quantdesk.backtest.lookahead import check_causality
from quantdesk.backtest.metrics import TradeStats
from quantdesk.backtest.report import cost_sensitivity, grade_strategy
from quantdesk.backtest.view import MarketView
from quantdesk.backtest.walkforward import rolling_windows, run_walk_forward
from quantdesk.core.clock import Timeframe, utc
from quantdesk.core.types import AssetClass, Direction, InstrumentSpec, MarketState, Signal
from quantdesk.data.bars import PartialBarPolicy, find_gaps, normalise
from quantdesk.data.sources.csv_source import CsvBarSource
from quantdesk.data.store import BarStore
from quantdesk.signals.base import SignalModule

CSV = Path(r"C:\Users\jontr\Downloads\AMEX_SPY, 5_3d354.csv")
TF = Timeframe.parse("5m")
SPY = InstrumentSpec("SPY", AssetClass.ETF, multiplier=1.0, tick_size=0.01)


class CoinFlip(SignalModule):
    """Zero-edge control. Deterministic, so two runs agree exactly.

    NOT a strategy. Its only job is to prove the reporting machinery can detect
    the absence of edge - if this grades anything other than FAIL after costs,
    the reporting is broken and every other result it produces is suspect.
    """

    warmup = 20

    def generate_signal(self, market_state: MarketState) -> Signal:
        digest = hashlib.md5(str(market_state.ts.timestamp()).encode()).digest()
        direction = Direction.LONG if digest[0] % 2 else Direction.SHORT
        return Signal(
            module=self.name,
            ts=market_state.ts,
            symbol=market_state.symbol,
            direction=direction,
            confidence=1.0,
            reasoning="control: deterministic coin flip, no edge by construction",
        )


def rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    if not CSV.exists():
        print(f"demo data not found: {CSV}")
        return 1

    # ---------------------------------------------------------------- 1. ingest
    rule("1. INGEST - real SPY 5-minute export")

    # These exports are indicator dumps and carry no volume column. Rather than
    # substituting zero and letting a volume-relative impact model quietly price
    # fills off invented liquidity, the source makes you say so out loud, and the
    # impact term is switched off below.
    source = CsvBarSource(CSV, "SPY", TF, allow_missing_volume=True)
    raw = source.fetch("SPY", TF, utc(2000, 1, 1), utc(2030, 1, 1))
    clean = normalise(raw, TF, utc(2030, 1, 1), PartialBarPolicy.DROP)
    gaps = find_gaps(clean, TF)

    print(f"  fetched            {len(raw):>8,}")
    print(f"  after policy       {len(clean):>8,}  (dropped {len(raw) - len(clean)} incomplete)")
    print(f"  gaps               {len(gaps):>8,}  (session boundaries and holidays)")
    print(f"  first              {clean[0].ts_open.isoformat()}")
    print(f"  last               {clean[-1].ts_close.isoformat()}")
    print("  NOTE: no volume in this file -> market-impact term disabled below.")

    store_path = ROOT / "data" / "demo_bars.sqlite"
    with BarStore(store_path) as store:
        store.write(clean, TF, source.capabilities.name)
        coverage = store.coverage("SPY", TF)
    print(f"  stored             {coverage[2]:>8,} rows -> {store_path.name}")

    view = MarketView("SPY", clean)

    # ------------------------------------------------------------- 2. lookahead
    rule("2. LOOKAHEAD AUDIT - the guard must catch a cheat")

    def sma_20(bars, i):
        window = bars[max(0, i - 19) : i + 1]
        return sum(b.close for b in window) / len(window)

    def normalised_close(bars, i):
        # The realistic bug: normalise by the max of the whole array. Nobody
        # writes bars[i+1]; they write this and never think about it again.
        return bars[i].close / max(b.close for b in bars)

    sample = clean[:2000]
    for feature in (sma_20, normalised_close):
        report = check_causality(feature, sample, name=feature.__name__)
        mark = "OK  " if report.passed else "FAIL"
        print(f"  [{mark}] {report.summary()}")

    # ------------------------------------------------------------ 3. cost sweep
    rule("3. CONTROL STRATEGY - a coin flip, which must not pass")

    config = EngineConfig(initial_equity=100_000.0, volatility_lookback=20, signal_lookback=100)

    def run_at_multiple(multiple: float):
        model = VolatilityScaledCostModel(
            commissions=CommissionSchedule(per_unit=0.005),
            # impact_coefficient=0 because this file has no volume. The spread
            # term still applies and still scales with volatility.
            params=SlippageParams(impact_coefficient=0.0),
            multiple=multiple,
        )
        engine = BacktestEngine(
            SPY, TF, cost_model=model, fill_model=None, sizer=FixedSizer(100.0), config=config
        )
        return engine.run(view, CoinFlip()).trades

    sens = cost_sensitivity(run_at_multiple, multiples=(0.0, 0.5, 1.0, 2.0))
    for line in sens.report_lines():
        print("  " + line)

    gross, net = sens.stats[0], sens.stats[2]

    # ----------------------------------------------------------- 4. walk-forward
    rule("4. WALK-FORWARD - out-of-sample on the same control")

    cost_model = VolatilityScaledCostModel(
        commissions=CommissionSchedule(per_unit=0.005),
        params=SlippageParams(impact_coefficient=0.0),
    )

    def run_slice(bar_slice):
        if len(bar_slice) < 60:
            return []
        engine = BacktestEngine(
            SPY, TF, cost_model=cost_model, sizer=FixedSizer(100.0), config=config
        )
        return engine.run(MarketView("SPY", list(bar_slice)), CoinFlip()).trades

    windows = rolling_windows(
        n_bars=len(clean), train_size=2000, test_size=500, embargo=100
    )
    wf = run_walk_forward(clean, windows, run_slice)
    for line in wf.report_lines():
        print("  " + line)

    # --------------------------------------------------------------- 5. verdict
    verdict = grade_strategy(
        "CoinFlip control", net=net, gross=gross, sensitivity=sens, walk_forward=wf
    )
    print()
    print(verdict.render())

    print()
    if verdict.may_paper_trade:
        print("!! The control strategy PASSED. The reporting machinery is broken -")
        print("!! a coin flip cannot have edge. Do not trust any other result until")
        print("!! this is understood.")
        return 2
    print("Control correctly refused promotion. The instrument can detect no-edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
