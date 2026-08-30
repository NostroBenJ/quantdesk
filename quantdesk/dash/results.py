"""Serialise a completed evaluation to JSON so the dashboard can read it back.

The dashboard is a READER. It never runs a backtest, never fetches market data,
and never recomputes a statistic. Everything it displays was produced by
`backtest/` and written here, which means the number on the screen is
byte-identical to the number the engine graded - there is no second
implementation to drift.

One JSON file per run in `runs/`. Small enough to keep forever, and diffable,
so "what changed between run 12 and run 13" is answerable with `git diff`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..backtest.engine import BacktestResult
from ..backtest.metrics import MIN_MEANINGFUL_TRADES, TradeStats, drawdown_series
from ..backtest.report import CostSensitivity, StrategyVerdict
from ..backtest.walkforward import WalkForwardResult

RUNS_DIR = Path("runs")


def _num(value: float) -> float | str:
    """JSON has no Infinity or NaN. An infinite profit factor is a real result
    (no losing trades) and must survive the round trip as something the UI can
    label honestly, rather than becoming `null` or a large finite lie."""
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
    return value


def _stats_dict(stats: TradeStats) -> dict[str, Any]:
    return {
        "n": stats.n,
        "wins": stats.wins,
        "losses": stats.losses,
        "win_rate": _num(stats.win_rate),
        "profit_factor": _num(stats.profit_factor),
        "expectancy": _num(stats.expectancy),
        "gross_expectancy": _num(stats.gross_expectancy),
        "net_pnl": _num(stats.net_pnl),
        "gross_profit": _num(stats.gross_profit),
        "gross_loss": _num(stats.gross_loss),
        "total_costs": _num(stats.total_costs),
        "cost_per_trade": _num(stats.cost_per_trade),
        "edge_to_cost_ratio": _num(stats.edge_to_cost_ratio),
        "average_win": _num(stats.average_win),
        "average_loss": _num(stats.average_loss),
        "largest_win": _num(stats.largest_win),
        "largest_loss": _num(stats.largest_loss),
        "is_meaningful": stats.is_meaningful,
    }


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Everything the dashboard needs about one evaluation."""

    run_id: str
    payload: dict[str, Any]

    @property
    def path(self) -> Path:
        return RUNS_DIR / f"{self.run_id}.json"


def build_payload(
    name: str,
    result: BacktestResult,
    verdict: StrategyVerdict,
    sensitivity: CostSensitivity | None = None,
    walk_forward: WalkForwardResult | None = None,
    notes: str = "",
) -> dict[str, Any]:
    equity = result.equity
    curve = [
        {"ts": ts.isoformat(), "equity": value}
        for ts, value in result.equity_curve
    ]
    drawdowns = drawdown_series(equity) if equity else []
    estats = result.equity_stats()

    payload: dict[str, Any] = {
        "schema": 1,
        "name": name,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": result.symbol,
        "module": result.module,
        "timeframe": {
            "label": result.timeframe.label,
            "periods_per_year": result.timeframe.periods_per_year,
        },
        "verdict": {
            "grade": verdict.grade.value,
            "may_paper_trade": verdict.may_paper_trade,
            "reasons": list(verdict.reasons),
        },
        "net": _stats_dict(verdict.net) if verdict.net else None,
        "gross": _stats_dict(verdict.gross) if verdict.gross else None,
        "equity_curve": curve,
        "max_drawdown_series": drawdowns,
        "equity_stats": {
            "n_periods": estats.n_periods,
            "total_return": _num(estats.total_return),
            "sharpe": _num(estats.sharpe),
            "sortino": _num(estats.sortino),
            "max_drawdown": _num(estats.max_drawdown),
            "max_drawdown_duration": estats.max_drawdown_duration,
            "final_equity": _num(estats.final_equity),
            "peak_equity": _num(estats.peak_equity),
            "min_equity": _num(estats.min_equity),
            "ruined": estats.ruined,
        },
        "execution": {
            "orders_submitted": result.orders_submitted,
            "orders_filled": result.orders_filled,
            "orders_partially_filled": result.orders_partially_filled,
            "orders_unfilled": result.orders_unfilled,
            "orders_expired": result.orders_expired,
            "fill_rate": _num(result.fill_rate),
            "forced_liquidation": result.forced_liquidation,
        },
        "warnings": list(result.warnings),
        "modules": {
            module: _stats_dict(stats) for module, stats in result.ledger.stats().items()
        },
        "min_meaningful_trades": MIN_MEANINGFUL_TRADES,
        "notes": notes,
    }

    if sensitivity is not None:
        payload["cost_sensitivity"] = {
            "breakeven_multiple": _num(sensitivity.breakeven_multiple),
            "rows": [
                {"multiple": m, **_stats_dict(s)}
                for m, s in zip(sensitivity.multiples, sensitivity.stats)
            ],
        }

    if walk_forward is not None:
        payload["walk_forward"] = {
            "degradation": _num(walk_forward.degradation),
            "consistent_windows": walk_forward.consistent_windows,
            "total_windows": len(walk_forward.out_of_sample),
            "combined_oos": _stats_dict(walk_forward.combined_oos),
            "combined_is": _stats_dict(walk_forward.combined_is),
            "windows": [
                {
                    "index": w.index,
                    "train_size": w.train_size,
                    "test_size": w.test_size,
                    "embargo": w.embargo_size,
                    "is_profit_factor": _num(is_s.profit_factor),
                    "is_n": is_s.n,
                    "oos_profit_factor": _num(oos.profit_factor),
                    "oos_n": oos.n,
                    "oos_net": _num(oos.net_pnl),
                }
                for w, is_s, oos in zip(
                    walk_forward.windows, walk_forward.in_sample, walk_forward.out_of_sample
                )
            ],
        }

    return payload


def save_run(payload: dict[str, Any], run_id: str | None = None, runs_dir: Path | None = None) -> Path:
    directory = runs_dir or RUNS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in payload.get("name", "run"))
    path = directory / f"{stamp}_{safe_name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def list_runs(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    """Newest first. Each entry is a summary, not the whole payload."""
    directory = runs_dir or RUNS_DIR
    if not directory.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt run file is reported, not skipped silently.
            out.append({"id": path.stem, "name": path.stem, "grade": "UNREADABLE", "error": True})
            continue
        net = payload.get("net") or {}
        out.append({
            "id": path.stem,
            "name": payload.get("name", path.stem),
            "created_at": payload.get("created_at"),
            "symbol": payload.get("symbol"),
            "module": payload.get("module"),
            "grade": payload.get("verdict", {}).get("grade", "?"),
            "profit_factor": net.get("profit_factor"),
            "n": net.get("n"),
            "error": False,
        })
    return out


def load_run(run_id: str, runs_dir: Path | None = None) -> dict[str, Any]:
    directory = runs_dir or RUNS_DIR
    path = directory / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no run {run_id!r} in {directory}")
    return json.loads(path.read_text(encoding="utf-8"))
