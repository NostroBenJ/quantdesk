"""The recorder. Fetch a chain, store it, write a health file, exit.

Designed to be run by cron (or Task Scheduler) rather than to be a daemon. A
one-shot process that exits is dramatically easier to reason about than a
long-lived one: it cannot leak, cannot wedge, and its failures are visible in the
scheduler's own exit codes.

THE FAILURE THIS IS BUILT AGAINST
---------------------------------
The previous capture pipeline died on 2026-08-12 and nobody noticed for nine
days. It died quietly - the scheduled tasks were terminated mid-run
(0xC000013A), then disabled, and nothing anywhere said so.

So every run writes `health.json` whether it succeeds or fails, and the health
file carries `missing_sessions`. A recorder that is not running produces a
health file that is either stale or full of gaps, and both are visible from the
dashboard. Silence is never interpreted as success.

Exit codes, for the scheduler:
    0  recorded
    1  fetch or store failed
    2  recorded, but the data failed a sanity check
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chain_store import ChainStore
from .options import ChainError
from .sources.cboe import CboeChainSource

#: A SPY chain has ~14,000 contracts. If a fetch returns dramatically fewer, the
#: feed changed or truncated, and storing it silently would corrupt the series.
MIN_EXPECTED_CONTRACTS = 2_000

#: Likewise for open interest. SPY runs ~20M+; an order-of-magnitude drop is a
#: data incident, not a market event.
MIN_EXPECTED_OI = 1_000_000.0


@dataclass(frozen=True, slots=True)
class RecorderConfig:
    symbols: tuple[str, ...] = ("SPY",)
    db_path: Path = Path("data/chains.sqlite")
    health_path: Path = Path("data/health.json")


def sanity_check(report: dict[str, Any]) -> list[str]:
    """Return reasons the snapshot looks wrong. Empty means it looks fine."""
    problems: list[str] = []
    if report["fetched"] < MIN_EXPECTED_CONTRACTS:
        problems.append(
            f"only {report['fetched']} contracts fetched (expected >= "
            f"{MIN_EXPECTED_CONTRACTS}) - feed may have truncated"
        )
    if report["total_oi"] < MIN_EXPECTED_OI:
        problems.append(
            f"total open interest {report['total_oi']:,.0f} below "
            f"{MIN_EXPECTED_OI:,.0f} - suspicious for a liquid underlying"
        )
    if report["spot"] <= 0:
        problems.append("non-positive spot")
    # A handful of parity breaks is normal (see options.Quality.PARITY_BREAK).
    # Half the book is not.
    if report["stored"] and report["parity_breaks"] > report["stored"] * 0.4:
        problems.append(
            f"{report['parity_breaks']} parity breaks across {report['stored']} "
            "stored contracts - the feed's own greeks are broadly inconsistent"
        )
    return problems


def record_once(config: RecorderConfig | None = None) -> int:
    cfg = config or RecorderConfig()
    started = datetime.now(tz=timezone.utc)
    health: dict[str, Any] = {
        "started_at": started.isoformat(),
        "ok": False,
        "symbols": {},
        "errors": [],
    }
    exit_code = 0

    source = CboeChainSource()
    try:
        with ChainStore(cfg.db_path) as store:
            for symbol in cfg.symbols:
                entry: dict[str, Any] = {}
                try:
                    chain = source.fetch_chain(symbol)
                    report = store.write_chain(chain)
                    problems = sanity_check(report)
                    entry = {**report, "problems": problems}
                    if problems:
                        exit_code = max(exit_code, 2)
                    print(
                        f"{symbol}: session {report['session_date']} "
                        f"spot {report['spot']:.2f} "
                        f"stored {report['stored']:,}/{report['fetched']:,} "
                        f"OI {report['total_oi']:,.0f}"
                    )
                    for problem in problems:
                        print(f"  ! {problem}")
                except (ChainError, OSError) as exc:
                    exit_code = 1
                    entry = {"error": f"{type(exc).__name__}: {exc}"}
                    health["errors"].append(f"{symbol}: {entry['error']}")
                    print(f"{symbol}: FAILED - {entry['error']}", file=sys.stderr)

                coverage = store.coverage(symbol)
                if coverage:
                    entry["coverage"] = coverage
                    missing = store.missing_sessions(symbol)
                    # Today is legitimately absent until the recorder runs.
                    today = datetime.now(tz=timezone.utc).date().isoformat()
                    entry["missing_sessions"] = [m for m in missing if m != today]
                health["symbols"][symbol] = entry
    except Exception as exc:  # noqa: BLE001 - the health file must be written regardless
        exit_code = 1
        health["errors"].append(f"{type(exc).__name__}: {exc}")
        health["traceback"] = traceback.format_exc()

    health["ok"] = exit_code == 0
    health["exit_code"] = exit_code
    health["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
    health["duration_seconds"] = (
        datetime.now(tz=timezone.utc) - started
    ).total_seconds()

    cfg.health_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.health_path.write_text(json.dumps(health, indent=2), encoding="utf-8")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    symbols = tuple(a.upper() for a in args) or ("SPY",)
    return record_once(RecorderConfig(symbols=symbols))


if __name__ == "__main__":
    raise SystemExit(main())
