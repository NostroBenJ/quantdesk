"""Structured JSON-lines logging.

One event per line, one file per UTC day. JSONL because it is appendable without
rewriting, greppable without a parser, and loadable by anything.

Every signal, order, fill, risk check, regime classification, and skip decision
goes here. The skips matter as much as the fills: "why didn't it take that trade"
is unanswerable after the fact unless the refusal was logged at the time, and
that question is most of what a trading journal is for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def _encode(value: Any) -> Any:
    """Make dataclasses, enums and datetimes JSON-safe without losing precision."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _encode(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        # JSON has no NaN/Infinity. Emitting them produces a file that json.loads
        # accepts but most other parsers reject - so serialise as a tagged string.
        return {"__float__": repr(value)}
    return value


class EventLog:
    """Append-only JSONL event log, rotated daily."""

    def __init__(self, directory: str | Path, run_id: str | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _path_for(self, ts: datetime) -> Path:
        return self.directory / f"{ts.astimezone(timezone.utc):%Y-%m-%d}.jsonl"

    def emit(self, event: str, ts: datetime | None = None, **fields: Any) -> dict[str, Any]:
        stamp = ts or datetime.now(tz=timezone.utc)
        record = {
            "ts": _encode(stamp),
            "logged_at": datetime.now(tz=timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **{k: _encode(v) for k, v in fields.items()},
        }
        with self._path_for(stamp).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record

    # Named helpers keep event names consistent across the codebase; a typo'd
    # event name is invisible until you try to query for it six weeks later.

    def signal(self, signal: Any, **extra: Any) -> dict[str, Any]:
        return self.emit("signal", ts=getattr(signal, "ts", None), signal=signal, **extra)

    def order(self, order: Any, **extra: Any) -> dict[str, Any]:
        return self.emit("order", ts=getattr(order, "ts", None), order=order, **extra)

    def fill(self, fill: Any, **extra: Any) -> dict[str, Any]:
        return self.emit("fill", ts=getattr(fill, "ts", None), fill=fill, **extra)

    def skip(self, ts: datetime, reason: str, **extra: Any) -> dict[str, Any]:
        """A trade that was NOT taken, and why. Do not skip logging skips."""
        return self.emit("skip", ts=ts, reason=reason, **extra)

    def risk_check(self, ts: datetime, check: str, passed: bool, **extra: Any) -> dict[str, Any]:
        return self.emit("risk_check", ts=ts, check=check, passed=passed, **extra)

    def regime(self, ts: datetime, classification: str, **extra: Any) -> dict[str, Any]:
        return self.emit("regime", ts=ts, classification=classification, **extra)

    def read_day(self, day: date) -> list[dict[str, Any]]:
        """Read back one day's events. The basis of the end-of-day journal."""
        path = self.directory / f"{day:%Y-%m-%d}.jsonl"
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
