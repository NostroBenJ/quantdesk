"""Narration and journalling via the Claude API. Advisory only, by construction.

THIS LAYER CANNOT TRADE. It takes a frozen snapshot of a decision that the
deterministic signal and risk layers have ALREADY made, and returns a string.
`explain` returns `str`. There is no code path by which its output re-enters the
decision - which is the only reliable way to keep a language model out of a
backtestable system.

Two jobs:
  1. Per-decision rationale - why this trade is or is not being taken, pulling in
     regime, active modules and risk state. The equivalent of the morning brief
     already generated from the GEX dashboard.
  2. End-of-day journal - trades taken, trades skipped and why, and any rule
     violations the risk layer caught. Formatted for the Obsidian vault.

Determinism note: model output is not reproducible, so nothing here is ever part
of a backtest. Narration runs on live and paper sessions only.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from ..core.types import Signal, Trade

DEFAULT_MODEL = "claude-opus-5"

_SYSTEM_PROMPT = """You are the narration layer of a systematic trading system.

You do not make trading decisions. The signal and risk layers have already
decided; your job is to explain what they decided and why, in plain English, and
to flag anything that looks internally inconsistent.

Rules:
- Never recommend a trade, a size, or a level. If asked, say that is not your role.
- Work only from the state given. Do not invent levels, prices, or context.
- If the stated reasoning does not support the action taken, say so plainly.
  Catching that disagreement is the most useful thing you do.
- Be concise and concrete. No hedging language, no motivational commentary.
"""


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Frozen snapshot of one decision point. Input to the narrator, never output."""

    ts: datetime
    symbol: str
    action: str
    signals: tuple[Signal, ...] = ()
    regime: Mapping[str, Any] = field(default_factory=dict)
    risk_state: Mapping[str, Any] = field(default_factory=dict)
    blocked_by: tuple[str, ...] = ()

    def to_prompt(self) -> str:
        lines = [
            f"Time (UTC): {self.ts.isoformat()}",
            f"Symbol: {self.symbol}",
            f"Action taken: {self.action}",
        ]
        if self.regime:
            lines.append("Regime: " + ", ".join(f"{k}={v}" for k, v in self.regime.items()))
        if self.signals:
            lines.append("Signals:")
            for s in self.signals:
                lines.append(
                    f"  - {s.module}: {s.direction.name} conf={s.confidence:.2f} - {s.reasoning}"
                )
        if self.risk_state:
            lines.append("Risk state: " + ", ".join(f"{k}={v}" for k, v in self.risk_state.items()))
        if self.blocked_by:
            lines.append("Blocked by: " + ", ".join(self.blocked_by))
        return "\n".join(lines)


class Narrator(ABC):
    @abstractmethod
    def explain(self, context: DecisionContext) -> str:
        """One paragraph on why this happened."""

    @abstractmethod
    def journal(
        self,
        day: date,
        trades: Sequence[Trade],
        skipped: Sequence[Mapping[str, Any]],
        violations: Sequence[Mapping[str, Any]],
    ) -> str:
        """Markdown end-of-day entry, ready for the Obsidian vault."""


class NullNarrator(Narrator):
    """Default. Costs nothing, calls nothing, and is what backtests use."""

    def explain(self, context: DecisionContext) -> str:
        return context.to_prompt()

    def journal(self, day, trades, skipped, violations) -> str:
        lines = [f"# Trading journal - {day:%Y-%m-%d}", "", "## Trades"]
        lines += [
            f"- {t.ts_entry:%H:%M} {t.direction.name} {t.symbol} x{t.quantity:g} "
            f"-> net {t.net_pnl:+.2f} ({t.reasoning})"
            for t in trades
        ] or ["- none"]
        lines += ["", "## Skipped"]
        lines += [f"- {s.get('ts', '')}: {s.get('reason', '')}" for s in skipped] or ["- none"]
        lines += ["", "## Rule violations caught"]
        lines += [
            f"- {v.get('check', '')}: {v.get('reason', '')}" for v in violations
        ] or ["- none"]
        return "\n".join(lines)


class ClaudeNarrator(Narrator):
    """Calls the Anthropic Messages API. `anthropic` is imported lazily."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. The narrator is optional - "
                    "set narrative.enabled to false to run without it."
                )
            import anthropic  # lazy

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _ask(self, prompt: str) -> str:
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    def explain(self, context: DecisionContext) -> str:
        return self._ask(
            "Explain this decision in one short paragraph. If the reasoning does "
            "not support the action, say so first.\n\n" + context.to_prompt()
        )

    def journal(self, day, trades, skipped, violations) -> str:
        summary = NullNarrator().journal(day, trades, skipped, violations)
        return self._ask(
            "Write the end-of-day journal entry in Markdown from the log below. "
            "Keep every number exactly as given - do not recompute or round them. "
            "Add a short 'What to check tomorrow' section listing only things the "
            "log actually raises.\n\n" + summary
        )
