"""Pre-trade risk gate: the machinery, not the rules.

The rules are deliberately absent. You said you would supply the exact rule set
for the firm you are on, and guessing at a prop firm's trailing-drawdown
definition is worse than having no checker at all - a checker that passes trades
it should block is more dangerous than no checker, because you will trust it.

What is here is the shape every check plugs into. Adding your firm's rules means
writing `RiskCheck` subclasses whose numbers come from
`config/accounts/<your account>.yaml`.

CHECKS STILL TO WRITE (each needs a definition from the firm's rules document):

  DailyLossLimit          - does it count unrealised P&L, and when does the day roll?
  TrailingDrawdown        - trails off closed equity or intraday peak equity?
  MaxConcurrentPositions  - per symbol, or across the account?
  MaxCorrelatedExposure   - correlation measured over what window?
  ConsistencyRule         - max share of total profit from a single day
  NoTradeWindow           - news blackouts, session boundaries

The exact wording of each of those matters more than the number attached to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from ..core.types import Order, Position


@dataclass(frozen=True, slots=True)
class AccountState:
    """Everything a risk check is allowed to consider."""

    ts: datetime
    equity: float
    starting_balance: float
    day_start_equity: float
    peak_equity: float
    realised_pnl_today: float
    unrealised_pnl: float
    positions: Mapping[str, Position] = field(default_factory=dict)
    trades_today: int = 0
    trading_days_completed: int = 0

    @property
    def open_position_count(self) -> int:
        return sum(1 for p in self.positions.values() if not p.is_flat)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The result of one check. `reason` is mandatory on a block.

    Every decision - pass and block alike - is logged. "Why didn't it take that
    trade" must be answerable six weeks later from the log alone.
    """

    check: str
    allowed: bool
    reason: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.allowed and not self.reason:
            raise ValueError(f"{self.check}: a block must state a reason")


class RiskCheck(ABC):
    """One rule. Deterministic, and cheap enough to run before every order."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def evaluate(self, order: Order, account: AccountState) -> RiskDecision:
        ...


class PreTradeGate:
    """Runs every check and blocks if ANY of them blocks.

    Fail-closed on purpose: an unverified account config, a check that raises, or
    an empty check list all result in refusal rather than approval. The default
    answer to "may I trade this" is no.
    """

    def __init__(self, checks: list[RiskCheck], account_verified: bool = False) -> None:
        self.checks = checks
        self.account_verified = account_verified

    def evaluate(self, order: Order, account: AccountState) -> list[RiskDecision]:
        if not self.account_verified:
            return [
                RiskDecision(
                    check="account_verified",
                    allowed=False,
                    reason=(
                        "account config is not marked verified - set account.verified "
                        "to true only after checking every rule against the firm's "
                        "own document"
                    ),
                )
            ]
        if not self.checks:
            return [
                RiskDecision(
                    check="gate_configured",
                    allowed=False,
                    reason="no risk checks configured; refusing to pass trades unchecked",
                )
            ]

        decisions: list[RiskDecision] = []
        for check in self.checks:
            try:
                decisions.append(check.evaluate(order, account))
            except Exception as exc:  # a broken check must not become an approval
                decisions.append(
                    RiskDecision(
                        check=check.name,
                        allowed=False,
                        reason=f"check raised {type(exc).__name__}: {exc}",
                    )
                )
        return decisions

    @staticmethod
    def approved(decisions: list[RiskDecision]) -> bool:
        return bool(decisions) and all(d.allowed for d in decisions)
