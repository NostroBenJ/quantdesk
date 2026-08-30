"""Cash-account mechanics for a small Robinhood options account.

This module exists because of a specific, correctable dishonesty: a backtest that
assumes continuous position sizing and unlimited intraday turnover produces an
equity curve that CANNOT BE ACHIEVED in a $500-1,000 Level 2 cash account. That
is the same family of error as lookahead - modelling an account you do not have -
and it flatters results in exactly the same direction.

THE THREE CONSTRAINTS THAT ACTUALLY BIND
----------------------------------------
1. **Granularity.** Contracts are integers. At a $750 account and a $250
   premium, your position sizes are 0, 1, 2 or 3 - and 3 is the whole account.
   Vol-targeted sizing computes things like "risk $7.50 per trade", which rounds
   to zero contracts every time. The sizer is not wrong; it is inapplicable.

2. **Settlement.** A cash account settles options T+1. Cash from today's sale is
   not available to buy again until tomorrow. Any backtest that re-enters the
   same day on the same dollars is spending money that does not exist yet. This
   is the constraint most likely to be silently violated, because nothing in a
   naive engine tracks it.

3. **Level 2 is long-only.** Single-leg long calls and puts. No spreads, no
   short premium, no assignment risk - but also no way to express a short-vol
   view, which is where the VRP research points. Worth being explicit about:
   the best-evidenced edge in the research is one this account cannot trade.

None of this is a reason not to proceed. It is a reason to know, before running
a backtest, which of its trades were fiction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import IntEnum

#: Options contract multiplier. 1 contract = 100 shares of exposure.
CONTRACT_MULTIPLIER = 100.0


class OptionLevel(IntEnum):
    """Robinhood options approval levels, as they constrain strategy."""

    NONE = 0
    COVERED = 1
    """Covered calls, cash-secured puts."""
    LONG = 2
    """Single-leg long calls and puts. No spreads, no naked short premium."""
    SPREADS = 3
    """Debit and credit spreads."""

    @property
    def allows_short_premium(self) -> bool:
        return self >= OptionLevel.SPREADS

    @property
    def allows_multi_leg(self) -> bool:
        return self >= OptionLevel.SPREADS


@dataclass(frozen=True, slots=True)
class AccountFeasibility:
    """Whether a strategy is executable at all, at this account size."""

    balance: float
    premium: float
    max_contracts: int
    position_fraction: float
    """Fraction of the account one contract consumes."""
    granularity_steps: int
    """How many distinct non-zero position sizes exist. 1 means the only choice
    is all-or-nothing."""
    tradeable: bool
    reasons: tuple[str, ...] = ()

    def report_lines(self) -> list[str]:
        lines = [
            f"balance {self.balance:,.2f}   premium/contract {self.premium * CONTRACT_MULTIPLIER:,.2f}",
            f"max contracts {self.max_contracts}   "
            f"one contract = {self.position_fraction:.1%} of the account",
            f"distinct position sizes available: {self.granularity_steps}",
        ]
        lines += [f"  ! {r}" for r in self.reasons]
        return lines


@dataclass
class CashAccount:
    """A cash options account with T+1 settlement.

    Tracks unsettled proceeds explicitly so a backtest cannot spend them. This is
    stateful by necessity - settlement is a function of history, not of the
    current bar - so the engine must call `settle(session_date)` as sessions roll.
    """

    starting_balance: float
    option_level: OptionLevel = OptionLevel.LONG
    settlement_days: int = 1
    """T+1 for options. Equities are also T+1 as of 2024."""

    settled_cash: float = field(init=False)
    #: session_date -> proceeds becoming available on that date
    pending: dict[date, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        self.settled_cash = self.starting_balance

    @property
    def unsettled_cash(self) -> float:
        return sum(self.pending.values())

    @property
    def total_equity(self) -> float:
        """Settled plus unsettled. NOT what you can trade with today."""
        return self.settled_cash + self.unsettled_cash

    def settle(self, session_date: date) -> float:
        """Release proceeds whose settlement date has arrived. Returns the amount."""
        due = [d for d in self.pending if d <= session_date]
        released = sum(self.pending.pop(d) for d in due)
        self.settled_cash += released
        return released

    def buy(self, premium: float, contracts: int, session_date: date) -> float:
        """Debit settled cash for a purchase. Raises if the cash is not there.

        Deliberately raises rather than allowing a negative balance: a cash
        account cannot go negative, and an engine that lets it is modelling
        margin the account does not have.
        """
        cost = self.cost_of(premium, contracts)
        if cost > self.settled_cash + 1e-9:
            raise InsufficientSettledCash(
                f"need {cost:,.2f} but only {self.settled_cash:,.2f} is settled "
                f"({self.unsettled_cash:,.2f} unsettled). A cash account cannot "
                "spend proceeds before they settle."
            )
        self.settled_cash -= cost
        return cost

    def sell(self, premium: float, contracts: int, session_date: date) -> date:
        """Credit proceeds, available `settlement_days` later. Returns that date."""
        proceeds = self.cost_of(premium, contracts)
        available = session_date + timedelta(days=self.settlement_days)
        self.pending[available] = self.pending.get(available, 0.0) + proceeds
        return available

    @staticmethod
    def cost_of(premium: float, contracts: int) -> float:
        """Cash cost of `contracts` at `premium` per share."""
        return premium * CONTRACT_MULTIPLIER * contracts

    def max_contracts(self, premium: float, max_fraction: float = 1.0) -> int:
        """Largest integer contract count affordable from SETTLED cash.

        `max_fraction` caps the share of the account a single position may use.
        Returns 0 when even one contract is unaffordable - which at small account
        sizes is a routine answer, not an error.
        """
        if premium <= 0:
            return 0
        budget = self.settled_cash * max_fraction
        return int(budget // (premium * CONTRACT_MULTIPLIER))

    def feasibility(self, premium: float, max_fraction: float = 1.0) -> AccountFeasibility:
        """Can this account trade this contract at all, and how granularly?"""
        cost_one = self.cost_of(premium, 1)
        max_n = self.max_contracts(premium, max_fraction)
        fraction = cost_one / self.settled_cash if self.settled_cash > 0 else math.inf
        reasons: list[str] = []

        if max_n == 0:
            reasons.append(
                f"one contract costs {cost_one:,.2f} against {self.settled_cash:,.2f} "
                f"settled at a {max_fraction:.0%} cap - not tradeable"
            )
        if 0 < fraction > 0.25:
            reasons.append(
                f"one contract is {fraction:.0%} of the account; position sizing is "
                "effectively all-or-nothing and no risk model can smooth that"
            )
        if max_n == 1:
            reasons.append(
                "only one position size exists (1 contract) - vol-targeted or "
                "confidence-scaled sizing has no effect here"
            )
        if not self.option_level.allows_short_premium:
            reasons.append(
                f"option level {int(self.option_level)} is long-only: short-premium "
                "and spread strategies cannot be executed regardless of signal"
            )

        return AccountFeasibility(
            balance=self.settled_cash,
            premium=premium,
            max_contracts=max_n,
            position_fraction=fraction,
            granularity_steps=max_n,
            tradeable=max_n > 0,
            reasons=tuple(reasons),
        )


class InsufficientSettledCash(RuntimeError):
    """Raised when a buy would spend unsettled proceeds."""


@dataclass(frozen=True, slots=True)
class ContractSizer:
    """Integer-contract sizing for a cash account.

    Replaces `VolatilityTargetSizer` for options in a small account. Vol
    targeting assumes size is continuous; here it is not, so the honest model is
    a cap on account fraction, floored to whole contracts.
    """

    max_account_fraction: float = 0.25
    max_contracts: int = 10

    def contracts(self, account: CashAccount, premium: float, confidence: float = 1.0) -> int:
        if premium <= 0 or confidence <= 0:
            return 0
        affordable = account.max_contracts(premium, self.max_account_fraction)
        # Confidence scales the cap, then floors. At small sizes this collapses to
        # 1 or 0 for almost any confidence, which is the true behaviour and should
        # not be hidden behind a fractional number.
        scaled = int(affordable * confidence)
        return max(0, min(scaled, self.max_contracts))
