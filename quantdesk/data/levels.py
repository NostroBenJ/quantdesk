"""Dealer gamma exposure, and the levels it implies.

WHY THIS EXISTS
---------------
`options.py` has kept `vendor_gamma` in a separate field since it was written,
with a note saying it exists "so `levels.compare()` can disagree with us out
loud". This is that module.

THE ONE THING TO UNDERSTAND BEFORE USING ANY NUMBER IN HERE
-----------------------------------------------------------
**The dealer sign is an assumption, not an observation.** Nobody publishes who
is long which contracts. Every GEX number anywhere - ours, SpotGamma's,
SqueezeMetrics', Unusual Whales' - rests on a guess about who holds what, and
the guesses differ.

The conventional guess, and our default, is that customers sell calls (covered
calls) and buy puts (protection), so dealers are LONG call open interest and
SHORT put open interest. That is `Convention.DEALER_LONG_CALLS`.

It is a guess. It is wrong for individual strikes constantly, and it may be
wrong in aggregate during regimes where customer behaviour shifts. So the
convention is a parameter, not a constant, and `zero_gamma()` under two
different conventions gives two different levels. If a study's conclusion
flips when the convention flips, the conclusion was about the convention.

This is also why we compute gamma ourselves rather than buying it. A vendor's
pre-computed GEX bakes in their sign convention, their rate, their dividend
yield and their IV choice, none of which they document, and none of which you
can vary to see whether your result depends on them.

WHAT ELSE IS ASSUMED
--------------------
* `RISK_FREE` and `DIVIDEND_YIELD`. CBOE does not publish the r and q behind
  its greeks (see `scripts/check_chain.py`). Ours are stated here rather than
  buried, and `compare()` exists partly because a systematic gap between our
  gamma and theirs is more likely a rate mismatch than a maths error.
* IV comes from the OTM wing at each strike. An ITM option's IV is an artifact
  of its bid/ask spread; feeding that to a greek makes OUR number wrong too.
* `T` is calendar-year fraction from `OptionContract.time_to_expiry`.

Standard library only, so the package stays importable with no dependencies -
the same rule the rest of quantdesk follows.

Run `python -m quantdesk.data.levels` for the self-test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum

from .options import OptionChain, OptionContract

# ASSUMPTION (market data): not published by the feed. Stated, not buried.
RISK_FREE = 0.042
DIVIDEND_YIELD = 0.012

#: Shares per option contract. SPX and SPY both use 100.
CONTRACT_MULTIPLIER = 100.0


class Convention(Enum):
    """Who is assumed to hold what. See the module docstring - this is a guess.

    DEALER_LONG_CALLS   customers sell calls and buy puts, so dealers are long
                        call OI and short put OI. The conventional choice.
    DEALER_SHORT_CALLS  the mirror image. Not a serious belief; it exists so a
                        study can show its result is not an artifact of the
                        convention by running both.
    ALL_LONG            no sign flip at all - every contract counts positive.
                        Measures gross gamma concentration rather than net
                        dealer positioning. Useful for finding walls, useless
                        for finding a flip level.
    """

    DEALER_LONG_CALLS = "dealer_long_calls"
    DEALER_SHORT_CALLS = "dealer_short_calls"
    ALL_LONG = "all_long"


def dealer_sign(contract: OptionContract, convention: Convention) -> float:
    if convention is Convention.ALL_LONG:
        return 1.0
    long_calls = convention is Convention.DEALER_LONG_CALLS
    if contract.is_call:
        return 1.0 if long_calls else -1.0
    return -1.0 if long_calls else 1.0


# --------------------------------------------------------------------- maths


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0) -> float:
    """d(delta)/dS. Identical for calls and puts at the same strike.

    Returns 0.0 rather than raising on a degenerate input (expired, zero vol,
    zero spot). A chain always contains some of these, and a NaN propagating
    into a sum is far worse than a strike contributing nothing.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * sqrt_t)


def gex_dollars(gamma: float, open_interest: float, spot: float,
                multiplier: float = CONTRACT_MULTIPLIER) -> float:
    """Gamma in dollars of delta per 1% move in spot.

    gamma is per share per $1. Times OI times the multiplier gives total delta
    change per $1. Times spot converts to dollars. Times spot again, over 100,
    restates it per 1% move instead of per $1 - which is the unit everyone
    quotes GEX in, and the only reason the spot term appears squared.
    """
    return gamma * open_interest * multiplier * spot * spot * 0.01


# ------------------------------------------------------------------ profile


@dataclass(frozen=True)
class StrikeGamma:
    """Net dealer gamma at one strike, in dollars per 1% move."""

    strike: float
    gex: float
    call_oi: float
    put_oi: float

    @property
    def total_oi(self) -> float:
        return self.call_oi + self.put_oi


@dataclass(frozen=True)
class GammaProfile:
    """The whole book's dealer gamma, by strike."""

    underlying: str
    spot: float
    as_of: date
    convention: Convention
    strikes: tuple[StrikeGamma, ...]
    contracts_used: int
    contracts_skipped: int

    @property
    def net_gex(self) -> float:
        """Total dealer gamma at the CURRENT spot."""
        return sum(s.gex for s in self.strikes)

    def largest_walls(self, n: int = 5) -> list[StrikeGamma]:
        """Strikes with the most gamma, by absolute size."""
        return sorted(self.strikes, key=lambda s: -abs(s.gex))[:n]

    def nearest_wall(self, above: bool) -> StrikeGamma | None:
        """The biggest gamma strike above (or below) spot."""
        side = [s for s in self.strikes
                if (s.strike > self.spot if above else s.strike < self.spot)]
        return max(side, key=lambda s: abs(s.gex)) if side else None


def _iv_lookup(chain: OptionChain, max_dte: int | None):
    """OTM-wing IV per (expiry, strike). ITM IV is a spread artifact."""
    return chain.otm_iv_by_strike(max_dte=max_dte)


def gamma_profile(chain: OptionChain, *,
                  convention: Convention = Convention.DEALER_LONG_CALLS,
                  max_dte: int | None = None,
                  spot: float | None = None,
                  r: float = RISK_FREE,
                  q: float = DIVIDEND_YIELD) -> GammaProfile:
    """Dealer gamma by strike, computed from OUR greeks.

    `spot` overrides the chain's spot so the same open interest can be
    revalued at a hypothetical price - which is what `zero_gamma` needs, and
    the reason it cannot just read a single number off the current profile.
    """
    as_of = chain.as_of_date
    S = chain.spot if spot is None else spot
    ivs = _iv_lookup(chain, max_dte)

    acc: dict[float, list[float]] = {}
    used = skipped = 0
    for c in chain.for_gex(max_dte=max_dte):
        T = c.time_to_expiry(as_of)
        sigma = ivs.get((c.expiry, c.strike), 0.0)
        if T <= 0 or sigma <= 0:
            skipped += 1
            continue
        g = bs_gamma(S, c.strike, T, r, sigma, q)
        if g <= 0:
            skipped += 1
            continue
        used += 1
        signed = dealer_sign(c, convention) * gex_dollars(
            g, c.open_interest, S)
        row = acc.setdefault(c.strike, [0.0, 0.0, 0.0])
        row[0] += signed
        if c.is_call:
            row[1] += c.open_interest
        else:
            row[2] += c.open_interest

    strikes = tuple(
        StrikeGamma(strike=k, gex=v[0], call_oi=v[1], put_oi=v[2])
        for k, v in sorted(acc.items()))
    return GammaProfile(
        underlying=chain.underlying, spot=S, as_of=as_of,
        convention=convention, strikes=strikes,
        contracts_used=used, contracts_skipped=skipped)


# ---------------------------------------------------------------- flip level


def net_gex_at(chain: OptionChain, spot: float, **kw) -> float:
    """Total dealer gamma if spot were `spot`, same open interest."""
    return gamma_profile(chain, spot=spot, **kw).net_gex


def zero_gamma(chain: OptionChain, *,
               convention: Convention = Convention.DEALER_LONG_CALLS,
               max_dte: int | None = None,
               lo_mult: float = 0.85, hi_mult: float = 1.15,
               tol: float = 0.01, max_iter: int = 60,
               r: float = RISK_FREE, q: float = DIVIDEND_YIELD) -> float | None:
    """The spot at which net dealer gamma changes sign.

    Bisection on `net_gex_at`. Gamma is a function of spot, so the profile has
    to be REVALUED at each candidate price - you cannot read this level off the
    current profile by finding where calls stop outnumbering puts. That
    shortcut is common and wrong.

    Returns None when the bracket does not contain a sign change, which is a
    real and frequent state: a book can be net long gamma at every price in a
    +/-15% band. None means "no flip here", never "flip at zero".
    """
    kw = dict(convention=convention, max_dte=max_dte, r=r, q=q)
    lo, hi = chain.spot * lo_mult, chain.spot * hi_mult
    f_lo, f_hi = net_gex_at(chain, lo, **kw), net_gex_at(chain, hi, **kw)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = net_gex_at(chain, mid, **kw)
        if hi - lo < tol or f_mid == 0.0:
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5 * (lo + hi)


# -------------------------------------------------------------- disagreement


@dataclass(frozen=True)
class GammaComparison:
    """Ours against the vendor's, per `options.py`'s standing promise."""

    n: int
    median_rel_error: float
    p90_rel_error: float
    worst: tuple[str, float, float, float] | None   # symbol, ours, theirs, rel


def compare(chain: OptionChain, *, max_dte: int | None = 45,
            near_the_money: float = 0.03,
            r: float = RISK_FREE,
            q: float = DIVIDEND_YIELD) -> GammaComparison | None:
    """Our gamma vs the feed's, out loud.

    Restricted to near-the-money contracts: in the deep wings gamma is close
    to zero, and a relative error there divides by nearly nothing and reports
    noise as disagreement.

    A systematic gap is more likely an r/q mismatch than a maths error, which
    is precisely why this reports rather than corrects.
    """
    as_of = chain.as_of_date
    ivs = _iv_lookup(chain, max_dte)
    rows: list[tuple[str, float, float, float]] = []

    for c in chain.for_pricing(max_dte=max_dte):
        if c.vendor_gamma is None or c.vendor_gamma <= 0:
            continue
        if abs(c.strike - chain.spot) / chain.spot > near_the_money:
            continue
        T = c.time_to_expiry(as_of)
        sigma = ivs.get((c.expiry, c.strike), 0.0)
        if T <= 0 or sigma <= 0:
            continue
        ours = bs_gamma(chain.spot, c.strike, T, r, sigma, q)
        theirs = c.vendor_gamma
        rows.append((c.symbol, ours, theirs, abs(ours - theirs) / theirs))

    if not rows:
        return None
    errs = sorted(x[3] for x in rows)
    mid = errs[len(errs) // 2]
    p90 = errs[min(int(0.90 * (len(errs) - 1)), len(errs) - 1)]
    return GammaComparison(n=len(rows), median_rel_error=mid,
                           p90_rel_error=p90,
                           worst=max(rows, key=lambda x: x[3]))


# ---------------------------------------------------------------- self-test


def verify() -> bool:
    """Numerical checks. Every formula earns trust by finite difference.

    The house rule from the research track: a closed form must be shown to be
    the derivative it claims to be, not asserted to be.
    """
    from datetime import datetime, timedelta, timezone

    fails: list[str] = []

    def check(label: str, got: float, want: float, tol: float = 1e-6) -> None:
        ok = abs(got - want) < tol
        if not ok:
            fails.append(label)
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: {got:.10f} "
              f"(want {want:.10f})")

    def check_true(label: str, cond: bool) -> None:
        if not cond:
            fails.append(label)
        print(f"  [{'ok' if cond else 'FAIL'}] {label}")

    S, K, T, r, sigma, q = 100.0, 100.0, 0.25, 0.042, 0.20, 0.012

    print("gamma is d(delta)/dS -- central difference")
    h = 1e-4

    def delta_call(s: float) -> float:
        d1 = ((math.log(s / K) + (r - q + 0.5 * sigma ** 2) * T)
              / (sigma * math.sqrt(T)))
        return math.exp(-q * T) * 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))

    fd = (delta_call(S + h) - delta_call(S - h)) / (2 * h)
    check("gamma vs FD of delta", bs_gamma(S, K, T, r, sigma, q), fd, 1e-9)

    print("\ngamma is also d2(price)/dS2 -- second difference")

    def price_call(s: float) -> float:
        sq = sigma * math.sqrt(T)
        d1 = (math.log(s / K) + (r - q + 0.5 * sigma ** 2) * T) / sq
        d2 = d1 - sq
        n = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        return s * math.exp(-q * T) * n(d1) - K * math.exp(-r * T) * n(d2)

    h2 = 1e-2
    fd2 = (price_call(S + h2) - 2 * price_call(S) + price_call(S - h2)) / (h2 ** 2)
    check("gamma vs 2nd FD of price", bs_gamma(S, K, T, r, sigma, q), fd2, 1e-6)

    print("\ndegenerate inputs return 0.0, never NaN")
    for label, args in (("expired", (S, K, 0.0, r, sigma, q)),
                        ("zero vol", (S, K, T, r, 0.0, q)),
                        ("zero spot", (0.0, K, T, r, sigma, q)),
                        ("negative T", (S, K, -1.0, r, sigma, q))):
        g = bs_gamma(*args)
        check_true(f"{label} -> 0.0", g == 0.0)

    print("\ngex_dollars scaling")
    # gamma 0.02, OI 1000, spot 100: 0.02*1000*100 = 2000 delta per $1.
    # A 1% move is $1, so this is also 2000*100*0.01 = 200,000 dollars.
    check("per-1% dollar scaling", gex_dollars(0.02, 1000, 100.0), 200000.0)
    check("scales linearly in OI",
          gex_dollars(0.02, 2000, 100.0) / gex_dollars(0.02, 1000, 100.0), 2.0)
    check("scales with spot squared",
          gex_dollars(0.02, 1000, 200.0) / gex_dollars(0.02, 1000, 100.0), 4.0)

    print("\ndealer sign conventions")
    mk = lambda right: OptionContract(
        symbol=f"X{right}", underlying="X", expiry=date(2026, 12, 18),
        strike=100.0, right=right, open_interest=1.0, iv=0.2, bid=1.0, ask=1.1)
    call, put = mk("C"), mk("P")
    check("long-calls: call +1", dealer_sign(call, Convention.DEALER_LONG_CALLS), 1.0)
    check("long-calls: put -1", dealer_sign(put, Convention.DEALER_LONG_CALLS), -1.0)
    check("short-calls: call -1", dealer_sign(call, Convention.DEALER_SHORT_CALLS), -1.0)
    check("short-calls: put +1", dealer_sign(put, Convention.DEALER_SHORT_CALLS), 1.0)
    check("all-long: call +1", dealer_sign(call, Convention.ALL_LONG), 1.0)
    check("all-long: put +1", dealer_sign(put, Convention.ALL_LONG), 1.0)
    check_true("the two dealer conventions are exact mirrors",
               all(dealer_sign(c, Convention.DEALER_LONG_CALLS)
                   == -dealer_sign(c, Convention.DEALER_SHORT_CALLS)
                   for c in (call, put)))

    # ---- an end-to-end book with a known answer -------------------------
    print("\nprofile and flip level on a constructed book")
    today = date(2026, 8, 29)
    expiry = today + timedelta(days=30)
    spot = 100.0

    def contract(right: str, strike: float, oi: float) -> OptionContract:
        return OptionContract(
            symbol=f"X{right}{strike:.0f}", underlying="X", expiry=expiry,
            strike=strike, right=right, bid=1.0, ask=1.1, last=1.05,
            volume=10.0, open_interest=oi, iv=0.20)

    # Calls stacked high, puts stacked low. Under DEALER_LONG_CALLS the book
    # is long gamma up top and short gamma down low, so a flip must exist
    # somewhere between the two clusters.
    contracts = tuple(
        [contract("C", k, 5000.0) for k in (105.0, 110.0, 115.0)]
        + [contract("P", k, 5000.0) for k in (85.0, 90.0, 95.0)])
    chain = OptionChain(
        underlying="X", spot=spot,
        fetched_at=datetime(2026, 8, 29, 20, 0, tzinfo=timezone.utc),
        contracts=contracts, source="synthetic", session_date=today)

    prof = gamma_profile(chain)
    check_true("every strike is represented", len(prof.strikes) == 6)
    check_true("all contracts were usable", prof.contracts_skipped == 0)
    check("gross gamma is positive under ALL_LONG",
          1.0 if gamma_profile(chain, convention=Convention.ALL_LONG).net_gex > 0
          else 0.0, 1.0)
    check_true("call strikes carry positive gex",
               all(s.gex > 0 for s in prof.strikes if s.strike > spot))
    check_true("put strikes carry negative gex",
               all(s.gex < 0 for s in prof.strikes if s.strike < spot))

    mirrored = gamma_profile(chain, convention=Convention.DEALER_SHORT_CALLS)
    check("mirroring the convention negates net gex",
          mirrored.net_gex, -prof.net_gex, tol=1e-6)

    flip = zero_gamma(chain)
    check_true("a flip level exists for this book", flip is not None)
    if flip is not None:
        residual = net_gex_at(chain, flip)
        scale = max(abs(net_gex_at(chain, chain.spot * 0.9)), 1.0)
        check_true(
            f"net gex at the flip is ~0 (residual {residual:,.0f} "
            f"vs {scale:,.0f} scale)", abs(residual) / scale < 1e-3)
        check_true(f"flip {flip:.2f} sits between the clusters",
                   95.0 < flip < 105.0)
        below, above = net_gex_at(chain, flip - 2), net_gex_at(chain, flip + 2)
        check_true("gamma really changes sign across it",
                   (below > 0) != (above > 0))

    print("\nno flip is reported as None, not as zero")
    only_calls = OptionChain(
        underlying="X", spot=spot,
        fetched_at=datetime(2026, 8, 29, 20, 0, tzinfo=timezone.utc),
        contracts=tuple(contract("C", k, 5000.0) for k in (105.0, 110.0)),
        source="synthetic", session_date=today)
    check_true("all-call book has no flip", zero_gamma(only_calls) is None)

    print("\nwalls")
    heavy = OptionChain(
        underlying="X", spot=spot,
        fetched_at=datetime(2026, 8, 29, 20, 0, tzinfo=timezone.utc),
        contracts=(contract("C", 105.0, 50000.0), contract("C", 110.0, 100.0),
                   contract("P", 95.0, 100.0)),
        source="synthetic", session_date=today)
    hp = gamma_profile(heavy)
    check("largest wall is the 50k-OI strike",
          hp.largest_walls(1)[0].strike, 105.0)
    check_true("nearest wall above spot is 105",
               hp.nearest_wall(above=True).strike == 105.0)
    check_true("nearest wall below spot is 95",
               hp.nearest_wall(above=False).strike == 95.0)

    print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
    return not fails


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
