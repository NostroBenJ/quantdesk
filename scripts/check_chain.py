"""Live smoke check of the free CBOE chain source, plus a gamma cross-check.

    python scripts/check_chain.py [SYMBOL]

Three things it establishes, in order of how much they matter:

1. The free feed actually returns a full, parseable chain (no key, no pagination).
2. The observed half-spread near the money - which turns `backtest/costs.py`
   from a volatility-scaled *estimate* into a *measurement*.
3. Our gamma versus CBOE's published gamma, contract by contract.

Point 3 is the replacement for the Unusual Whales `gex-levels` cross-check. The
rule from `options-math` applies unchanged: **a disagreement is a detector, not
a correction**. Never resolve a gap by overwriting ours with theirs. Two
computations that agree is evidence; one silently replacing the other is not.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# black_scholes.py lives in the research track one level up. Imported here and
# nowhere in the package, so quantdesk itself stays self-contained.
sys.path.insert(0, str(ROOT.parent))

from quantdesk.data.sources.cboe import CboeChainSource  # noqa: E402

# ASSUMPTION (market data): CBOE does not publish the r/q it used for its
# greeks. A systematic gamma gap may be a rate/dividend assumption mismatch
# rather than a maths error - which is exactly why we compare rather than adopt.
RISK_FREE = 0.042
DIVIDEND_YIELD = 0.012


def main(symbol: str = "SPY") -> int:
    try:
        from black_scholes import bs_greeks
    except ImportError:
        bs_greeks = None
        print("note: black_scholes.py not importable - skipping the gamma cross-check\n")

    source = CboeChainSource()
    print(f"fetching {symbol} chain from {source.name} ...")
    chain = source.fetch_chain(symbol)

    print(f"\nunderlying   {chain.underlying}  spot {chain.spot:.2f}", end="")
    if chain.spot_bid and chain.spot_ask:
        print(f"  (bid {chain.spot_bid:.2f} / ask {chain.spot_ask:.2f})")
    else:
        print()
    print(f"fetched_at   {chain.fetched_at.isoformat()}  [ours, UTC]")
    print(f"vendor stamp {chain.vendor_timestamp_raw!r}  [timezone unverified - not parsed]")
    print(f"expiries     {len(chain.expiries)}   first {chain.expiries[0]}  last {chain.expiries[-1]}")

    print("\nquality report")
    print("-" * 60)
    report = chain.quality_report()
    for key, value in report.items():
        pct = 100.0 * value / report["total"] if report["total"] else 0.0
        print(f"  {key:<22} {value:>7,}  {pct:>5.1f}%")
    print(f"  {'total open interest':<22} {chain.total_open_interest():>7,.0f}")

    print("\nobserved half-spread near the money")
    print("-" * 60)
    for dte in (1, 7, 30):
        half = chain.observed_half_spread(max_dte=dte)
        if half is None:
            print(f"  <= {dte:>3}d   no qualifying contract (correctly returns None)")
        else:
            print(f"  <= {dte:>3}d   {half:.4f}  ({half / chain.spot * 1e4:.2f} bps of spot)")
    print("  ^ this is a MEASUREMENT. costs.spread_vol_fraction is an estimate")
    print("    used only where no quote exists. Prefer this wherever a chain is available.")

    if bs_greeks is None:
        return 0

    print("\nfeed self-consistency: call vs put gamma at the same strike")
    print("-" * 60)
    breaks = chain.parity_breaks(max_dte=45)
    comparable = len({(c.expiry, c.strike) for c in chain.within_dte(45)})
    print(f"  {len(breaks)} parity break(s) across {comparable} strike/expiry pairs within 45 DTE")
    print("  (gamma is identical for a call and a put at one strike - by identity,")
    print("   so every row below is a fault in the FEED, not in the market)")
    for expiry, strike, call_g, put_g in breaks[:5]:
        ratio = max(call_g, put_g) / min(call_g, put_g)
        print(f"    {expiry}  K={strike:>7.1f}  call {call_g:.5f}  put {put_g:.5f}  ({ratio:.0f}x)")

    print("\ngamma cross-check: ours (black_scholes.py) vs CBOE's published")
    print("-" * 60)
    today = chain.as_of_date
    # Use the OTM wing's IV at each strike. An ITM option's IV is an artifact of
    # its bid/ask spread, and feeding it to bs_greeks makes OUR gamma wrong too.
    otm_iv = chain.otm_iv_by_strike(max_dte=45)
    diffs: list[float] = []
    rows: list[tuple] = []

    for contract in chain.for_pricing(max_dte=45):
        if contract.vendor_gamma is None or contract.vendor_gamma <= 0:
            continue
        t = contract.time_to_expiry(today)
        sigma = otm_iv.get((contract.expiry, contract.strike), 0.0)
        if t <= 0 or sigma <= 0:
            continue
        # Near the money only: deep wings have gamma close to zero, where a
        # relative comparison divides by nearly nothing and reports noise.
        if abs(contract.strike - chain.spot) / chain.spot > 0.03:
            continue
        ours = bs_greeks(
            S=chain.spot, K=contract.strike, T=t, r=RISK_FREE,
            sigma=sigma, q=DIVIDEND_YIELD,
            kind="call" if contract.is_call else "put",
        ).gamma
        theirs = contract.vendor_gamma
        rel = abs(ours - theirs) / theirs
        diffs.append(rel)
        rows.append((contract.symbol, contract.strike, contract.right, ours, theirs, rel))

    if not diffs:
        print("  no comparable contracts (needs live expiries with two-sided quotes)")
        return 0

    rows.sort(key=lambda r: -r[5])
    diffs.sort()
    median = statistics.median(diffs)
    p90 = diffs[int(0.90 * (len(diffs) - 1))]
    p99 = diffs[int(0.99 * (len(diffs) - 1))]

    print(f"  compared {len(diffs)} near-the-money contracts within 45 DTE")
    print(f"  median {median:>8.2%}    p90 {p90:>8.2%}    p99 {p99:>8.2%}    worst {max(diffs):>8.2%}")
    print(f"\n  {'contract':<22}{'K':>8} {'R':>2} {'ours':>12} {'cboe':>12} {'rel':>8}")
    for symbol, strike, right, ours, theirs, rel in rows[:5]:
        print(f"  {symbol:<22}{strike:>8.1f} {right:>2} {ours:>12.6f} {theirs:>12.6f} {rel:>7.1%}")

    # The verdict reads the TAIL, not just the median. A median of 2.5% with a
    # p99 of 5000% is not agreement - it is agreement everywhere except the
    # place where something is broken, and reporting only the median hides it.
    print()
    if median < 0.05 and p99 < 0.25:
        print("  VERDICT: agrees, tail included. Independent confirmation of the")
        print("  gamma maths - what UW's gex-levels used to provide, for free.")
    elif median < 0.05:
        print(f"  VERDICT: agrees in the body ({median:.1%}), breaks in the tail ({p99:.0%} at p99).")
        print("  Do NOT read this as a pass. Check the parity-break list above -")
        print("  a strike appearing in both is a contract whose feed IV is unusable,")
        print("  and its gamma must be excluded from GEX rather than trusted.")
    else:
        print("  VERDICT: DISAGREES. Do not 'fix' this by adopting CBOE's number.")
        print("  Most likely causes, in order: the r/q assumption above, CBOE's")
        print("  IV being inverted from a stale mid, or a real bug in ours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "SPY"))
