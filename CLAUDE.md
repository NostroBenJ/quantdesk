# CLAUDE.md — quantdesk

Modular trading research and execution system. The honesty infrastructure exists
before the strategies, and that ordering is the point of the project.

Inherits the spirit of the parent `Downloads/CLAUDE.md`: verify numerically, do
not assert. This file adds what is specific to a system that places orders.

## Non-negotiables

**Python 3.11+, type hints throughout, pytest for every module.** No exceptions.

**The core is dependency-free.** `core/`, `backtest/`, and the pure parts of
`data/` import nothing outside the stdlib. yfinance, pandas, PyYAML, pyarrow and
anthropic are imported *inside the function that needs them*, so `import
quantdesk` works in a bare interpreter. This is the same rule as `plot_greeks()`
in `black_scholes.py`, extended to the whole system.

**No parameter is a literal.** Everything tunable lives in `config/`. If you find
yourself typing a number into a strategy, it belongs in YAML.

**Comment every assumption about microstructure, data availability, or broker
behaviour** with the literal tag `ASSUMPTION (...)`. They are greppable on
purpose — they are the list of things that have not been checked against reality.

## The causality rule

Every feature gets a causality test, the same way every pricing function gets a
finite-difference test:

```python
assert_causal(my_feature, bars)   # truncation + future-perturbation
```

Three lines. It catches leaks that code review does not, because the realistic
bug is not `bars[i+1]` — it is `max(b.close for b in bars)` inside a
normalisation, written once and never looked at again.

Run both checks, not one. Truncation catches length-dependence that perturbation
misses; perturbation catches value-dependence that truncation can miss. There is
a test that proves they catch different things — do not "simplify" it away.

## Time

- Everything is UTC. `ensure_utc` **rejects** naive datetimes rather than
  assuming — a silently localised series is shifted by the author's offset, which
  then looks like alpha.
- A bar is identified by `ts_open` and becomes knowable at `ts_close`. Only
  `ts_close` is ever compared against "now".
- `Timeframe.periods_per_year` is never a magic 252. For 5-minute bars it is
  19,656. Getting this wrong understates vol by 8.8x and oversizes every position
  by the same factor.

## Execution

- Decisions at bar `i`'s close fill inside bar `i+1`. The fill model is never
  handed the decision bar. An order on the final bar **expires**.
- Fill prices are clamped into the execution bar's range. An unclamped slippage
  estimate can manufacture profit on a short.
- Fills are deterministic by default. Random fills make a parameter change
  indistinguishable from a seed change, which makes walk-forward unfalsifiable.

## Costs

Slippage is a function of volatility and size, never a constant. A flat
per-trade assumption is roughly right in calm markets and badly wrong in fast
ones — and signals fire disproportionately in fast markets, so the error is
correlated with activity in exactly the direction that flatters the result.

Sign matters: long gamma can rest on the bid and offer, short gamma must cross
(Bennett). `crosses_spread` is a parameter.

## Reporting

Never report a net figure without the gross figure adjacent to it. The gap
between them *is* the finding. The cost-sensitivity table exists because an
equity curve is very good at hiding a PF that drifts from 1.2 gross to 0.8 net.

Every statistic carries its sample size. Below 30 trades, `is_meaningful` is
False and the verdict is INSUFFICIENT DATA — not a small PASS.

`MARGINAL` does not graduate to paper. That is the whole reason the grade exists.

## The narrative layer cannot trade

`narrative/` takes a frozen snapshot and returns `str`. There is no path by which
its output re-enters a decision. Model output is not reproducible, so it never
touches a backtest. If you are tempted to let it gate a trade, don't — the
signal and risk layers are deterministic and backtestable precisely so that the
thing making decisions is the thing that was measured.

## Testing style

Tests assert against hand-computed values and closed forms, not against whatever
the code printed the first time it ran.

**A test that cannot fail proves nothing.** Two tests here were initially written
against a synthetic series where each bar opened exactly at the previous close —
so "fills at next open, not this close" and "ATR is gap-aware" both passed
vacuously. Both now construct explicit gaps. When you write a test, ask what data
would make it fail, and use that data.
