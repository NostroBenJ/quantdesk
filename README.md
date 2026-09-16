# quantdesk

Modular trading research and execution system. The honesty infrastructure was
built first, on purpose: walk-forward, realistic costs, and a lookahead guard all
exist and are tested before a single signal module has been written.

**Status:** data layer, options-chain layer, backtest engine and research
dashboard complete and tested (114 tests). Signal, regime, and risk layers are
interfaces only — deliberately.

```bash
python -m pytest tests/ -q
python scripts/demo_pipeline.py
```

---

## Why this order

A backtest is a measuring instrument. Building strategies before validating the
instrument means every result afterwards is uninterpretable — you cannot tell a
real edge from a measurement artifact, and the artifacts all point the same way
(too optimistic).

So `scripts/demo_pipeline.py` runs a **coin flip** through the whole pipeline. Its
true edge is zero, which makes it the only strategy whose correct answer is known
in advance. On your real SPY 5-minute data it grades:

```
profit factor   gross     0.93   net     0.19
breakeven cost multiple: never profitable
walk-forward: 36 window(s), 0/36 profitable out-of-sample
VERDICT: FAIL
```

The instrument can detect the absence of edge. That is the prerequisite for
believing it when it detects presence.

## The three questions every strategy must answer

| Question | Mechanism | Where |
|---|---|---|
| Does the edge survive realistic costs? | gross vs net profit factor | `backtest/report.py` |
| How wrong can the cost estimate be first? | breakeven cost multiple | `CostSensitivity` |
| Does it survive out-of-sample? | walk-forward degradation | `backtest/walkforward.py` |

`grade_strategy()` returns PASS / MARGINAL / FAIL / INSUFFICIENT DATA with the
reasons attached. **MARGINAL does not graduate** — marginal cases are the ones
that feel promising and are not.

## Layout

```
quantdesk/
  core/         clock.py (UTC + annualisation), types.py, config.py     [zero deps]
  data/         bars.py (partial-bar policy), store.py (SQLite), source.py,
                options.py — chain schema + quality filters          [zero deps]
    sources/    yahoo.py, csv_source.py, cboe.py (free chain)   [lazy imports]
  backtest/     engine.py, view.py, lookahead.py, costs.py, fills.py,
                metrics.py, walkforward.py, report.py                   [zero deps]
  signals/      base.py — SignalModule ABC + per-module ledger
  risk/         base.py — PreTradeGate machinery (rules await your firm's doc)
  execution/    base.py — paper adapter + live adapter w/ hard confirmation gate
  narrative/    base.py — Claude narrator, advisory only, cannot trade
  obs/          logging.py — JSONL event log
  dash/         app.py, results.py — research review UI, reads runs/ only
```

The core and backtest packages have **no third-party dependencies**. `import
quantdesk` works in a bare interpreter. yfinance, pandas, PyYAML, and anthropic
are imported inside the functions that need them — the same pattern
`black_scholes.py::plot_greeks` already uses.

## The four anti-lookahead mechanisms

1. **`ts_close`.** A bar is identified by its open but only becomes *knowable* at
   its close. `MarketView` will not show a strategy any bar whose `ts_close`
   is after the current instant.
2. **Partial-bar policy.** The currently-forming bar is labelled and dropped by
   default. Vendors do not flag it; `data/bars.py` infers it.
3. **Next-bar execution.** The fill model is only ever handed `bars[i+1]`. There
   is no code path that fills at the close that generated the signal. An order on
   the final bar *expires* rather than filling.
4. **Numerical causality checks.** `assert_causal(feature, bars)` computes a
   feature twice — once with the series truncated at T, once with everything
   after T shocked — and fails if either result differs.

Mechanism 4 is the direct analogue of the finite-difference rule in the parent
`CLAUDE.md`. It catches three distinct classes of leak, and the tests prove it
catches each:

```python
def subtle_cheat(bars, i):
    return bars[i].close / max(b.close for b in bars)   # max() over the future
```

Nobody writes `bars[i+1]`. They write that, and never think about it again.

## What the research changed

Five design decisions came out of the papers in `research/` (full notes in
[`research/DIGEST.md`](research/DIGEST.md)):

- **Realized vol takes the bar timeframe as a required argument** (Bergomi Ch.1,
  fn.1 — RV is only defined relative to the return time-scale). Annualising
  5-minute bars with `sqrt(252)` instead of `sqrt(19656)` understates vol 8.8x
  and oversizes every position by the same factor.
- **Slippage is sign-aware** (Bennett): long gamma can rest on the bid and offer;
  short gamma must cross. Charging both the same spread flatters short-vol.
- **Short-dated option costs belong in cash terms, not vol terms** (Bennett).
- **Fill probability decays exponentially in distance from mid**,
  `p = exp(-kappa*d)` (Lucic & Tse 2024) — `ExponentialQueueFill`.
- **The regime layer gets the Skew Stickiness Ratio**, a measurable statistic
  independent of GEX, with named endpoints at 0 / 1 / 2 (Bergomi 2.61, Derman L9).

Bergomi and Lucic & Tse independently derive the same expression for options
edge: `(sigma_realized^2 - sigma_implied^2)/2 * dollar_gamma`. That is the
quantity any options signal module has to estimate.

## Options data, subscription-free

The Unusual Whales subscription lapsed and is not being renewed. It turns out
almost nothing was lost, because UW supplied the *chain* and a cross-check — it
never did the maths.

`data/sources/cboe.py` reads CBOE's public delayed-quotes endpoint: no key, no
pagination, **14,100 SPY contracts in one request** (UW's paged walk returned
13,958). It also carries **bid/ask**, which UW did not — so the option-side
spread becomes a measurement instead of the volatility-scaled estimate the bar
cost model has to fall back on.

```bash
python scripts/check_chain.py SPY
```

That script is the replacement for UW's `gex-levels` cross-check: it prices
near-the-money gamma with `black_scholes.py` and compares against CBOE's own
published greeks. Current agreement is **median 1.21%, p99 11.6%** across 941
contracts. Getting there required fixing two real bugs the comparison exposed:

1. **Use the OTM wing's IV at each strike.** An ITM option is nearly all
   intrinsic, so its IV is an artifact of the bid/ask spread. CBOE's own numbers
   contradict themselves here — at K=779 their call gamma was 0.00510 and their
   put gamma 0.00010, a factor of 51 at the same strike, where put-call parity
   requires them to be identical. `OptionChain.parity_breaks()` turns that
   identity into a free data-quality check.
2. **An overnight fetch belongs to the previous session.** 04:29 UTC is 00:29
   ET. Taking the UTC date understated every DTE by one — 33% of `T` on a 3-DTE
   option, which showed up as our gamma running 3x below CBOE's.

Gone for good with the subscription: flow alerts, dark pool prints, net-premium
ticks. Those are proprietary aggregations and are not recoverable from public
data.

## Dashboard

```bash
python scripts/make_demo_run.py     # writes evaluation runs to runs/
python scripts/run_dash.py          # http://127.0.0.1:8010
```

Structure lifted from `nyam_bias` — FastAPI + Jinja + vanilla JS, no build step —
with the pre-market bias analysis removed and the honesty panels in its place.

**The dashboard computes nothing.** Every figure is read from a run JSON written
by `backtest/`, so there is no second implementation that can drift from the one
that produced the grade. Panels are ordered by how much they tell you: verdict,
gross-vs-net, cost sensitivity, walk-forward, and the equity curve *last* —
it is the prettiest panel and the least informative, so it does not get to lead.

Two calibration runs ship with it, and deliberately neither is a PASS (a PASS
example would have to be constructed to pass, which is the habit this system
exists to break):

| run | grade | what it pins |
|---|---|---|
| CoinFlip control | FAIL, PF 0.19 over 10,004 trades | the report can detect the absence of edge |
| BuyAndHold | INSUFFICIENT DATA, PF ∞ on n=1 | the sample-size guard fires *before* the headline — an infinite PF looks like a discovery |

The CoinFlip run also surfaces ruin: with fixed size and no risk limits, equity
went to −190,535. The report says **RUINED** rather than "290% drawdown",
because past zero a drawdown percentage is arithmetic on an account that would
already have been closed out.

## Graduation gates

```
backtest  --> paper : PASS verdict required (walk-forward + cost sensitivity)
paper     --> live  : minimum sample size YOU specify + human sign-off
```

`LiveExecutionAdapter.submit()` raises `ConfirmationRequired` unless a human has
confirmed that specific order. The confirmation token includes side, quantity,
symbol and timestamp, so it cannot be reused for a different order. There is no
bypass flag — a confirmation gate that can be disabled is not a gate.

## What is not modelled (and will flatter you)

Stated explicitly because unstated assumptions are what turn a 1.3 into a 0.8:

- **No bid/ask in free data.** Every spread is an *estimate*
  (`spread_vol_fraction`, default 0.15). This is the largest single source of
  error in any net P&L reported here. Calibrate against real fills immediately.
- Queue position, and adverse selection on resting orders.
- Borrow cost on shorts, and financing.
- Yahoo prices are retroactively split/dividend adjusted — the series you
  backtest is not the series that printed at the time.
- Yahoo intraday history is capped at ~30 days; `check_request` raises rather
  than silently truncating your backtest window.

Every such assumption in the code is tagged `ASSUMPTION (...)` — grep for it.

## Next

1. Prop-firm rule documents → `risk/` checks. The *wording* of the
   trailing-drawdown definition matters more than the number.
2. GEX regime module on top of `data/options.py`.
3. Then signal modules, each benchmarked against the controls above.

Nothing in step 3 is worth writing before steps 1 and 2, and none of it was worth
writing before this.

The strategy screens run since then live in `scripts/screen_*.py`, with outputs in
`results/`. Each test was pre-registered in its own commit before the result was
seen, so `git log --oneline` shows the pre-registration and the result separately.

Market data is not included (licensed vendors: Databento, London Strategic Edge,
CBOE DataShop). The loaders in `quantdesk/data/sources/` show the expected formats.
