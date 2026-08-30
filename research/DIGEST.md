# Research digest — what each source actually gives us

Written 2026-08-20. Every item below is tagged with the module it feeds.
Sources are in `research/papers/`, raw text extractions in `research/notes/`.

---

## 1. Bergomi — *Stochastic Volatility Modeling*, Ch.1 (Introduction)

**The one equation that matters.** Ch.1 derives Black-Scholes not from Brownian
motion but from break-even accounting. Delta-hedge a short option over `dt`,
expand to order `dt` and order `dS^2`, and the carry P&L collapses to:

    P&L  =  -(1/2) * S^2 * d2P/dS2 * [ (dS/S)^2  -  sigma_hat^2 * dt ]

i.e. **P&L per period = -(1/2) * dollar_gamma * (realized_var - implied_var)**.

Consequences we use directly:

- **This is the VRP, per-trade.** `vrp_study.py` already measures the aggregate
  version. This is the same quantity at position level, and it is the correct
  P&L attribution for any options position we hold.
- **Break-even move.** P&L is zero when `|dS/S| = sigma_hat * sqrt(dt)`. That is a
  cost-free, non-arbitrary definition of "how far must spot move for a gamma
  position to pay" — a natural stop/target scale. Feeds `risk/sizing.py`.
- **Realized vol must be measured at the hedging frequency.** Bergomi's footnote 1
  is blunt: serial correlation is irrelevant to pricing, but *the measure of
  realized volatility depends on the time scale of returns used*, and the relevant
  scale is the delta-hedge frequency. So our RV estimator takes the bar timeframe
  as a parameter and never silently defaults to daily. Implemented in
  `data/bars.py::realized_volatility`.
- **Asymmetry of short gamma.** Short gamma: gain bounded, loss unbounded. Desks
  therefore shift bid/offer away from an unbiased vol forecast. The risk layer
  must not size long and short gamma symmetrically.

**Model-usability test.** A pricing function is usable only if theta and gamma have
opposite signs everywhere; otherwise you either bleed forever or print free money.
A cheap invariant to assert over any surface we fit.

## 2. Bergomi Ch.2 — Local Volatility, and the Skew Stickiness Ratio

The SSR is the transferable result:

    R_T = (1 / S_T) * d(sigma_ATMF_T) / d(ln S)                     (2.61)

— how much ATM implied vol moves per unit spot move, **normalised by the ATM skew**.
Generalised (2.62), it is the regression coefficient of ATMF vol on `ln S`, over skew.

- `R = 1` → **sticky strike** (fixed-strike vols frozen; ATM slides along the smile)
- `R = 0` → **sticky delta** (whole smile translates with spot)
- Local vol gives `R_T = 1 + (1/T) * integral(S_t / S_T dt)` ≈ **2** for short dates
  (2.64) — which Bergomi notes *overestimates* when skew is strong.

**Why we care:** SSR is a measurable, non-black-box regime statistic. We already
have SPY+VIX daily joined in `spy_vix_daily.csv`. Regressing daily `d(ATM IV)` on
`d(ln S)` over a rolling window and dividing by measured skew yields a live number
in roughly `[0, 2]` saying which vol regime we are in — and it is *independent* of
the GEX signal, so it is a genuine cross-check rather than a restatement of it.

Bergomi also notes realized SSR sits **above 1**: neither sticky-strike nor
sticky-delta describes reality, which is exactly why measuring beats assuming.

## 3. Derman — E4718 Lecture 9, *Patterns of Volatility Change*

Same territory as Bergomi Ch.2 from the desk side, and it supplies the
regime-switching claim we can actually test:

> during calm upward-trending periods the market satisfied the sticky strike rule,
> and during fearful periods it comes closer to satisfying the sticky implied tree rule.

Three rules, under the linear-skew approximation `Sigma(S,K) = Sigma_0 - b(K - S_0)`:

| Rule | Future smile | Implied model |
|---|---|---|
| Sticky strike | `Sigma(S,K) = f(K)` | Black-Scholes-ish (can't honestly carry a skew) |
| Sticky delta / moneyness | `Sigma(S,K) = f(K/S)` | jump-diffusion, stochastic vol |
| Sticky implied tree | `Sigma(S,K) ≈ Sigma_0 - b(K + S - 2*S_0)` | local volatility |

Note the third row: ATM vol moves at **twice** the skew rate as spot moves — the
`SSR = 2` of Bergomi (2.64). Two independent sources, same number. The regime
classifier therefore gets a defensible 0/1/2 scale with named endpoints.

Derman is also explicit that none of these rules hold over long periods; they are
short-horizon descriptions. So: short lookback, re-estimate every bar, never
persist a regime label.

## 4. Lucic & Tse (2024) — *Optimal Option Market Making and Volatility Arbitrage*

The SSRN PDF is Cloudflare-blocked to automated fetch and the Risk.net version is
paywalled — **but the authors publish a companion Colab notebook**, and we have it:
`research/papers/lucic-tse-colab.ipynb`, their own reference implementation.

Core value-function PIDE, from the notebook:

    dh/dt + (sigma^2 - sigma_imp^2)/2 * DollarGamma(t,s) * q  -  beta*q^2
          + (sigma^2/2) * s^2 * d2h/ds2
          + (lambda_a * e^-1 / kappa_a) * exp[-kappa_a * (h(t,s,q) - h(t,s,q-1))]
          + (lambda_b * e^-1 / kappa_b) * exp[-kappa_b * (h(t,s,q) - h(t,s,q+1))] = 0

Three things we lift out of it:

1. **The edge term is Bergomi's equation again**:
   `(sigma^2 - sigma_imp^2)/2 * DollarGamma * q`. Two independent sources
   converging on the same expression for options edge is worth more than either
   alone. Any options-side signal module must produce an estimate of this term.
2. **The fill model.** Order arrival is `lambda(delta) = lambda_0 * exp(-kappa * delta)`:
   fill probability decays **exponentially** in how far from mid you quote. Their
   calibration convention is the useful part — `kappa` chosen so that average order
   count halves when depth increases by one vol point. This is a far more honest
   paper-fill model than "filled at mid" or "filled at the signal price", and it is
   what `backtest/fills.py` implements in generalised form.
   **The single most directly usable item in the paper.**
3. **Inventory penalty structure**: running `beta*q^2` plus terminal `alpha*q^2`;
   optimal quotes skew away from mid as inventory builds. The same algebra applies
   to any position-limit-aware sizing, not just market making.

Their parameter block is a useful magnitude anchor: `lam = 252*50` (50 fills/day/side),
`kappa = 0.75/(0.01*vega)`, horizon `T = 2/252` (half a business day).

## 5. Bennett — *Trading Volatility* (Santander; free from the author)

Practitioner reference, 317pp. The parts that changed our code:

- **Long gamma can sit on the bid and offer; short gamma must cross the spread.**
  Delta-hedging cost is *sign-dependent*. A cost model charging both the same
  spread is wrong in the direction that flatters short-vol strategies — precisely
  the error we are trying to eliminate. `backtest/costs.py` takes `crosses_spread`.
- **Bid-offer is more stable in cash terms than in vol terms across maturities.**
  Short-dated options have tiny premium, so a small cash move is a large IV move;
  the IV bid-offer channel is wide and near-useless there. For 0–2 DTE SPY (what
  the CISD work trades) **model cost in cash/premium terms, not vol terms.**
- **Hedge frequency is regime-dependent**: trending → hedge less often;
  choppy/range-bound → hedge more often. A fixed rehedge schedule mis-prices both.
- **Backtest warning, stated plainly**: the optimal call-overwriting strike depends
  entirely on whether the sample window had a positive or negative index return. A
  parameter that looks optimal in-sample is often just a proxy for the sample's
  drift. That is the walk-forward argument, from a practitioner rather than a
  statistician.
- Rules of thumb worth keeping as assertions: `annual vol ≈ 16 × daily % move`
  (since `sqrt(252) ≈ 16`); **skew should decay by sqrt(time)**; a put spread must
  always cost more than zero. Cheap surface-sanity checks.

## 6. ArturSepp/StochVolModels

MIT, 232 stars, actively maintained (pushed 2026-08-20), by the co-author of the
log-normal SV model it implements. Fourier/MGF pricing plus Monte-Carlo validation
for Heston and Karasinski-Sepp log-normal SV, and calibration against real chains
(`examples/calibration/load_cboe_option_chain.py`).

**How we use it: as a reference oracle, not a dependency.** Its value is that it
prices the same vanillas by a completely different route — Fourier transform of the
MGF, plus an independent Monte-Carlo. That is a stronger check on `black_scholes.py`
than another closed form would be, and it is the same philosophy as the
finite-difference rule already in CLAUDE.md: verify numerically, by an independent
method.

His README points at `qis` (QuantInvestStrats) for backtesting. Worth a look later,
but we are deliberately writing our own engine: the honesty constraints
(walk-forward by default, cost sensitivity, lookahead guard) are the entire point
and are not something to outsource.

---

## What this means for the build

The sources converge on one thing: **options edge = (realized variance − implied
variance) × dollar gamma**, and everything else is cost, inventory, and regime.
Two of the five derive that expression independently.

Concretely, the research changed five decisions in the code that is now written:

1. Realized-vol estimators take the bar timeframe as an explicit parameter, and
   annualise from it (Bergomi fn.1) — `data/bars.py`.
2. Slippage is sign-aware: short gamma crosses the spread, long gamma need not
   (Bennett) — `backtest/costs.py`.
3. Short-dated option costs are modelled in cash, not vol terms (Bennett) —
   documented on `OptionCostModel`.
4. The paper-fill model uses exponential fill-probability decay in distance from
   mid, `p = exp(-kappa * delta)`, rather than assuming the quoted price
   (Lucic & Tse) — `backtest/fills.py::ExponentialQueueFill`.
5. The regime layer gets a measurable statistic independent of GEX — the Skew
   Stickiness Ratio, with named endpoints at 0, 1 and 2 (Bergomi 2.61, Derman L9).
