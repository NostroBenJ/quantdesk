"""
screen_library.py -- the QuantConnect strategy-library shortlist, corrected
as one family.

PRE-REGISTERED. Every hypothesis below was written before any statistic was
computed. Holdout: the last 30% of sessions is SEALED and never read here.

WHY A SHORTLIST AND NOT ALL 83
------------------------------
Eighty-three tests at alpha=0.05 produce about four false positives by
construction, and picking the best-looking survivor from that is trading
noise with extra steps. The honest correction for 83 would be |t| > 3.48.

Sixty-three of the 83 are not testable or not tradeable here anyway: the
equity cross-sectional strategies need fundamentals, a survivorship-free
universe and the ability to short; forex and commodity futures need
accounts that are gone; the ML and sentiment ones need alternative data.

Two more are dropped for stated reasons rather than convenience:

* January Barometer -- one observation per year, roughly 19 in the sample.
  Structurally underpowered before it starts.
* Volatility Risk Premium (straddles) -- no historical option prices, and
  the VIX-versus-realised version of this question has already been run
  twice in this project.

Six remain. Correcting across six gives a demanding but survivable bar
instead of an impossible one, and the family is fixed in advance so the
survivor cannot be chosen after the fact.

EVERY TEST IS A DIFFERENCE, NEVER A LEVEL
-----------------------------------------
SPY drifts up, so every calendar bucket has a positive mean and "this
window is positive" is not a finding -- it is the equity risk premium
showing up wherever you look. Each hypothesis is therefore the CONTRAST
between the strategy's days and the days it sits out, which is the only
form of the question that can be wrong.

THE HYPOTHESES

L1  PRIMARY. Overnight anomaly. Decompose each session into close-to-open
    and open-to-close. Prior: the overnight leg carries nearly all of the
    equity premium and the intraday leg is roughly flat. Among the most
    replicated facts in equities, across decades and dozens of markets.

L2  Turn of the month. The last session of a month plus the first three
    against every other session.

L3  Pre-holiday effect. The session before a market holiday against every
    other session. Holidays are INFERRED from weekdays with no trading,
    not hard-coded, so the calendar cannot drift stale.

L4  VIX predicts index returns. Forward 21-session SPY return by VIX
    percentile. Prior: high VIX predicts higher forward returns.

L5  Sector momentum. Top three 12-month-momentum SPDR sectors against the
    bottom three, monthly, non-overlapping.

L6  Asset-class trend following. Each of six ETFs held only above its
    10-month moving average, against buy-and-hold. NOTE: this strategy's
    published claim is drawdown reduction rather than higher return, so a
    null on RETURN is not a refutation of it -- stated in advance.

COSTS, charged to every trade
-----------------------------
SPY round trip about 0.5bp (a cent of spread on a ~$600 share, plus
slack); sector and cross-asset ETFs about 2bp. L1 in particular trades
every session, so 250 round trips a year is ~125bp of annual cost -- the
overnight premium has to clear that, and it is charged rather than waved
past.

ASSUMPTION (data availability): prices are RAW, not dividend-adjusted, and
this is deliberate. An overnight return computed from an adjusted close to
a raw open books every dividend as a fake overnight gap. Keeping both legs
raw removes that. It leaves SPY's ~1.3% annual dividend as a real gap down
at each ex-date open, which biases the overnight leg DOWNWARD -- against
L1's hypothesis. If overnight still wins, it wins conservatively.

Run: python scripts/screen_library.py
"""

from __future__ import annotations

import datetime as dt
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.etf_daily import (  # noqa: E402
    ASSETS, SECTORS, build, market_holidays,
)
from quantdesk.data.research_stats import (  # noqa: E402
    Screen, describe_split, welch_overlap,
)

HOLDOUT_FRACTION = 0.30
COST_SPY_BP = 0.5
COST_ETF_BP = 2.0
VIX_CSV = ROOT / "spy_vix_term.csv"


def load_vix() -> dict[dt.date, float]:
    import csv
    if not VIX_CSV.exists():
        return {}
    with VIX_CSV.open(encoding="utf-8") as fh:
        return {dt.date.fromisoformat(r["date"]): float(r["vix"])
                for r in csv.DictReader(fh) if r["vix"]}


def main() -> int:
    data = build()
    spy = data["SPY"]
    days = sorted(spy)
    cut = int(len(days) * (1 - HOLDOUT_FRACTION))
    ins = days[:cut]
    holidays = market_holidays(days)

    print("SPY {0:,} sessions; in-sample {1:,} ({2} .. {3}); holdout {4:,} "
          "SEALED".format(len(days), len(ins), ins[0], ins[-1],
                          len(days) - cut))
    print("Six strategies, pre-registered as one family. Every test is a "
          "DIFFERENCE, never a level.\n")

    screen = Screen(cost_bp=COST_SPY_BP)

    # ---------------------------------------------------------------- L1
    print("L1  PRIMARY: overnight anomaly (close-to-open vs open-to-close)")
    overnight, intraday = [], []
    for a, b in zip(ins, ins[1:]):
        overnight.append(10000.0 * (spy[b]["open"] / spy[a]["close"] - 1.0))
        intraday.append(10000.0 * (spy[b]["close"] / spy[b]["open"] - 1.0))
    b1 = {"overnight": overnight, "intraday": intraday}
    for k in b1:
        screen.record(k, b1[k], descriptive=True)
    describe_split(b1, "overnight", "intraday", screen=screen,
                   label="L1 overnight minus intraday")
    yrs = len(overnight) / 252.0
    print("    annualised: overnight {0:+.1f}%  intraday {1:+.1f}%  "
          "(cost of trading it nightly: -{2:.1f}%/yr)".format(
              sum(overnight) / 100.0 / yrs, sum(intraday) / 100.0 / yrs,
              252 * COST_SPY_BP / 100.0))
    print()

    # ---------------------------------------------------------------- L2
    print("L2  turn of the month (last session + first three)")
    by_month: dict[tuple, list] = {}
    for d in ins:
        by_month.setdefault((d.year, d.month), []).append(d)
    tom = set()
    months = sorted(by_month)
    for i, key in enumerate(months):
        tom.add(by_month[key][-1])
        if i + 1 < len(months):
            tom.update(by_month[months[i + 1]][:3])
    b2: dict[str, list] = {"turn of month": [], "other days": []}
    for a, b in zip(ins, ins[1:]):
        r = 10000.0 * (spy[b]["close"] / spy[a]["close"] - 1.0)
        b2["turn of month" if b in tom else "other days"].append(r)
    for k in b2:
        screen.record(k, b2[k], descriptive=True)
    describe_split(b2, "turn of month", "other days", screen=screen,
                   label="L2 turn of month")
    print()

    # ---------------------------------------------------------------- L3
    print("L3  pre-holiday effect ({0} holidays inferred from the calendar)"
          .format(len(holidays)))
    pre = set()
    for i, d in enumerate(ins[:-1]):
        nxt = d + dt.timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += dt.timedelta(days=1)
        if nxt in holidays:
            pre.add(d)
    b3: dict[str, list] = {"pre-holiday": [], "other days": []}
    for a, b in zip(ins, ins[1:]):
        r = 10000.0 * (spy[b]["close"] / spy[a]["close"] - 1.0)
        b3["pre-holiday" if a in pre else "other days"].append(r)
    for k in b3:
        screen.record(k, b3[k], descriptive=True)
    describe_split(b3, "pre-holiday", "other days", screen=screen,
                   label="L3 pre-holiday")
    print()

    # ---------------------------------------------------------------- L4
    print("L4  VIX percentile -> forward 21-session SPY return")
    vix = load_vix()
    H = 21
    have = [d for d in ins if d in vix]
    if len(have) < 500:
        print("    insufficient VIX overlap -- skipped")
    else:
        levels = sorted(vix[d] for d in have)
        lo_c = levels[len(levels) // 3]
        hi_c = levels[2 * len(levels) // 3]
        idx = {d: i for i, d in enumerate(ins)}
        b4: dict[str, list] = {"high VIX": [], "low VIX": []}
        for d in have:
            i = idx[d]
            if i + H >= len(ins):
                continue
            r = 10000.0 * (spy[ins[i + H]]["close"] / spy[d]["close"] - 1.0)
            if vix[d] > hi_c:
                b4["high VIX"].append(r)
            elif vix[d] < lo_c:
                b4["low VIX"].append(r)
        for k in b4:
            screen.record(k, b4[k], descriptive=True)
        w = welch_overlap(b4["high VIX"], b4["low VIX"], H)
        print("    -> high minus low VIX: {0:+.2f}bp  t={1:+.2f}  "
              "95% CI [{2:+.2f}, {3:+.2f}]  (overlap corrected)".format(
                  w["diff"], w["t"], w["lo"], w["hi"]))
        screen.register_difference("L4 VIX percentile", w, "bp")
    print()

    # ---------------------------------------------------------------- L5
    print("L5  sector momentum: top 3 vs bottom 3, monthly non-overlapping")
    month_ends = [by_month[k][-1] for k in months]
    b5: dict[str, list] = {"top 3": [], "bottom 3": []}
    for i in range(12, len(month_ends) - 1):
        now, nxt = month_ends[i], month_ends[i + 1]
        back = month_ends[i - 12]
        scored = []
        for s in SECTORS:
            bars = data.get(s, {})
            if now in bars and back in bars and nxt in bars:
                scored.append((bars[now]["close"] / bars[back]["close"] - 1.0,
                               s))
        if len(scored) < 6:
            continue
        scored.sort(reverse=True)
        for label, group in (("top 3", scored[:3]), ("bottom 3", scored[-3:])):
            rets = [10000.0 * (data[s][nxt]["close"] / data[s][now]["close"]
                               - 1.0) - COST_ETF_BP for _, s in group]
            b5[label].append(sum(rets) / len(rets))
    for k in b5:
        screen.record(k, b5[k], descriptive=True)
    describe_split(b5, "top 3", "bottom 3", screen=screen,
                   label="L5 sector momentum")
    print()

    # ---------------------------------------------------------------- L6
    print("L6  10-month trend following vs buy-and-hold, six ETFs")
    print("    (published claim is drawdown reduction, not higher return --")
    print("     a null on RETURN does not refute it)")
    b6: dict[str, list] = {"trend filtered": [], "buy and hold": []}
    for t in ASSETS:
        bars = data.get(t, {})
        tdays = sorted(bars)
        tme = []
        seen: dict[tuple, dt.date] = {}
        for d in tdays:
            seen[(d.year, d.month)] = d
        tme = [seen[k] for k in sorted(seen)]
        for i in range(10, len(tme) - 1):
            if tme[i] > ins[-1]:
                break
            window = [bars[tme[j]]["close"] for j in range(i - 9, i + 1)]
            ma = sum(window) / len(window)
            r = 10000.0 * (bars[tme[i + 1]]["close"] / bars[tme[i]]["close"]
                           - 1.0)
            b6["buy and hold"].append(r)
            b6["trend filtered"].append(
                r - COST_ETF_BP if bars[tme[i]]["close"] > ma else 0.0)
    for k in b6:
        screen.record(k, b6[k], descriptive=True)
    describe_split(b6, "trend filtered", "buy and hold", screen=screen,
                   label="L6 trend following")
    print()

    screen.summary()
    print("\nSix strategies fixed as one family before the first statistic. "
          "The survivor, if any, was not chosen after the fact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
