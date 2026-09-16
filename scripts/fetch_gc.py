"""
fetch_gc.py -- price, then optionally buy, GC 1-minute history.

WHY IT PRICES FIRST AND BUYS SECOND
-----------------------------------
The $20 CBOE DataShop order in this project delivered two days of data
instead of fourteen years, and the tell was that the price looked too
good. A surprisingly cheap quote IS the signal to check scope before
paying. Databento exposes `metadata.get_cost`, which returns the exact
billed figure for a query before it runs, so there is no reason to ever
find out afterwards.

    python scripts/fetch_gc.py              # price only, spends nothing
    python scripts/fetch_gc.py --download   # actually buy and save

CALIBRATION, from the ES pull already on disk: glbx-mdp3 ohlcv-1m for
2010-06-06..2026-08-30 billed 0.47 GB at $70/GB, about $33. Gold lists more
contract months than the index complex but generates far fewer messages, so
the expectation is the same order of magnitude. If get_cost comes back wildly
outside $15-50, something about the query is wrong -- stop and read it rather
than paying and finding out.

WHY THE PARENT SYMBOL AND NOT GC.c.0
------------------------------------
`GC.FUT` returns every outright; `databento_glbx.front_month()` then picks the
active contract by daily volume, which is the roll this project already
tests against. `GC.c.0` would be cheaper -- one symbol rather than dozens --
but it hands the roll to the vendor, and a continuous series with someone
else's splice is a different dataset from the one the ES results were built
on. At these prices, keep the roll.

ASSUMPTION (data availability): GLBX.MDP3 coverage starts 2010-06-06, the
same boundary the ES pull hit. Requesting earlier silently returns less
rather than erroring, so the start date is pinned to the known boundary.

The key is read from DATABENTO_API_KEY. It is never written to disk, never
logged, and never printed -- the script reports its length and prefix so a
wrong or truncated key is diagnosable without exposing the value.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
SYMBOL = "GC.FUT"
STYPE = "parent"
START = "2010-06-06"
END = "2026-08-30"

OUT_DIR = Path((str(Path.home()) + r"\Downloads\databento_spx\gc_full"))

#: Refuse to spend more than this without a human looking at it again.
#: Calibrated from the ES pull (~$33); anything far outside means the query
#: is not what was intended.
COST_CEILING_USD = 60.0


def _client():
    """databento is imported inside the function so this module stays
    importable in a bare interpreter, matching the rule the rest of the
    package follows."""
    import databento as db

    key = os.environ.get("DATABENTO_API_KEY", "")
    if not key:
        raise SystemExit(
            "DATABENTO_API_KEY is not set.\n"
            "Set it in the environment -- do not paste it into a file that\n"
            "gets committed, and do not paste it into chat.")
    # Report shape, never value. A truncated or wrong-vendor key is
    # diagnosable from this without the secret leaving the process.
    print("key: {0} chars, prefix {1!r}".format(len(key), key[:3]))
    if not key.startswith("db-"):
        print("WARNING: Databento keys start with 'db-'. This one does not,")
        print("so it is probably for a different vendor. Sending it anyway")
        print("would put a live credential in the wrong company's auth log.")
        raise SystemExit("refusing to send a key that does not match the vendor")
    return db.Historical(key)


def main() -> int:
    c = _client()
    print("\n{0}  {1}  {2}  {3} .. {4}".format(
        DATASET, SCHEMA, SYMBOL, START, END))

    cost = c.metadata.get_cost(
        dataset=DATASET, symbols=[SYMBOL], schema=SCHEMA,
        start=START, end=END, stype_in=STYPE)
    size = c.metadata.get_billable_size(
        dataset=DATASET, symbols=[SYMBOL], schema=SCHEMA,
        start=START, end=END, stype_in=STYPE)

    print("billable: {0:.3f} GB".format(size / 1e9))
    print("cost:     ${0:.2f}".format(cost))
    print("(ES ohlcv-1m over a comparable span billed 0.47 GB / ~$33)")

    if "--download" not in sys.argv:
        print("\nPriced only -- nothing spent. Re-run with --download to buy.")
        return 0

    if cost > COST_CEILING_USD:
        print("\nREFUSING: ${0:.2f} is above the ${1:.2f} ceiling in this "
              "script.".format(cost, COST_CEILING_USD))
        print("That is not a hard limit on your account, it is a check that")
        print("the query is the one intended. Read it, then raise the ceiling")
        print("deliberately if the number is right.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "glbx-mdp3-gc-{0}-{1}.{2}.dbn.zst".format(
        START.replace("-", ""), END.replace("-", ""), SCHEMA)
    print("\ndownloading to {0} ...".format(dest))
    data = c.timeseries.get_range(
        dataset=DATASET, symbols=[SYMBOL], schema=SCHEMA,
        start=START, end=END, stype_in=STYPE, path=str(dest))
    print("saved {0:,} bytes".format(dest.stat().st_size))
    print("\nNext: point ES_GLOB in scripts/screen_ablation.py at this file,")
    print("or copy it to a screen_gold_5m.py that resamples 1m -> 5m and runs")
    print("the same four hypotheses at the resolution the bot actually trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
