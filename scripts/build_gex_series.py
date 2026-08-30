"""Build the daily SPX dealer-gamma series from the purchased Databento data.

    python scripts/build_gex_series.py [--limit N] [--out PATH]

For each session: merge the SPX and SPXW books, recover spot from put-call
parity, solve IV from the quotes, and compute net GEX and the zero-gamma
flip at three horizons.

PRE-REGISTERED, before any of this was run: **7DTE is the primary
horizon.** 1DTE and 30DTE are secondaries and count against the multiple-
testing correction. The reason is stated in advance so it cannot be chosen
afterwards: 7DTE gamma is large enough to matter and still alive at the
next session's open, whereas much of 1DTE gamma expires inside the very
return window it is supposed to predict.

WHAT EACH ROW MEANS
-------------------
`session` is the session the OPEN INTEREST describes -- the day BEFORE the
file's date, because OPRA publishes OI at 06:30 ET for the prior close.
The signal on row D is therefore knowable before session D opens, which is
what makes it tradeable rather than hindsight.

`spot` is the parity-implied FORWARD, not the index level. That is the
right reference for comparing against ES, which is itself a forward.

Resumable: rows already in the output are skipped, so an interrupted run
continues rather than restarting.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.levels import (  # noqa: E402
    Convention, gamma_profile, solve_chain_iv, zero_gamma,
)
from quantdesk.data.sources.databento_opra import (  # noqa: E402
    build_chain, implied_spot, load_definitions, load_session, merge_roots,
)

DOWNLOADS = Path(r"C:\Users\jontr\Downloads")
WORK = DOWNLOADS / "databento_spx" / "all"

#: job_id -> (root, schema)
ORDERS = {
    "OPRA-20260830-MKMHBN64CY": ("SPX", "statistics"),
    "OPRA-20260830-7SFYGSCNEN": ("SPX", "definition"),
    "OPRA-20260830-KBS99GANT7": ("SPXW", "statistics"),
    "OPRA-20260830-MNPB6KU4UR": ("SPXW", "definition"),
}

HORIZONS = (1, 7, 30)
PRIMARY = 7

FIELDS = (["session", "file_date", "spot", "contracts", "total_oi"]
          + [f"{k}_{h}" for h in HORIZONS
             for k in ("net_gex", "flip", "used", "call_wall", "put_wall")])


def extract_all() -> None:
    """Unpack every order once. Idempotent."""
    WORK.mkdir(parents=True, exist_ok=True)
    for job, (root, schema) in ORDERS.items():
        target = WORK / f"{root}_{schema}"
        if target.exists() and any(target.iterdir()):
            continue
        target.mkdir(parents=True, exist_ok=True)
        path = DOWNLOADS / f"{job}.zip"
        if not path.exists():
            print(f"  MISSING {path.name}")
            continue
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if n.endswith(".zst")]
            z.extractall(target, members=members)
        print(f"  extracted {len(members):,} -> {target.name}")


def file_dates(root: str, schema: str) -> dict[dt.date, Path]:
    out: dict[dt.date, Path] = {}
    folder = WORK / f"{root}_{schema}"
    for p in folder.glob("*.zst"):
        m = re.search(r"(\d{8})", p.name)
        if m:
            out[dt.datetime.strptime(m.group(1), "%Y%m%d").date()] = p
    return out


def existing(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with out_path.open(encoding="utf-8") as fh:
        return {r["file_date"] for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "spx_gex_daily.csv"))
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("extracting orders ...")
    extract_all()

    sets = {(r, s): file_dates(r, s)
            for r in ("SPX", "SPXW") for s in ("statistics", "definition")}
    for key, files in sets.items():
        print("  {0:<22} {1:,} files".format("/".join(key), len(files)))

    # A session is usable only when all four files exist for it.
    days = sorted(set.intersection(*[set(v) for v in sets.values()]))
    done = existing(out_path)
    todo = [d for d in days if d.isoformat() not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"\n{len(days):,} complete sessions; {len(done):,} already built; "
          f"{len(todo):,} to do\n")

    write_header = not out_path.exists()
    started = time.time()
    ok = fail = 0

    with out_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for i, day in enumerate(todo, 1):
            try:
                stats = merge_roots(
                    load_session(sets[("SPX", "statistics")][day]),
                    load_session(sets[("SPXW", "statistics")][day]))
                defs = merge_roots(
                    load_definitions(sets[("SPX", "definition")][day]),
                    load_definitions(sets[("SPXW", "definition")][day]))

                # OI published 06:30 on `day` describes the prior session.
                session = next(iter(
                    s.oi_session for s in stats.values() if s.oi_session), None)
                if session is None:
                    fail += 1
                    continue

                # Spot from the standard root: its deep strikes have two
                # liquid legs, which is what parity needs.
                spot = implied_spot(
                    {k: v for k, v in stats.items()
                     if defs.get(k) and defs[k].asset == "SPX"},
                    defs, session) or implied_spot(stats, defs, session)
                chain = build_chain(stats, defs, session, spot=spot)
                if chain is None:
                    fail += 1
                    continue

                solved = solve_chain_iv(chain, max_dte=max(HORIZONS))
                row = {
                    "session": session.isoformat(),
                    "file_date": day.isoformat(),
                    "spot": round(chain.spot, 4),
                    "contracts": len(chain),
                    "total_oi": int(chain.total_open_interest()),
                }
                for h in HORIZONS:
                    prof = gamma_profile(solved, max_dte=h)
                    flip = zero_gamma(solved, max_dte=h)
                    above = prof.nearest_wall(above=True)
                    below = prof.nearest_wall(above=False)
                    row[f"net_gex_{h}"] = round(prof.net_gex, 2)
                    row[f"flip_{h}"] = round(flip, 4) if flip else ""
                    row[f"used_{h}"] = prof.contracts_used
                    row[f"call_wall_{h}"] = above.strike if above else ""
                    row[f"put_wall_{h}"] = below.strike if below else ""
                writer.writerow(row)
                fh.flush()
                ok += 1
            except Exception as exc:                 # noqa: BLE001
                fail += 1
                print(f"  {day}: {type(exc).__name__}: {exc}")

            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - started) / i
                left = rate * (len(todo) - i)
                print("  {0:>4}/{1}  ok {2}  fail {3}  {4:.1f}s/session  "
                      "~{5:.0f} min left".format(
                          i, len(todo), ok, fail, rate, left / 60))

    print(f"\ndone: {ok:,} sessions written, {fail:,} failed -> {out_path}")
    print(f"primary horizon is {PRIMARY}DTE, registered before the run.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
