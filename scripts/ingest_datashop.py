"""Load a purchased Cboe DataShop order into the chain store.

    python scripts/ingest_datashop.py <dir-or-files>... [--symbol SPY]
                                      [--snapshot 1545|eod] [--db PATH]
                                      [--dry-run]

Handles a directory of daily or monthly files, .csv or .csv.gz, in any mix.
Idempotent -- `write_chain` upserts on (underlying, fetched_at, symbol), so
re-running after an interrupted load resumes rather than duplicating.

The report at the end is the point. A 14-year order is ~3,500 files and
nobody reads 3,500 lines of output, so what matters is: how many sessions
landed, what the calendar gaps are, and whether any file was skipped. A
silent partial load is the failure mode worth engineering against.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantdesk.data.chain_store import ChainStore  # noqa: E402
from quantdesk.data.sources.cboe_datashop import has_calcs, ingest  # noqa: E402


def collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(
                q for q in p.rglob("*")
                if q.suffix in (".csv", ".gz") and q.is_file()))
        elif p.is_file():
            files.append(p)
        else:
            print(f"  not found, ignored: {item}")
    return files


def weekday_gaps(sessions: list[str]) -> list[str]:
    """Weekdays with no session between the first and last recorded.

    Holidays land in here too, so this is a list to eyeball rather than an
    error. A handful is normal; a run of twenty means part of the order did
    not load.
    """
    if len(sessions) < 2:
        return []
    have = {date.fromisoformat(s) for s in sessions}
    start, end = min(have), max(have)
    out, cur = [], start
    while cur <= end:
        if cur.weekday() < 5 and cur not in have:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="directory or files")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--snapshot", default="1545", choices=("1545", "eod"))
    ap.add_argument("--db", default=str(ROOT / "data" / "chains.sqlite"))
    ap.add_argument("--dry-run", action="store_true",
                    help="read and report, write nothing")
    args = ap.parse_args()

    files = collect(args.inputs)
    if not files:
        print("no .csv or .csv.gz files found")
        return 1

    print(f"{len(files):,} files")
    if not has_calcs(files[0]):
        print("  NOTE: no Calcs columns in the first file. Vendor greeks and")
        print("        IV will be absent, so levels.compare() has nothing to")
        print("        check against and IV must be solved from the quotes.")

    if args.dry_run:
        from quantdesk.data.sources.cboe_datashop import load_chains
        chains = load_chains(files[0], snapshot=args.snapshot,
                             underlying=args.symbol)
        print(f"  dry run on {files[0].name}: {len(chains)} session(s)")
        for c in chains[:3]:
            print(f"    {c.as_of_date}  spot {c.spot:.2f}  "
                  f"{len(c):,} contracts  OI {c.total_open_interest():,.0f}")
        return 0

    done = [0]

    def progress(report):
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"  {done[0]:,} sessions ... latest {report['session_date']}")

    with ChainStore(args.db) as store:
        summary = ingest(files, store, snapshot=args.snapshot,
                         underlying=args.symbol, progress=progress)
        sessions = store.sessions(args.symbol)
        coverage = store.coverage(args.symbol)

    print(f"\nread {summary['files']:,}/{len(files):,} files")
    print(f"sessions in store : {len(sessions):,}")
    print(f"contracts written : {summary['contracts']:,}")
    if coverage:
        print(f"range             : {coverage['first_session']} .. "
              f"{coverage['last_session']}")
        print(f"distinct sources  : {coverage['distinct_sources']}")
        if coverage["distinct_sources"] > 1:
            print("  Two sources now span this series. Any study crossing the")
            print("  boundary is comparing different instruments -- see the")
            print("  note at the top of chain_store.py.")

    gaps = weekday_gaps(sessions)
    print(f"weekday gaps      : {len(gaps)}")
    if gaps:
        print(f"  first few: {gaps[:8]}")
        print("  Holidays belong here. A long run does not -- check the order.")

    if summary["skipped"]:
        print(f"\nSKIPPED {len(summary['skipped'])} files:")
        for line in summary["skipped"][:10]:
            print(f"  {line}")
        return 2

    print("\nNext: verify the open-interest alignment before trusting any GEX.")
    print("OPRA publishes OI in the morning for the PRIOR close, so a row's")
    print("open_interest may describe the session before its quote_date. That")
    print("would shift every result by a day. See verify_oi_lag().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
