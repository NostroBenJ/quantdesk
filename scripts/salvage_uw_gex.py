"""Rescue the GEX series out of the old NYAM dashboard snapshots.

WHY THIS IS A SCRIPT AND NOT A SOURCE
-------------------------------------
These snapshots came from Unusual Whales, that subscription has lapsed, and
the eight sessions in `dev/NYAM Terminal/engine/data_store/capture` cannot be
re-fetched at any price. So they get copied out into a format that outlives
the dashboard that wrote them, once, and then this script is done.

WHAT THEY ARE, AND WHAT THEY ARE NOT
------------------------------------
They carry Unusual Whales' ANSWER - net_gex, gamma_flip, walls, a 194-strike
GEX profile - and not the inputs. There is no per-strike open interest in
them, so our own gamma cannot be recomputed and `levels.compare()` has
nothing to bite on. Treat every number here as a vendor's opinion computed
under a sign convention they do not publish, which is the same warning that
sits at the top of `levels.py`.

Their real use is as a reference series: if our flip level and theirs ever
sit on the same session, the gap between them is measurable. Right now they
do not overlap - UW covers 2026-08-07..08-14, our own chains start 08-20 -
so no comparison is possible yet. Recorded anyway, because the overlap is
only impossible until it isn't.

Run: python scripts/salvage_uw_gex.py [capture_dir] [out_csv]
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

DEFAULT_CAPTURE = Path(
    r"C:\Users\jontr\dev\NYAM Terminal\engine\data_store\capture")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "uw_gex_archive.csv"

FIELDS = [
    "session_date", "trading_session", "weekday", "is_rerun",
    "generated_at", "provider", "spot", "net_gex", "regime",
    "gamma_flip", "call_wall", "put_wall", "control_node", "atm_iv",
    "put_call_ratio", "call_oi", "put_oi", "call_oi_change", "put_oi_change",
    "profile_strikes",
]


def trading_session_for(captured: date) -> date:
    """The session a capture DESCRIBES, which is not always the day it ran.

    A Saturday or Sunday capture is not a weekend market. It is Friday's book,
    re-read - the spot is Friday's frozen last print, while the GEX moves,
    because OPRA publishes settled open interest after the session closes.

    That makes the weekend rows re-computations of Friday against revised
    inputs, not new observations. Three files, one session. Counting them as
    three is the overlap mistake in its most literal form, and it is invisible
    unless something maps captures back onto sessions - which is this.
    """
    if captured.weekday() == 5:              # Saturday -> Friday
        return captured - timedelta(days=1)
    if captured.weekday() == 6:              # Sunday -> Friday
        return captured - timedelta(days=2)
    return captured


def rows_from(capture_dir: Path):
    seen_sessions: set[date] = set()
    for day in sorted(p for p in capture_dir.iterdir() if p.is_dir()):
        snap = day / "SPY_snapshot.json"
        if not snap.exists():
            print(f"  {day.name}: no SPY_snapshot.json, skipped")
            continue
        try:
            doc = json.loads(snap.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  {day.name}: unreadable ({type(exc).__name__}), skipped")
            continue

        gex = doc.get("gex")
        if not isinstance(gex, dict):
            print(f"  {day.name}: no gex block, skipped")
            continue

        profile = gex.get("profile")
        captured = date.fromisoformat(day.name)
        session = trading_session_for(captured)
        row = {
            "session_date": day.name,
            "trading_session": session.isoformat(),
            "weekday": captured.strftime("%a"),
            # True when an earlier file already described this session.
            "is_rerun": session in seen_sessions,
            "generated_at": doc.get("generated_at", ""),
            # `mock: true` snapshots are demo data and must never enter a
            # series that anything later treats as observed.
            "provider": ("MOCK" if doc.get("mock")
                         else doc.get("provider", "unknown")),
            "profile_strikes": len(profile) if isinstance(profile, list) else 0,
        }
        seen_sessions.add(session)
        for key in FIELDS:
            if key in row:
                continue
            row[key] = gex.get(key, "")
        yield row, profile


def main(argv: list[str]) -> int:
    capture = Path(argv[0]) if argv else DEFAULT_CAPTURE
    out = Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    if not capture.is_dir():
        print(f"capture dir not found: {capture}")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    profiles_dir = out.parent / "uw_gex_profiles"
    profiles_dir.mkdir(exist_ok=True)

    print(f"reading {capture}")
    rows = []
    for row, profile in rows_from(capture):
        rows.append(row)
        if isinstance(profile, list) and profile:
            # The per-strike curve is the part with the most information in
            # it, so it gets its own file rather than being flattened away.
            path = profiles_dir / f"SPY_{row['session_date']}.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["strike", "gex"])
                for point in profile:
                    w.writerow([point.get("strike"), point.get("gex")])
        print("  {0} {1}  spot {2:<8} flip {3:<8} regime {4:<9} {5}".format(
            row["session_date"], row["weekday"], row["spot"],
            row["gamma_flip"], row["regime"],
            "RERUN of " + row["trading_session"] if row["is_rerun"] else ""))

    if not rows:
        print("nothing to salvage")
        return 1

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    mocks = [r["session_date"] for r in rows if r["provider"] == "MOCK"]
    sessions = {r["trading_session"] for r in rows}
    reruns = [r["session_date"] for r in rows if r["is_rerun"]]

    print(f"\nwrote {len(rows)} captures covering {len(sessions)} "
          f"TRADING SESSIONS -> {out}")
    print(f"per-strike profiles -> {profiles_dir}")
    if reruns:
        print(f"  {len(reruns)} re-describe a session already covered: {reruns}")
        print("  dedupe on `trading_session` before treating these as "
              "independent observations.")
    if mocks:
        print(f"WARNING: {len(mocks)} are MOCK data, not observations: {mocks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
