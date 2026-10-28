"""Poll NSE announcements once and store every new filing.

    python -m scripts.live_ingest
    python -m scripts.live_ingest --symbols RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK
    python -m scripts.live_ingest --unmatched     # check the name mapping, no fetch

Exchange-wide items are all stored (the feed cannot be backfilled); --symbols
only chooses which tracked names the report prints.

Exit codes: 0 ok, 1 the poll failed (and is recorded as failed in job_runs).
Scheduled every 15 minutes by Task Scheduler via pythonw, so it also appends a
one-line summary to logs/live_ingest.log — pythonw has no console.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from live import store
from live.filings import DEFAULT_SYMBOLS, NSE_NAMES, _norm, tracked_symbol
from live.ingest import run_once

LOG = Path(__file__).resolve().parent.parent / "logs" / "live_ingest.log"


def _log_line(run: dict) -> str:
    gap = run["coverage_gap_s"]
    return (f"{run['started_at']} job={run['id']} state={run['state']} "
            f"fetched={run['fetched']} inserted={run['inserted']} dup={run['duplicates']} "
            f"rejected={run['rejected']} attempts={run['attempts']} "
            f"gap={'-' if gap is None else f'{gap}s'}"
            + (f" error={run['error']!r}" if run["error"] else ""))


def _unmatched(db: Path) -> int:
    """Names that look like a tracked company but did not map, plus tracked
    symbols with nothing stored yet. Either means NSE_NAMES needs a look."""
    conn = store.connect(db, read_only=True)
    companies = [r[0] for r in conn.execute("SELECT DISTINCT company FROM filings")]
    seen = {tracked_symbol(c) for c in companies} - {None}
    for sym, name in NSE_NAMES.items():
        core = _norm(name)
        near = [c for c in companies if core in _norm(c) and tracked_symbol(c) != sym]
        status = "ok" if sym in seen else "no filings yet"
        print(f"  {sym:<10} {status:<15} {('near-misses: ' + repr(near)) if near else ''}")
    return 0


def main(argv=None) -> int:
    if sys.stdout is not None:      # None under pythonw; print() is then a no-op
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m scripts.live_ingest")
    p.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--unmatched", action="store_true")
    args = p.parse_args(argv)

    if args.unmatched:
        return _unmatched(args.db)

    run = run_once(args.db)
    line = _log_line(run)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(f"job {run['id']}  {run['state'].upper()}  attempts={run['attempts']}")
    if run["state"] != "ok":
        print(f"  error: {run['error']}")
        return 1
    print(f"  fetched {run['fetched']:>5}   inserted {run['inserted']:>5}   "
          f"duplicate {run['duplicates']:>5}   rejected {run['rejected']:>3}")
    print(f"  feed covers {run['feed_oldest']} .. {run['feed_newest']} (UTC)")
    gap = run["coverage_gap_s"]
    if gap:
        print(f"  ** COVERAGE GAP: feed turned over; up to {gap / 60:.0f} min of filings may be missing **")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    conn = store.connect(args.db, read_only=True)
    per_symbol: dict[str, tuple[int, str]] = {}
    for r in conn.execute("SELECT company, COUNT(*), MAX(published_at) FROM filings GROUP BY company"):
        sym = tracked_symbol(r[0])
        if sym:
            n, latest = per_symbol.get(sym, (0, ""))
            per_symbol[sym] = (n + r[1], max(latest, r[2]))

    print(f"\n  tracked (as of {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)")
    for sym in symbols:
        if sym not in NSE_NAMES:
            print(f"  {sym:<10} not in NSE_NAMES")
            continue
        n, latest = per_symbol.get(sym, (0, "-"))
        print(f"  {sym:<10} {n:>4} stored   latest {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
