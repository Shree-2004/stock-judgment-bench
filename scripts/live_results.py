"""Collect quarterly results for the 41 tracked stocks (fundamentals track, step 1).

    python -m scripts.live_results                        # all 41; the first run backfills ~6 quarters
    python -m scripts.live_results --symbols HEROMOTOCO,HDFCBANK
    python -m scripts.live_results --status               # what is stored, no fetch

Also runs after each nightly ledger round (scripts.live_predict). Nothing reads
these rows yet; step 2 builds the block a model would see.

Exit codes: 0 ok, 1 every listing failed (recorded as failed in job_runs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from live import store
from live.filings import NSE_NAMES
from live.results import quarter_facts, run_once, summary

LOG = Path(__file__).resolve().parent.parent / "logs" / "live_results.log"


def status(db: Path) -> int:
    conn = store.connect(db, read_only=True)
    rows = conn.execute("""SELECT symbol, COUNT(*), MAX(quarter_end), MAX(published_at) FROM results
                           GROUP BY symbol""").fetchall()
    have = {r[0]: r for r in rows}
    for sym in NSE_NAMES:
        r = have.get(sym)
        if not r:
            print(f"  {sym:<11} nothing stored")
            continue
        latest = conn.execute("""SELECT facts_json FROM results WHERE symbol = ? AND statement = 'Consolidated'
                                 ORDER BY quarter_end DESC, published_at DESC LIMIT 1""", (sym,)).fetchone()
        facts = quarter_facts(json.loads(latest[0]))[0] if latest else {}
        profit = facts.get("ProfitLossForPeriod") or facts.get("ProfitLossForThePeriod")
        print(f"  {sym:<11} {r[1]:>3} filings   latest quarter {r[2]}   published {r[3][:16]}"
              + (f"   profit ₹{float(profit) / 1e7:,.0f} Cr" if profit else "   (no consolidated profit fact)"))
    return 0


def main(argv=None) -> int:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m scripts.live_results")
    p.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    p.add_argument("--symbols", default=",".join(NSE_NAMES))
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)
    if args.status:
        return status(args.db)

    run = run_once([s.strip().upper() for s in args.symbols.split(",") if s.strip()], args.db)
    line = f"{run['started_at']} job={run['id']} {summary(run)}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    for r in json.loads(run["rejects_json"] or "[]"):
        print(f"  reject {r['company']}: {r['reason']}")
    return 0 if run["state"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
