"""M4: settle due calls, then print the pre-registered analysis.

    python -m scripts.settle              # settle what's due, then analyse
    python -m scripts.settle --no-fetch   # analyse what's already settled

Runs nightly from scripts.live_predict too. Settlement only ever appends to
`outcomes`; see live/settle.py for the rules.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from live import store
from live.filings import IST
from live.ledger import EVIDENCE_START, ensure_schema
from live.settle import analyse, settle_pending


def calendar() -> list[date]:
    """NSE sessions from the start of the store up to the latest posted close."""
    from data.windows import nse_trading_days
    return list(nse_trading_days(EVIDENCE_START, datetime.now(IST).date()))


def run(conn, cal=None) -> str:
    """Settle what's due; one-line summary for the nightly log."""
    out = settle_pending(conn, cal if cal is not None else calendar())
    return (f"{out['settled']} settled, {out['waiting']} waiting"
            + (f", {len(out['failed'])} failed ({out['failed'][0]})" if out["failed"] else ""))


def main(argv: list[str] | None = None) -> int:
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=store.DEFAULT_DB)
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args(argv)

    conn = store.connect(args.db)
    ensure_schema(conn)
    cal = calendar()
    if not args.no_fetch:
        print("settlement:", run(conn, cal))
    report = analyse(conn, cal)
    if not report or not any(r["secondary"]["dates"] for r in report.values()):
        print("nothing to analyse yet: no eligible call has a settled outcome.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
