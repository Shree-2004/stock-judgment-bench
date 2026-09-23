"""Fill the BSE disclosure cache for the whole benchmark. Resumable.

~3,700 windows at ~0.35s each is roughly 25 minutes. Kill and rerun freely —
completed cells are skipped, so an interruption costs only the in-flight call.

    python -m scripts.fetch_disclosures
    python -m scripts.fetch_disclosures --limit 40      # smoke test
    python -m scripts.fetch_disclosures --refetch-failed
    python -m scripts.fetch_disclosures --status
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from data.disclosures import cache_stats, fetch_and_cache, load_cached
from data.universe import UNIVERSE
from data.windows import build_windows, summarise


def _fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m scripts.fetch_disclosures")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--refetch-failed", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)

    if args.status:
        print(cache_stats())
        return 0

    windows = build_windows()
    print(summarise(windows), flush=True)
    cells = [(c, w.cutoff) for c in UNIVERSE for w in windows]
    print(f"{len(UNIVERSE)} tickers x {len(windows)} windows = {len(cells):,} cells", flush=True)

    todo = []
    for company, cutoff in cells:
        cached = load_cached(company.symbol, cutoff)
        if cached and (cached.status != "failed" or not args.refetch_failed):
            continue
        todo.append((company, cutoff))

    print(f"{len(cells) - len(todo):,} already cached, {len(todo):,} to fetch", flush=True)
    if not todo:
        print("nothing to do.")
        print(cache_stats())
        return 0

    if args.limit:
        todo = todo[: args.limit]
        print(f"--limit: fetching {len(todo):,} of them", flush=True)

    started = time.monotonic()
    counts = {"ok": 0, "empty": 0, "failed": 0}

    for i, (company, cutoff) in enumerate(todo, 1):
        entry = fetch_and_cache(
            symbol=company.symbol,
            scrip_code=company.scrip_code,
            cutoff=cutoff,
            refetch_failed=args.refetch_failed,
        )
        counts[entry.status] = counts.get(entry.status, 0) + 1

        if i % 50 == 0 or i == len(todo):
            elapsed = time.monotonic() - started
            eta = (elapsed / i) * (len(todo) - i)
            print(
                f"  [{i:>5}/{len(todo)}] {company.symbol:<11} {cutoff}  "
                f"ok={counts['ok']} empty={counts['empty']} failed={counts['failed']}  "
                f"elapsed={_fmt(elapsed)} eta={_fmt(eta)}",
                flush=True,
            )

    print(f"\ndone at {datetime.now():%H:%M:%S}", flush=True)
    print(cache_stats(), flush=True)
    if counts["failed"]:
        print(f"\n{counts['failed']} cells failed. Rerun with --refetch-failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
