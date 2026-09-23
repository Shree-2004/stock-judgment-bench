"""Fetch and cache the benchmark grid. Resumable. No scoring here.

Separated from scoring on purpose: the first baseline run fetched and scored in
one pass, so when yfinance throttled it silently scored a biased half-grid and
printed a clean-looking table. Fetching is now its own step that must finish
before anything is scored.

    python -m scripts.build_grid
    python -m scripts.build_grid --status
    python -m scripts.build_grid --retry-failed
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from data.universe import UNIVERSE
from data.windows import build_windows
from evals.grid import fetch_cell, grid_stats, load_cell


def _fmt(s: float) -> str:
    return f"{s:.0f}s" if s < 90 else (f"{s/60:.0f}m" if s < 5400 else f"{s/3600:.1f}h")


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m scripts.build_grid")
    p.add_argument("--status", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    windows = build_windows()
    tickers = [c.ticker for c in UNIVERSE]
    cutoffs = [w.cutoff for w in windows]

    if args.status:
        print(json.dumps(grid_stats(tickers, cutoffs), indent=2))
        return 0

    todo = []
    for w in windows:
        for c in UNIVERSE:
            cell = load_cell(c.ticker, w.cutoff)
            if cell and (cell.status != "failed" or not args.retry_failed):
                continue
            todo.append((c.ticker, w.cutoff, w.forward_end))

    print(f"{len(tickers)} tickers x {len(windows)} windows = {len(tickers)*len(windows):,} cells")
    print(f"{len(tickers)*len(windows) - len(todo):,} cached, {len(todo):,} to fetch", flush=True)
    if not todo:
        print(json.dumps(grid_stats(tickers, cutoffs), indent=2))
        return 0
    if args.limit:
        todo = todo[: args.limit]
        print(f"--limit: fetching {len(todo):,}", flush=True)

    started = time.monotonic()
    ok = failed = 0
    for i, (ticker, cutoff, forward_end) in enumerate(todo, 1):
        row = fetch_cell(ticker, cutoff, forward_end, refetch_failed=args.retry_failed)
        if row.status == "ok":
            ok += 1
        else:
            failed += 1
        if i % 50 == 0 or i == len(todo):
            el = time.monotonic() - started
            print(f"  [{i:>5}/{len(todo)}] {ticker:<14} {cutoff}  ok={ok} failed={failed}  "
                  f"elapsed={_fmt(el)} eta={_fmt((el/i)*(len(todo)-i))}", flush=True)

    print()
    print(json.dumps(grid_stats(tickers, cutoffs), indent=2))
    if failed:
        print(f"\n{failed} cells failed. Rerun with --retry-failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
