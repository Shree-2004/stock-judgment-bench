"""Score every baseline over the frozen grid. No network, no LLM calls.

Reads the cached grid only. If the grid has holes this refuses to run, because
the first version of this script fetched and scored in one pass and, when
yfinance throttled, quietly scored half the data and printed a table that
looked publishable.

Run this BEFORE generating briefs: it fixes the metric and shows the bar while
it is still impossible to tune the scoring toward a flattering model result.

    python -m scripts.build_grid          # first
    python -m scripts.run_baselines
    python -m scripts.run_baselines --allow-incomplete   # diagnostics only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.universe import UNIVERSE
from data.windows import build_windows
from evals.baselines import BASELINES, build_baseline_rows
from evals.grid import GridIncompleteError, grid_stats, load_grid
from evals.scoring import score_l3

OUT = Path(__file__).resolve().parent.parent / "cache" / "baselines.json"


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m scripts.run_baselines")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="score a partial grid — diagnostics only, never a result")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)

    windows = build_windows()
    tickers = [c.ticker for c in UNIVERSE]
    cutoffs = [w.cutoff for w in windows]

    stats = grid_stats(tickers, cutoffs)
    print(json.dumps(stats, indent=2), flush=True)

    try:
        by_date = load_grid(tickers, cutoffs, allow_incomplete=args.allow_incomplete)
    except GridIncompleteError as exc:
        print(f"\nrefusing to score: {exc}", file=sys.stderr)
        return 1

    if args.allow_incomplete and not stats["complete"]:
        print("\n*** PARTIAL GRID — diagnostics only, not a result ***", flush=True)

    pairs = sum(len(v) for v in by_date.values())
    print(f"\nscoring {pairs:,} (ticker,date) pairs over {len(by_date)} dates\n", flush=True)

    baseline_rows = build_baseline_rows(by_date)
    results = {}
    print(f"{'baseline':<16} {'dates':>6} {'rank corr':>10} {'t':>7} "
          f"{'spread%':>9} {'w/spread':>9} {'hit':>6}")
    print("-" * 70)
    for b in BASELINES:
        r = score_l3(b.name, baseline_rows[b.name])
        s = r.summary()
        results[b.name] = {"summary": s, "describe": b.describe,
                           "dates": [d.as_dict() for d in r.dates]}
        print(f"{b.name:<16} {s['dates']:>6} {str(s['mean_rank_correlation']):>10} "
              f"{str(s['rank_correlation_t']):>7} {str(s['mean_spread_pct']):>9} "
              f"{s['dates_with_spread']:>9} {str(s['spread_hit_rate']):>6}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "grid_stats": stats,
        "partial": bool(args.allow_incomplete and not stats["complete"]),
        "results": results,
    }, indent=1), encoding="utf-8")
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
