"""Print the benchmark result. Reads frozen artefacts only — no network, no LLM.

    python -m scripts.report
    python -m scripts.report --allow-incomplete    # mid-run diagnostics
    python -m scripts.report --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import settings
from data.universe import UNIVERSE
from data.windows import build_windows
from evals.baselines import BASELINES, build_baseline_rows
from evals.grid import GridIncompleteError, grid_stats, load_grid
from evals.report import build_report
from evals.scoring import score_l3
from llm.prompts import DEFAULT_VERSION


def _s(v, width=10):
    return str(v).rjust(width) if v is not None else "—".rjust(width)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m scripts.report")
    p.add_argument("--prompt-version", default=DEFAULT_VERSION)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    windows = build_windows()
    tickers = [c.ticker for c in UNIVERSE]
    cutoffs = [w.cutoff for w in windows]

    stats = grid_stats(tickers, cutoffs)
    try:
        grid = load_grid(tickers, cutoffs, allow_incomplete=args.allow_incomplete)
    except GridIncompleteError as exc:
        print(f"refusing to report: {exc}", file=sys.stderr)
        return 1

    partial = not stats["complete"]
    if partial:
        print("*** PARTIAL DATA — diagnostics only, NOT a result ***\n")

    baseline_rows = build_baseline_rows(grid)
    baselines = {b.name: score_l3(b.name, baseline_rows[b.name]).summary()
                 for b in BASELINES}

    rep = build_report(args.prompt_version, args.model or settings.llm_model, grid, baselines,
                       disclosure_limit=settings.disclosure_limit, cutoffs=cutoffs)

    print(f"grid: {stats['ok']}/{stats['expected']} cells  "
          f"({len(grid)} dates scoreable)\n")

    l1 = rep["l1_grounding"]
    print("L1 — numeric grounding")
    print(f"  briefs                      {l1['briefs']:,}  "
          f"(failed {l1['failed']}, unparsed {l1['unparsed']})")
    print(f"  mean grounding              {l1['mean_grounding']}")
    print(f"  clean rate                  {l1['clean_rate']}")
    print(f"  briefs w/ a wrong figure    {l1['briefs_with_a_wrong_figure']}")
    print(f"  briefs w/ invented label    {l1['briefs_with_an_invented_label']}")

    l2 = rep["l2_attribution_tier1"]
    print("\nL2 — attribution (tier 1 only; tier 2 heuristic is not a metric)")
    print(f"  briefs checked              {l2['briefs_checked']:,}")
    print(f"  claims                      {l2['claims']:,}")
    print(f"  fabricated citations        {l2['fabricated_citations']}")
    print(f"  uncited claims              {l2['uncited_claims']}")
    print(f"  citation validity           {l2['citation_validity']}")

    print(f"\nverdicts: {rep['verdict_distribution']}")
    j = rep["l3_join"]
    print(f"join: {j['joined']:,} scored, {j['no_grid_row']:,} no grid row, "
          f"{j['unparsed_or_failed']:,} unusable")

    print("\nL3 — outcome, one observation per date")
    print(f"  {'strategy':<22} {'dates':>6} {'rank corr':>10} {'t':>7} "
          f"{'spread%':>9} {'hit':>6}")
    print("  " + "-" * 64)
    rows = [("model:" + args.prompt_version, rep["l3_outcome"])]
    rows += [(name, s) for name, s in baselines.items()]
    for name, s in rows:
        print(f"  {name:<22} {s['dates']:>6} {_s(s['mean_rank_correlation'])} "
              f"{_s(s['rank_correlation_t'], 7)} {_s(s['mean_spread_pct'], 9)} "
              f"{_s(s['spread_hit_rate'], 6)}")

    if partial:
        print("\n*** PARTIAL DATA — not a result ***")

    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"\nsaved -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
