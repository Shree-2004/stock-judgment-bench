"""The v1 result must not move.

`results-final.json` is the published finding (rank corr −0.0153, t −0.65 over
90 dates). These tests fail if anything — a refactor, a scoring change, a stray
cache file, the calendar advancing — would make `scripts.report` print a
different number.

The cutoffs are committed in `golden/v1_cutoffs.json` so the recompute needs no
network. The recompute itself reads the local cache (gitignored, refetchable), so it skips on a fresh clone rather than failing.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from data.universe import UNIVERSE
from data.windows import BENCHMARK_END, build_windows

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results-final.json"
BASELINES = ROOT / "cache" / "baselines.json"
CUTOFFS = Path(__file__).parent / "golden" / "v1_cutoffs.json"
BRIEFS = ROOT / "cache" / "briefs"

FROZEN = json.loads(RESULTS.read_text(encoding="utf-8"))
V1_CUTOFFS = [date.fromisoformat(c) for c in json.loads(CUTOFFS.read_text(encoding="utf-8"))]


def test_golden_cutoffs_are_the_90_non_overlapping_v1_dates():
    assert len(V1_CUTOFFS) == FROZEN["l3_outcome"]["dates"] == 90
    assert V1_CUTOFFS == sorted(set(V1_CUTOFFS))
    assert V1_CUTOFFS[0] == date(2019, 1, 2)
    assert V1_CUTOFFS[-1] < BENCHMARK_END


def test_headline_numbers_are_the_published_ones():
    # A hand-written copy of the README's claims. If results-final.json is ever
    # regenerated with different numbers, this is the test that says so.
    l3 = FROZEN["l3_outcome"]
    assert (l3["dates"], l3["mean_rank_correlation"], l3["rank_correlation_t"]) == (90, -0.0153, -0.65)
    assert FROZEN["l1_grounding"]["briefs"] == 3690
    assert FROZEN["l1_grounding"]["briefs_with_a_wrong_figure"] == 4
    assert FROZEN["l2_attribution_tier1"]["fabricated_citations"] == 0


def test_committed_baselines_match_the_result():
    stored = json.loads(BASELINES.read_text(encoding="utf-8"))
    assert {k: v["summary"] for k, v in stored["results"].items()} == FROZEN["baselines"]


@pytest.mark.network
def test_build_windows_still_yields_the_golden_cutoffs():
    # The bug this pins: `end` used to default to today, so the window list grew
    # by one every 21 trading days and the benchmark stopped being reproducible.
    assert [w.cutoff for w in build_windows()] == V1_CUTOFFS


@pytest.mark.skipif(not BRIEFS.exists(), reason="local brief/grid cache not present")
def test_report_recomputed_from_cache_is_identical():
    from evals.baselines import BASELINES as STRATEGIES, build_baseline_rows
    from evals.grid import load_grid
    from evals.report import build_report
    from evals.scoring import score_l3

    grid = load_grid([c.ticker for c in UNIVERSE], V1_CUTOFFS)
    rows = build_baseline_rows(grid)
    baselines = {b.name: score_l3(b.name, rows[b.name]).summary() for b in STRATEGIES}
    rep = build_report(FROZEN["prompt_version"], FROZEN["model"], grid, baselines,
                       disclosure_limit=5, cutoffs=V1_CUTOFFS)

    # Round-trip through JSON so tuples/ints compare exactly as the file does.
    assert json.loads(json.dumps(rep)) == FROZEN
