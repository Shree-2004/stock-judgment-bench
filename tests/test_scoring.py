"""Tests for L3 scoring, outcomes and baselines.

The load-bearing test here is `test_always_bullish_cannot_be_scored`. If that
ever starts returning a number, the benchmark has quietly become a measure of
Indian market drift rather than of judgement, and every result after that point
is worthless.
"""
from __future__ import annotations

from datetime import date

import pytest

from evals.baselines import BY_NAME, build_baseline_rows
from evals.scoring import VERDICT_SCORE, L3Result, score_l3, score_one_date, spearman


# ---------------------------------------------------------------------------
# spearman
# ---------------------------------------------------------------------------

def test_perfect_agreement_is_one():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_perfect_disagreement_is_minus_one():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_monotone_but_nonlinear_still_scores_one():
    """Rank correlation, not linear — a convex relationship is still perfect."""
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)


def test_constant_input_is_none_not_zero():
    """No variation means nothing to correlate. Zero would claim 'no relation',
    which is a different and unearned statement."""
    assert spearman([1, 1, 1, 1], [4, 3, 2, 1]) is None
    assert spearman([1, 2, 3, 4], [5, 5, 5, 5]) is None


def test_ties_use_average_ranks():
    """Verdicts produce heavy ties (10 of 16 BULLISH in the live probe), so
    this is the normal path, not an edge case.

    Tied predictors cannot perfectly order distinct outcomes, so the ceiling
    here is below 1.0 by construction: ranks [1.5, 1.5, 3.5, 3.5] against
    [1, 2, 3, 4] gives 2/sqrt(5). A scorer that reported 1.0 would be
    overstating how much a three-valued verdict can possibly resolve.
    """
    assert spearman([1, 1, 2, 2], [10, 20, 30, 40]) == pytest.approx(0.8944271909999159)

    # Average ranks, not first-seen order: reversing the tied pair changes nothing.
    assert spearman([1, 1, 2, 2], [20, 10, 30, 40]) == pytest.approx(0.8944271909999159)


def test_a_three_valued_verdict_has_a_ceiling_below_one():
    """Worth knowing when reading the model's score: BULLISH/NEUTRAL/BEARISH
    across 41 names can never reach 1.0 even if the ordering is flawless."""
    scores = [1.0] * 14 + [0.0] * 14 + [-1.0] * 13
    returns = list(range(41, 0, -1))          # perfectly ordered, best first
    best_possible = spearman(scores, returns)
    assert best_possible is not None
    assert 0.9 < best_possible < 1.0


def test_too_few_points_is_none():
    assert spearman([1, 2], [3, 4]) is None


# ---------------------------------------------------------------------------
# the guard that matters
# ---------------------------------------------------------------------------

def test_always_bullish_cannot_be_scored():
    """Cross-sectional scoring must make market drift unexploitable.

    A strategy that is bullish on everything has a constant score vector: no
    rank correlation, and no short leg for a spread. If this ever yields a
    number, the benchmark is measuring the bull market.
    """
    result = score_one_date(
        "2024-07-19",
        scores=[1.0] * 6,
        returns=[5.0, 3.0, 8.0, 1.0, 4.0, 6.0],       # a strong up month
        verdicts=["BULLISH"] * 6,
    )
    assert result.rank_correlation is None
    assert result.long_short_spread is None
    assert result.n_bearish == 0


def test_always_bullish_baseline_scores_no_dates_at_all():
    grid = {
        "2024-07-19": [{"change_pct_1mo": m, "forward_return": r}
                       for m, r in [(1.0, 5.0), (2.0, 3.0), (-1.0, 8.0), (0.5, 1.0)]],
    }
    rows = build_baseline_rows(grid)
    result = score_l3("always_bullish", rows["always_bullish"])

    assert result.mean_rank_correlation is None
    assert result.mean_spread is None
    assert result.dates_with_spread == 0


# ---------------------------------------------------------------------------
# per-date scoring
# ---------------------------------------------------------------------------

def test_spread_needs_both_legs():
    both = score_one_date("d", [1.0, -1.0], [5.0, 1.0], ["BULLISH", "BEARISH"])
    assert both.long_short_spread == pytest.approx(4.0)

    one_leg = score_one_date("d", [1.0, 0.0], [5.0, 1.0], ["BULLISH", "NEUTRAL"])
    assert one_leg.long_short_spread is None


def test_rank_correlation_survives_a_missing_short_leg():
    """The reason rank correlation is primary: it still works here."""
    result = score_one_date(
        "d", [1.0, 0.0, 1.0, 0.0], [5.0, 1.0, 6.0, 2.0],
        ["BULLISH", "NEUTRAL", "BULLISH", "NEUTRAL"],
    )
    assert result.long_short_spread is None
    assert result.rank_correlation is not None


def test_verdict_counts_are_reported():
    r = score_one_date("d", [1, -1, 0], [1, 2, 3],
                       ["BULLISH", "BEARISH", "NEUTRAL"])
    assert (r.n_bullish, r.n_bearish, r.n_neutral) == (1, 1, 1)


# ---------------------------------------------------------------------------
# aggregation: one observation per date, not per brief
# ---------------------------------------------------------------------------

def _grid(n_dates: int, per_date: int = 5):
    return {
        f"2024-{m + 1:02d}-01": [
            (float(i % 3 - 1), float((i * 7 + m) % 11 - 5),
             ["BEARISH", "NEUTRAL", "BULLISH"][i % 3])
            for i in range(per_date)
        ]
        for m in range(n_dates)
    }


def test_one_observation_per_date():
    """3,690 briefs must not become 3,690 degrees of freedom."""
    result = score_l3("x", _grid(9, per_date=41))
    assert len(result.dates) == 9
    assert result.summary()["dates"] == 9


def test_dates_with_too_few_names_are_dropped():
    by_date = {"2024-01-01": [(1.0, 2.0, "BULLISH"), (0.0, 1.0, "NEUTRAL")]}
    assert score_l3("x", by_date).dates == []


def test_t_statistic_uses_date_count():
    result = score_l3("x", _grid(12, per_date=20))
    t = result.rank_correlation_t
    assert t is None or isinstance(t, float)
    assert len(result.dates) == 12


def test_t_statistic_needs_at_least_three_dates():
    assert score_l3("x", _grid(2, per_date=10)).rank_correlation_t is None


def test_empty_result_summarises_without_crashing():
    summary = L3Result(label="none").summary()
    assert summary["dates"] == 0
    assert summary["mean_rank_correlation"] is None


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def test_momentum_reads_the_snapshot_not_a_fresh_lookup():
    row = {"change_pct_1mo": 7.5, "forward_return": 0.0}
    assert BY_NAME["momentum"].score_fn(row, None) == 7.5


def test_momentum_and_reversal_are_exact_opposites():
    row = {"change_pct_1mo": 7.5, "forward_return": 0.0}
    assert BY_NAME["reversal"].score_fn(row, None) == -BY_NAME["momentum"].score_fn(row, None)


def test_missing_momentum_is_neutral_not_an_error():
    assert BY_NAME["momentum"].score_fn({"forward_return": 0.0}, None) == 0.0


def test_random_baseline_is_reproducible():
    grid = {f"2024-0{m}-01": [{"change_pct_1mo": 1.0, "forward_return": 2.0}] * 6
            for m in range(1, 5)}
    first = build_baseline_rows(grid, seed=7)["random"]
    second = build_baseline_rows(grid, seed=7)["random"]
    assert first == second


def test_baseline_verdicts_follow_score_sign():
    b = BY_NAME["momentum"]
    assert b.verdict(3.0) == "BULLISH"
    assert b.verdict(-3.0) == "BEARISH"
    assert b.verdict(0.0) == "NEUTRAL"


def test_verdict_score_mapping_is_ordered():
    assert VERDICT_SCORE["BEARISH"] < VERDICT_SCORE["NEUTRAL"] < VERDICT_SCORE["BULLISH"]


# ---------------------------------------------------------------------------
# outcomes (network)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_forward_return_is_computed_from_adjusted_closes():
    from evals.outcomes import get_outcome

    out = get_outcome("RELIANCE.NS", date(2024, 7, 19), date(2024, 8, 20))
    assert out.start_price > 0 and out.end_price > 0
    assert -50 < out.forward_return_pct < 50
    assert out.benchmark_return_pct is not None
    assert out.excess_return_pct == pytest.approx(
        out.forward_return_pct - out.benchmark_return_pct
    )


@pytest.mark.network
def test_outcome_rejects_a_window_it_cannot_price():
    from evals.outcomes import OutcomeError, get_outcome

    with pytest.raises(OutcomeError):
        get_outcome("RELIANCE.NS", date(1995, 1, 2), date(1995, 2, 2))


# ---------------------------------------------------------------------------
# grid integrity
#
# Added after the first full baseline run scored 1,826 of 3,690 pairs and
# printed a clean-looking table. yfinance had throttled mid-run; every
# exception was caught and counted, so the job "succeeded". Worse, the
# missingness was systematic — tickers iterate in a fixed order, so throttling
# always dropped the same tail of the universe.
#
# A partial grid must now be an error, not a footnote.
# ---------------------------------------------------------------------------

def test_load_grid_refuses_holes_by_default(tmp_path, monkeypatch):
    from datetime import date as _date

    from evals import grid as G

    monkeypatch.setattr(G, "GRID_ROOT", tmp_path)
    cutoffs = [_date(2024, 1, 2), _date(2024, 2, 1)]
    tickers = ["A.NS", "B.NS"]

    G.fetch_cell  # touch, so the import is meaningful
    G._write_atomic(
        G.cell_path("A.NS", cutoffs[0]),
        G.GridRow("A.NS", "2024-01-02", "2024-02-01", 1.0, 10.0, 2.0, 1.0, "ok").as_dict(),
    )

    with pytest.raises(G.GridIncompleteError):
        G.load_grid(tickers, cutoffs)


def test_load_grid_allows_holes_only_when_asked(tmp_path, monkeypatch):
    from datetime import date as _date

    from evals import grid as G

    monkeypatch.setattr(G, "GRID_ROOT", tmp_path)
    cutoff = _date(2024, 1, 2)
    tickers = [f"T{i}.NS" for i in range(4)]
    for t in tickers[:3]:
        G._write_atomic(
            G.cell_path(t, cutoff),
            G.GridRow(t, "2024-01-02", "2024-02-01", 1.0, 10.0, 2.0, 1.0, "ok").as_dict(),
        )

    by_date = G.load_grid(tickers, [cutoff], allow_incomplete=True)
    assert len(by_date["2024-01-02"]) == 3


def test_failed_cells_are_excluded_from_the_grid(tmp_path, monkeypatch):
    from datetime import date as _date

    from evals import grid as G

    monkeypatch.setattr(G, "GRID_ROOT", tmp_path)
    cutoff = _date(2024, 1, 2)
    tickers = [f"T{i}.NS" for i in range(4)]
    for i, t in enumerate(tickers):
        status = "failed" if i == 0 else "ok"
        G._write_atomic(
            G.cell_path(t, cutoff),
            G.GridRow(t, "2024-01-02", "2024-02-01",
                      None if i == 0 else 1.0, None if i == 0 else 10.0,
                      None if i == 0 else 2.0, None if i == 0 else 1.0,
                      status, "boom" if i == 0 else None).as_dict(),
        )

    stats = G.grid_stats(tickers, [cutoff])
    assert stats["failed"] == 1 and stats["ok"] == 3
    assert not stats["complete"]

    by_date = G.load_grid(tickers, [cutoff], allow_incomplete=True)
    assert len(by_date["2024-01-02"]) == 3


def test_dates_below_the_minimum_name_count_are_dropped(tmp_path, monkeypatch):
    from datetime import date as _date

    from evals import grid as G

    monkeypatch.setattr(G, "GRID_ROOT", tmp_path)
    cutoff = _date(2024, 1, 2)
    tickers = ["A.NS", "B.NS"]
    for t in tickers:
        G._write_atomic(
            G.cell_path(t, cutoff),
            G.GridRow(t, "2024-01-02", "2024-02-01", 1.0, 10.0, 2.0, 1.0, "ok").as_dict(),
        )

    # 2 names is too thin to rank-correlate; the date must not be scored.
    assert G.load_grid(tickers, [cutoff], allow_incomplete=True) == {}
