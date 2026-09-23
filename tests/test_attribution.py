"""Tests for L2 attribution.

The split between tiers is the point. Tier 1 is exact and is the reported
result; tier 2 is a heuristic that measurement showed does not work, kept only
so it can be recalibrated later against hand labels.
"""
from __future__ import annotations

import pytest

from evals.attribution import (
    SUPPORT_THRESHOLD,
    check_attribution,
    content_words,
    overlap_ratio,
)
from llm.brief import Driver


class Filing:
    def __init__(self, headline):
        self.headline = headline


FILINGS = [
    Filing("Board Meeting Intimation for Considering Audited Financial Results"),
    Filing("Acquisition of Udhaiyam Dhall Brand by the Company"),
    Filing("Allotment of 23,728 equity shares under ESOP"),
]


# ---------------------------------------------------------------------------
# TIER 1 — exact, and the only thing reported as a score
# ---------------------------------------------------------------------------

def test_citing_a_filing_that_was_never_supplied_is_caught_exactly():
    """Four filings shown, model cited [7]. No interpretation makes that right."""
    result = check_attribution([Driver("something happened", 7)], FILINGS)

    assert result.checks[0].status == "fabricated_citation"
    assert result.checks[0].tier == 1
    assert result.fabricated == 1
    assert not result.tier1_clean
    assert result.l2_score == 0.0


@pytest.mark.parametrize("bad", [0, -1, 4, 99])
def test_out_of_range_citations_are_all_fabricated(bad):
    result = check_attribution([Driver("claim", bad)], FILINGS)
    assert result.checks[0].status == "fabricated_citation"


def test_an_uncited_claim_is_a_tier_1_failure():
    result = check_attribution([Driver("Demand is clearly improving", None)], FILINGS)
    assert result.checks[0].status == "uncited"
    assert result.checks[0].tier == 1
    assert result.uncited == 1
    assert not result.tier1_clean


def test_valid_citations_are_tier_1_clean():
    result = check_attribution(
        [Driver("Acquisition of Udhaiyam brand", 2), Driver("ESOP allotment", 3)],
        FILINGS,
    )
    assert result.tier1_clean
    assert result.l2_score == 1.0
    assert all(c.tier == 2 for c in result.checks)


def test_citations_are_one_based_as_shown_to_the_model():
    """The prompt numbers filings [1]..[n]; off-by-one here would invent errors."""
    result = check_attribution([Driver("board meeting", 1)], FILINGS)
    assert result.checks[0].status != "fabricated_citation"
    assert "Board Meeting" in result.checks[0].cited_headline


def test_no_claims_scores_none_not_zero():
    result = check_attribution([], FILINGS)
    assert result.claims == 0
    assert result.l2_score is None


def test_mixed_tier1_outcomes_score_proportionally():
    result = check_attribution(
        [Driver("good", 1), Driver("also good", 2),
         Driver("made up", 9), Driver("no citation", None)],
        FILINGS,
    )
    assert result.fabricated == 1 and result.uncited == 1
    assert result.l2_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TIER 2 — the heuristic, and its documented failure
# ---------------------------------------------------------------------------

def test_stopwords_and_filing_boilerplate_are_stripped():
    words = content_words("Disclosure under Regulation 30 of SEBI LODR for the Company")
    assert "disclosure" not in words and "regulation" not in words
    assert "company" not in words and "sebi" not in words


def test_overlap_is_asymmetric_so_short_claims_are_not_punished():
    """Jaccard would penalise a concise claim against a long filing."""
    claim = "Udhaiyam acquisition"
    long_filing = "Acquisition of Udhaiyam Dhall Brand " + "additional wording " * 20
    assert overlap_ratio(claim, long_filing) == 1.0


def test_overlap_of_an_empty_claim_is_zero_not_an_error():
    assert overlap_ratio("", "anything at all") == 0.0
    assert overlap_ratio("the and of", "anything") == 0.0


def test_high_overlap_is_marked_supported():
    result = check_attribution([Driver("Acquisition of Udhaiyam Dhall Brand", 2)], FILINGS)
    assert result.checks[0].status == "supported"
    assert result.checks[0].overlap >= SUPPORT_THRESHOLD


def test_a_correct_paraphrase_is_wrongly_flagged_weak():
    """The measured failure, pinned so it cannot be forgotten.

    'edtech' is the right word for an Embibe acquisition and appears nowhere in
    the filing. Token overlap scores this 0.0 and calls it weak. It is correct.
    This is why tier 2 is not reported as a metric and needs an entailment
    judge instead.
    """
    filings = [Filing("Indiavidual Learning Private Limited ('Embibe'), a subsidiary")]
    result = check_attribution(
        [Driver("Acquisition of edtech and tech startups expands digital portfolio", 1)],
        filings,
    )
    assert result.checks[0].status == "weak_match"     # the heuristic's verdict
    assert result.checks[0].tier == 2                  # never reported as a score
    assert result.tier1_clean                          # the honest result: fine
    assert result.l2_score == 1.0


def test_tier2_is_excluded_from_the_reported_score():
    filings = [Filing("Completely unrelated wording here")]
    result = check_attribution([Driver("some paraphrased claim", 1)], filings)
    assert result.weak == 1
    assert result.l2_score == 1.0          # tier 1 only
    assert "tier2_heuristic_NOT_A_METRIC" in result.as_dict()
