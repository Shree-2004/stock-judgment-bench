"""Tests for brief parsing and the L1 grounding check.

The parser tests encode a deliberate stance: forgiving about formatting, strict
about content. The L1 tests encode the two failure modes that a substring-based
grounding check gets wrong — rounding and field-blindness.
"""
from __future__ import annotations

import pytest

from evals.number_check import check_numbers
from llm.brief import parse_brief
from llm.prompts import DEFAULT_VERSION, build_prompt, get, render_figures_block

WELL_FORMED = """\
VERDICT: BULLISH
CONFIDENCE: 3
KEY_FIGURES:
- price: 3020.65
- change_pct_1mo: 2.98
- market_cap: 20438050139042.25
DRIVERS:
- "Viacom18 partnership announced" -> [2]
- "Board meeting called for results and dividend" -> [1]
BULL_CASE: Retail and Jio keep compounding.
BEAR_CASE: Refining margins are cyclical.
"""

TRUTH = {
    "price": 3020.65,
    "previous_close": 2860.80,
    "change_pct_1d": 5.5876,
    "change_pct_1mo": 2.9849,
    "market_cap": 20438050139042.25,
    "week52_high": 3020.65,
    "week52_low": 2226.50,
}


# ---------------------------------------------------------------------------
# parsing: the happy path
# ---------------------------------------------------------------------------

def test_parses_a_well_formed_brief():
    b = parse_brief(WELL_FORMED)

    assert b.parsed
    assert b.verdict == "BULLISH"
    assert b.confidence == 3
    assert b.cited_figures == {
        "price": 3020.65, "change_pct_1mo": 2.98, "market_cap": 20438050139042.25,
    }
    assert [d.citation for d in b.drivers] == [2, 1]
    assert b.drivers[0].claim == "Viacom18 partnership announced"
    assert b.bull_case.startswith("Retail and Jio")
    assert b.parse_errors == []


# ---------------------------------------------------------------------------
# parsing: forgiving about shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", [
    "**VERDICT:** BULLISH",
    "VERDICT : BULLISH",
    "# VERDICT: BULLISH",
    "> VERDICT: BULLISH",
    "verdict: bullish",
])
def test_markdown_and_case_noise_is_absorbed(variant):
    b = parse_brief(f"{variant}\nCONFIDENCE: 2\n")
    assert b.verdict == "BULLISH"


def test_preamble_before_the_contract_is_ignored():
    b = parse_brief("Sure! Here is the brief you asked for.\n\n" + WELL_FORMED)
    assert b.verdict == "BULLISH"
    assert len(b.cited_figures) == 3


@pytest.mark.parametrize("line, label, value", [
    ("- price: 3,020.65", "price", 3020.65),
    ("* **price**: 3020.65", "price", 3020.65),
    ("price = 3020.65", "price", 3020.65),
    ("- change_pct_1d: -1.30", "change_pct_1d", -1.30),
])
def test_figure_lines_survive_formatting_drift(line, label, value):
    b = parse_brief(f"VERDICT: NEUTRAL\nKEY_FIGURES:\n{line}\n")
    assert b.cited_figures[label] == value


@pytest.mark.parametrize("driver_line, citation", [
    ('- "claim here" -> [2]', 2),
    ("- claim here -> 2", 2),
    ("- claim here [2]", 2),
    ("- claim here (see [2])", 2),
])
def test_driver_citations_survive_formatting_drift(driver_line, citation):
    b = parse_brief(f"VERDICT: NEUTRAL\nDRIVERS:\n{driver_line}\n")
    assert b.drivers[0].citation == citation


# ---------------------------------------------------------------------------
# parsing: strict about content
# ---------------------------------------------------------------------------

def test_an_uncited_driver_is_recorded_not_discarded():
    """An uncited claim is exactly what L2 is hunting for — keep it."""
    b = parse_brief('VERDICT: BULLISH\nDRIVERS:\n- "Demand is clearly improving"\n')
    assert len(b.drivers) == 1
    assert b.drivers[0].citation is None
    assert "improving" in b.drivers[0].claim


def test_explicit_none_drivers_is_valid_and_empty():
    b = parse_brief("VERDICT: NEUTRAL\nDRIVERS: none\nBULL_CASE: x\n")
    assert b.drivers == []
    assert b.parse_errors == []


@pytest.mark.parametrize("bad", ["VERDICT: MAYBE", "VERDICT: quite positive"])
def test_a_verdict_outside_the_three_values_is_an_error(bad):
    b = parse_brief(bad)
    assert not b.parsed
    assert any("verdict" in e.lower() for e in b.parse_errors)


@pytest.mark.parametrize("bad", ["CONFIDENCE: 9", "CONFIDENCE: 0"])
def test_confidence_outside_1_to_5_is_an_error(bad):
    b = parse_brief(f"VERDICT: NEUTRAL\n{bad}\n")
    assert b.confidence is None
    assert any("confidence" in e.lower() for e in b.parse_errors)


def test_an_unparseable_figure_line_is_reported_not_guessed():
    b = parse_brief("VERDICT: NEUTRAL\nKEY_FIGURES:\n- price is about three thousand\n")
    assert b.cited_figures == {}
    assert any("unparseable" in e for e in b.parse_errors)


def test_empty_response_is_a_parse_failure_not_a_neutral_brief():
    b = parse_brief("")
    assert not b.parsed
    assert "empty response" in b.parse_errors


def test_prose_response_fails_to_parse_rather_than_scoring_zero():
    """A failed parse must stay distinguishable from a wrong brief."""
    b = parse_brief("Reliance looks strong. Revenue is growing and Jio is doing well.")
    assert not b.parsed


# ---------------------------------------------------------------------------
# L1: the two failure modes substring checks get wrong
# ---------------------------------------------------------------------------

def test_rounding_is_accepted_not_flagged_as_hallucination():
    """A substring check fails this. It is honest rounding, not fabrication."""
    result = check_numbers({"market_cap": 20_440_000_000_000.0}, TRUTH)
    assert result.checks[0].status == "matched"
    assert result.wrong == 0


def test_a_right_number_on_the_wrong_label_is_caught():
    """3020.65 is real — it is the price, not the 52-week low.

    A substring check passes this, because the number exists in the source.
    Field-aware checking is the whole point.
    """
    result = check_numbers({"week52_low": 3020.65}, TRUTH)
    assert result.checks[0].status == "wrong"
    assert result.score == 0.0


def test_a_fabricated_value_is_caught():
    result = check_numbers({"price": 1510.32}, TRUTH)   # the split-adjusted trap
    assert result.checks[0].status == "wrong"


def test_an_invented_label_is_unverifiable_not_wrong():
    """Inventing a field is a different failure from inventing a value."""
    result = check_numbers({"pe_ratio": 28.4}, TRUTH)
    assert result.checks[0].status == "unverifiable"
    assert result.wrong == 0
    assert result.score is None          # nothing checkable was cited
    assert not result.clean


def test_score_is_none_not_zero_when_nothing_checkable_was_cited():
    """Scoring silence as failure would reward verbosity."""
    assert check_numbers({}, TRUTH).score is None


def test_mixed_citations_score_proportionally():
    result = check_numbers(
        {"price": 3020.65,          # right
         "week52_low": 2226.50,     # right
         "change_pct_1d": 99.0,     # wrong
         "dividend_yield": 0.4},    # unverifiable, excluded from the score
        TRUTH,
    )
    assert (result.matched, result.wrong, result.unverifiable) == (2, 1, 1)
    assert result.score == pytest.approx(2 / 3)


def test_tolerance_boundary():
    truth = {"price": 100.0}
    assert check_numbers({"price": 100.9}, truth, tolerance=0.01).matched == 1
    assert check_numbers({"price": 101.5}, truth, tolerance=0.01).wrong == 1


def test_negative_and_zero_truth_values_do_not_blow_up():
    truth = {"change_pct_1d": -1.30, "pct_off_52w_high": 0.0}
    result = check_numbers({"change_pct_1d": -1.30, "pct_off_52w_high": 0.0}, truth)
    assert result.matched == 2


def test_end_to_end_parse_then_check():
    brief = parse_brief(WELL_FORMED)
    result = check_numbers(brief.cited_figures, TRUTH)
    assert result.cited == 3
    assert result.clean
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def test_figures_block_uses_snapshot_labels_verbatim():
    block = render_figures_block({"price": 3020.65, "week52_high": 3020.65})
    assert "- price: 3,020.65" in block
    assert "- week52_high: 3,020.65" in block


def test_every_prompt_version_is_self_identifying():
    for version in ("v1_structured", "v2_prose"):
        assert get(version).version == version


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(KeyError):
        get("v99")


def test_build_prompt_embeds_the_cutoff_and_forbids_later_knowledge():
    class FakeSnap:
        ticker, company_name, currency, as_of = "RELIANCE.NS", "Reliance", "INR", "2024-06-03"

        def figures(self):
            return {"price": 3020.65}

    class FakeDisc:
        def as_prompt_block(self):
            return "[1] 2024-05-30  [Board Meeting] Results"

    prompt, rendered = build_prompt(FakeSnap(), FakeDisc(), DEFAULT_VERSION)

    assert prompt.version == "v1_structured"
    assert "2024-06-03" in rendered
    assert "nothing about events after 2024-06-03" in rendered
    assert "[1] 2024-05-30" in rendered
    assert "- price: 3,020.65" in rendered


# ---------------------------------------------------------------------------
# L1: rounding of small-magnitude figures
#
# Found by the benchmark's own first live run, not by inspection: a model told
# to round to two decimals wrote change_pct_1d: 0.42 against a truth of 0.4248.
# Relative error 1.13% > 1% tolerance, so pure relative tolerance called correct
# rounding a hallucination. Percentage fields sit near zero constantly, so this
# would have biased a large share of every run.
# ---------------------------------------------------------------------------

def test_two_decimal_rounding_of_a_small_percentage_is_not_a_hallucination():
    truth = {"change_pct_1d": 0.4247830822906595}
    result = check_numbers({"change_pct_1d": 0.42}, truth)
    assert result.checks[0].status == "matched"
    assert result.clean


@pytest.mark.parametrize("cited, truth_value", [
    (0.42, 0.4247830822906595),
    (-0.63, -0.6321),
    (5.75, 5.7513),
    (-17.40, -17.3987),
    (0.0, 0.0004),
])
def test_correct_rounding_matches_at_any_magnitude(cited, truth_value):
    result = check_numbers({"change_pct_1d": cited}, {"change_pct_1d": truth_value})
    assert result.checks[0].status == "matched"


@pytest.mark.parametrize("cited, truth_value", [
    (0.52, 0.4247830822906595),   # not a rounding of the truth
    (4.20, 0.4247830822906595),   # decimal-point slip
    (-0.42, 0.4247830822906595),  # sign flipped
])
def test_fabrication_still_fails_despite_the_rounding_allowance(cited, truth_value):
    result = check_numbers({"change_pct_1d": cited}, {"change_pct_1d": truth_value})
    assert result.checks[0].status == "wrong"


def test_the_split_adjusted_price_trap_still_fails():
    """1510.32 is the value yfinance stores; 3020.65 is what was quoted."""
    result = check_numbers({"price": 1510.32}, {"price": 3020.65})
    assert result.checks[0].status == "wrong"


def test_integer_citation_of_a_fractional_truth_rounds_correctly():
    # round(3020.65, 0) == 3021 -> matched (and it is inside 1% anyway).
    assert check_numbers({"price": 3021.0}, {"price": 3020.65}).matched == 1
    # A small-magnitude truth isolates the rounding path from the tolerance
    # path: round(0.4248, 0) == 0, so 1.0 is neither a rounding nor within 1%.
    assert check_numbers({"change_pct_1d": 1.0}, {"change_pct_1d": 0.4248}).wrong == 1
