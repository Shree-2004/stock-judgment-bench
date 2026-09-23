"""Tests for the fixed disclosure-selection rule.

This rule decides what the model is allowed to read, so it is part of the
experiment and has to be pinned. `test_cutoff_*` are leak tests — the
disclosure-side equivalent of the guards in test_market_data.py.
"""
from __future__ import annotations

from datetime import date

import pytest

from data.news_data import (
    DisclosureBundle,
    Item,
    clean_text,
    is_noise,
    normalise_headline,
    select_disclosures,
)

CUTOFF = date(2024, 9, 2)


def row(headline: str, filed_at: str, category: str = "Company Update",
        subcategory: str = "General") -> dict:
    return {"headline": headline, "filed_at": filed_at,
            "category": category, "subcategory": subcategory}


# ---------------------------------------------------------------------------
# LEAK TESTS: the cutoff boundary
# ---------------------------------------------------------------------------

def test_cutoff_excludes_filings_on_the_cutoff_day():
    """A brief written as of the cutoff morning cannot see that day's filings."""
    rows = [
        row("Filed the day before", "2024-08-30T14:00:00"),
        row("Filed on the cutoff morning", "2024-09-02T09:30:00"),
        row("Filed on the cutoff afternoon", "2024-09-02T15:45:00"),
    ]
    items, dropped = select_disclosures(rows, CUTOFF, limit=10)

    assert [i.headline for i in items] == ["Filed the day before"]
    assert dropped["after_cutoff"] == 2


def test_cutoff_boundary_is_midnight_exactly():
    rows = [
        row("One second before midnight", "2024-09-01T23:59:59"),
        row("Midnight itself", "2024-09-02T00:00:00"),
    ]
    items, dropped = select_disclosures(rows, CUTOFF, limit=10)

    assert [i.headline for i in items] == ["One second before midnight"]
    assert dropped["after_cutoff"] == 1


def test_no_selected_filing_ever_postdates_the_cutoff():
    from datetime import datetime

    rows = [row(f"filing {d}", f"2024-08-{d:02d}T10:00:00") for d in range(20, 32)]
    rows += [row("later", "2024-09-05T10:00:00")]
    items, _ = select_disclosures(rows, CUTOFF, limit=20)

    boundary = datetime(2024, 9, 2)
    assert items, "nothing survived; the test would pass vacuously"
    for i in items:
        assert datetime.fromisoformat(i.filed_at) < boundary


def test_unparseable_timestamps_are_dropped_not_assumed_safe():
    rows = [row("no stamp", ""), row("bad stamp", "not-a-date"),
            row("good", "2024-08-30T10:00:00")]
    items, _ = select_disclosures(rows, CUTOFF, limit=10)
    assert [i.headline for i in items] == ["good"]


# ---------------------------------------------------------------------------
# the noise filter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subcat", [
    "Reg. 39 (3) - Details of Loss of Certificate / Duplicate Certificate",
    "reg. 39 (3) - details of loss of certificate / duplicate certificate",
    "Newspaper Publication",
])
def test_administrative_subcategories_are_noise(subcat):
    assert is_noise(Item("anything", "Others", subcat, "2024-08-30T10:00:00"))


@pytest.mark.parametrize("headline", [
    "Disclosure under Regulation 30 of the Securities and Exchange Board of India",
    "Announcement under Regulation 30 (LODR)-Newspaper Publication",
    "Compliances-Reg. 39 (3) - Details of Loss of Certificate",
    "Intimation Under Regulation 30 Of The SEBI",
])
def test_bare_regulation_citations_are_noise(headline):
    assert is_noise(Item(headline, "Company Update", "General", "2024-08-30T10:00:00"))


@pytest.mark.parametrize("headline", [
    "Allotment of 3302544 Equity Shares pursuant to exercise of Warrants",
    "Transcript Of Earnings Call For The First Quarter Ended June 30, 2023",
    "TCS Completes 150 Digital Transformation Projects Using Freshworks",
    "Board Meeting Intimation for Considering And Approving The Audited Financial Results",
    "Post Buyback Public Announcement for Buyback of Equity Shares",
])
def test_substantive_filings_survive_the_noise_filter(headline):
    assert not is_noise(Item(headline, "Company Update", "General", "2024-08-30T10:00:00"))


def test_noise_is_counted_separately_from_duplicates():
    rows = [
        row("Disclosure under Regulation 30 of SEBI", "2024-08-30T10:00:00"),
        row("Real news about a plant", "2024-08-29T10:00:00"),
        row("Real news about a plant", "2024-08-28T10:00:00"),
    ]
    items, dropped = select_disclosures(rows, CUTOFF, limit=10)
    assert dropped["noise"] == 1
    assert dropped["duplicate"] == 1
    assert len(items) == 1


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("<b>Format of Initial Disclosure</b><br/>", "Format of Initial Disclosure"),
    ("<table border=''1px''><tr><td>Sr. No.</td></tr></table>", "Sr. No."),
    ("Profit &amp; Loss&nbsp;Statement", "Profit & Loss Statement"),
    ("TA''ZIZ and Reliance Sign", "TA'ZIZ and Reliance Sign"),
    ("Line one\nLine two", "Line one Line two"),
    ("   padded   ", "padded"),
    ("", ""),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_cleaning_happens_on_read_so_the_cache_stays_verbatim():
    rows = [row("<b>Board Meeting</b> for results", "2024-08-30T10:00:00")]
    items, _ = select_disclosures(rows, CUTOFF, limit=5)
    assert items[0].headline == "Board Meeting for results"
    # the input dict is untouched
    assert rows[0]["headline"] == "<b>Board Meeting</b> for results"


# ---------------------------------------------------------------------------
# de-duplication, ordering, limits
# ---------------------------------------------------------------------------

def test_normalise_headline_collapses_case_and_punctuation():
    assert normalise_headline("Board Meeting!") == normalise_headline("board meeting")


def test_most_recent_first_and_limit_respected():
    rows = [
        row("oldest", "2024-08-20T10:00:00"),
        row("newest", "2024-08-31T10:00:00"),
        row("middle", "2024-08-25T10:00:00"),
    ]
    items, _ = select_disclosures(rows, CUTOFF, limit=2)
    assert [i.headline for i in items] == ["newest", "middle"]


def test_selection_is_deterministic():
    rows = [row(f"filing {i}", f"2024-08-{20 + (i % 10):02d}T10:00:00") for i in range(15)]
    first = [i.headline for i in select_disclosures(rows, CUTOFF, limit=5)[0]]
    for _ in range(5):
        assert [i.headline for i in select_disclosures(rows, CUTOFF, limit=5)[0]] == first


def test_empty_input_is_not_an_error():
    items, dropped = select_disclosures([], CUTOFF)
    assert items == []
    assert dropped == {"after_cutoff": 0, "noise": 0, "duplicate": 0}


# ---------------------------------------------------------------------------
# the prompt block — citation handles for the L2 check
# ---------------------------------------------------------------------------

def test_prompt_block_numbers_filings_from_one():
    bundle = DisclosureBundle(
        symbol="RELIANCE", cutoff="2024-09-02", status="ok",
        items=[Item("First filing", "Board Meeting", "General", "2024-08-30T10:00:00"),
               Item("Second filing", "Result", "General", "2024-08-29T10:00:00")],
    )
    block = bundle.as_prompt_block()
    assert block.startswith("[1] 2024-08-30  [Board Meeting] First filing")
    assert "[2] 2024-08-29  [Result] Second filing" in block


def test_prompt_block_is_explicit_when_the_window_is_empty():
    bundle = DisclosureBundle(symbol="X", cutoff="2024-09-02", status="empty")
    assert "no disclosures" in bundle.as_prompt_block()
