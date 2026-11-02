"""The fundamentals block (fundamentals track, step 2), offline.

What is under test:
- a filing counts only once NSE has published it, and a revision only from
  its own publication time
- banks are labelled by what they file ("Interest earned"), never "revenue"
- a quarter filed only as year-to-date is derived from two filed figures, or
  left out; never guessed
- year-on-year, four-quarter EPS and P/E appear only when their inputs exist
- one statement series, never mixed
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from live import store
from live.fundamentals import Quarter, load
from live.results import Listing, parse_xbrl

FIX = Path(__file__).parent / "fixtures"
JOB = 1          # the one job_runs row in a fresh store


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "live.sqlite")
    assert store.start_job(c, "ingest_results", "nse_integrated_filing", "2026-11-01T00:00:00+00:00") == JOB
    return c


def _quarter_start(end: str) -> str:
    d = date.fromisoformat(end)
    return date(d.year, d.month - 2, 1).isoformat()


def _file(conn, symbol, end, published, facts=None, statement="Consolidated", ytd=None, kind="Original", ref=None):
    """One results row. `facts` go in a three-month context, `ytd` in a
    1-April-to-`end` context."""
    periods, by_ctx = {}, {}
    if facts is not None:
        periods["OneD"], by_ctx["OneD"] = {"start": _quarter_start(end), "end": end}, {k: str(v) for k, v in facts.items()}
    if ytd is not None:
        d = date.fromisoformat(end)
        fy = date(d.year if d.month > 3 else d.year - 1, 4, 1).isoformat()
        periods["YTD"], by_ctx["YTD"] = {"start": fy, "end": end}, {k: str(v) for k, v in ytd.items()}
    item = Listing(symbol, ref or f"https://x/{symbol}/{statement}/{end}/{kind}/{published}", None, kind,
                   statement, end, published, published)
    store.insert_result(conn, item, {"periods": periods, "facts": by_ctx}, b"<xbrl/>", published, JOB)


def _co(revenue, profit, eps):
    return {"RevenueFromOperations": revenue * 1e7, "ProfitLossForPeriod": profit * 1e7,
            "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations": eps}


FOUR = (("2025-12-31", "2026-02-05T10:00:00+00:00", _co(100, 10, 2.0)),
        ("2026-03-31", "2026-05-05T10:00:00+00:00", _co(110, 11, 2.5)),
        ("2026-06-30", "2026-08-05T10:00:00+00:00", _co(120, 12, 3.0)),
        ("2026-09-30", "2026-10-28T15:38:16+00:00", _co(130, 13, 3.5)))


def test_results_published_after_the_call_are_invisible(conn):
    for end, pub, facts in FOUR:
        _file(conn, "HERO", end, pub, facts)
    before = load(conn, "HERO", "2026-10-28T13:00:00+00:00")      # 18:30 IST call, results at 21:08 IST
    after = load(conn, "HERO", "2026-10-29T13:00:00+00:00")
    assert (before.latest.label, after.latest.label) == ("Q1 FY27", "Q2 FY27")


def test_a_revision_counts_only_from_its_own_publication(conn):
    _file(conn, "X", "2026-06-30", "2026-08-05T10:00:00+00:00", _co(120, 12, 3.0))
    _file(conn, "X", "2026-06-30", "2026-09-01T10:00:00+00:00", _co(120, 9, 2.0), kind="Revision")
    assert load(conn, "X", "2026-08-20T00:00:00+00:00").latest.profit == 12 * 1e7
    assert load(conn, "X", "2026-09-02T00:00:00+00:00").latest.profit == 9 * 1e7


def test_four_quarter_eps_pe_and_growth(conn):
    for end, pub, facts in FOUR:
        _file(conn, "HERO", end, pub, facts)
    _file(conn, "HERO", "2025-09-30", "2025-11-01T10:00:00+00:00", _co(100, 10, 2.0))
    f = load(conn, "HERO", "2026-11-02T13:00:00+00:00", close=330.0)
    figs = f.figures()
    assert figs == {
        "Revenue from operations, latest quarter (Rs cr)": 130.0,
        "Net profit, latest quarter (Rs cr)": 13.0,
        "EPS, latest quarter (Rs)": 3.5,
        "Revenue from operations growth YoY (%)": 30.0,
        "Net profit growth YoY (%)": 30.0,
        "EPS, last 4 quarters (Rs)": 11.0,
        "P/E on last 4 quarters' EPS": 30.0,
    }
    block = f.as_prompt_block(date(2026, 11, 2))
    assert block.startswith("Latest results: Q2 FY27 (quarter ended 30 Sep 2026), consolidated, published 28 Oct 2026")
    assert "- P/E on last 4 quarters' EPS: 30.00" in block


def test_no_year_ago_quarter_means_no_growth_and_the_block_says_so(conn):
    _file(conn, "X", "2026-09-30", "2026-10-28T10:00:00+00:00", _co(130, 13, 3.5))
    f = load(conn, "X", "2026-11-02T00:00:00+00:00", close=100)
    assert not any("growth" in k or "4 quarters" in k or "P/E" in k for k in f.figures())
    assert "not available, so no year-on-year figures" in f.as_prompt_block(date(2026, 11, 2))


def test_growth_off_a_loss_and_pe_on_negative_eps_are_left_out(conn):
    _file(conn, "X", "2025-09-30", "2025-10-28T10:00:00+00:00", _co(100, -5, -1.0))
    _file(conn, "X", "2025-12-31", "2026-01-28T10:00:00+00:00", _co(100, -5, -1.0))
    _file(conn, "X", "2026-03-31", "2026-04-28T10:00:00+00:00", _co(100, -5, -1.0))
    _file(conn, "X", "2026-06-30", "2026-07-28T10:00:00+00:00", _co(100, -5, -1.0))
    _file(conn, "X", "2026-09-30", "2026-10-28T10:00:00+00:00", _co(120, 3, 0.5))
    figs = load(conn, "X", "2026-11-02T00:00:00+00:00", close=100).figures()
    assert "Net profit growth YoY (%)" not in figs and "P/E on last 4 quarters' EPS" not in figs
    assert figs["Revenue from operations growth YoY (%)"] == 20.0


def test_a_gap_in_the_last_four_quarters_means_no_four_quarter_eps(conn):
    for end, pub, facts in FOUR[:1] + FOUR[2:]:                       # Mar-2026 missing
        _file(conn, "X", end, pub, facts)
    _file(conn, "X", "2025-09-30", "2025-11-01T10:00:00+00:00", _co(100, 10, 2.0))
    assert load(conn, "X", "2026-11-02T00:00:00+00:00", close=330).eps_4q() is None


def test_a_bank_is_labelled_by_what_it_files(conn):
    store.insert_result(conn, Listing("HDFCBANK", "https://x/hdfc", None, "Original", "Consolidated", "2026-09-30",
                                      "2026-10-17T12:17:00+00:00", "17-Oct-2026 17:47:13"),
                        parse_xbrl((FIX / "results_hdfcbank_q2fy27.xml").read_bytes()), b"x", "now", JOB)
    figs = load(conn, "HDFCBANK", "2026-11-02T00:00:00+00:00").figures()
    assert figs["Interest earned, latest quarter (Rs cr)"] == 95131.63
    assert figs["Net profit, latest quarter (Rs cr)"] == 20700.03
    assert not any(k.startswith("Revenue") for k in figs)


def test_a_half_year_only_filing_is_derived_from_two_filed_figures(conn):
    # Coal India, Sep 2026: the filing carries Apr-Sep only. Real figures.
    _file(conn, "COALINDIA", "2026-06-30", "2026-07-27T12:19:08+00:00",
          ytd=_co(46254.80, 8849.81, 14.36))                               # Q1: its quarter is its YTD
    _file(conn, "COALINDIA", "2026-09-30", "2026-10-29T12:53:19+00:00", ytd=_co(90433.54, 15397.24, 24.94))
    q = load(conn, "COALINDIA", "2026-11-02T00:00:00+00:00").latest
    assert q.derived and q.end == date(2026, 9, 30)
    assert round(q.revenue / 1e7, 2) == 44178.74 and round(q.eps, 2) == 10.58
    assert "derived from year-to-date figures" in load(conn, "COALINDIA", "2026-11-02T00:00:00+00:00") \
        .as_prompt_block(date(2026, 11, 2))


def test_an_underivable_quarter_is_left_out_not_guessed(conn):
    # Q1's own quarter is its year-to-date, so it can be subtracted; a missing Q1 filing cannot.
    _file(conn, "X", "2026-03-31", "2026-05-05T00:00:00+00:00", _co(110, 11, 2.5))
    _file(conn, "X", "2026-09-30", "2026-10-29T00:00:00+00:00", ytd=_co(250, 25, 6.0))
    assert load(conn, "X", "2026-11-02T00:00:00+00:00").latest.end == date(2026, 3, 31)


def test_one_statement_series_is_used_never_mixed(conn):
    _file(conn, "X", "2026-06-30", "2026-08-01T00:00:00+00:00", _co(999, 99, 9.0), statement="Consolidated")
    _file(conn, "X", "2026-06-30", "2026-08-01T00:00:00+00:00", _co(100, 10, 1.0), statement="Standalone")
    _file(conn, "X", "2026-09-30", "2026-10-29T00:00:00+00:00", _co(130, 13, 1.3), statement="Standalone")
    f = load(conn, "X", "2026-11-02T00:00:00+00:00")
    assert f.statement == "Standalone" and [q.revenue / 1e7 for q in f.quarters] == [130, 100]


def test_nothing_filed_yet_says_so(conn):
    f = load(conn, "X", "2026-11-02T00:00:00+00:00")
    assert f.figures() == {} and f.as_prompt_block(date(2026, 11, 2)).startswith("(no quarterly results")


@pytest.mark.parametrize("end,label", [("2026-09-30", "Q2 FY27"), ("2026-03-31", "Q4 FY26"),
                                       ("2025-12-31", "Q3 FY26"), ("2026-06-30", "Q1 FY27")])
def test_indian_fiscal_quarter_labels(end, label):
    assert Quarter(date.fromisoformat(end), "", None, None, None, None).label == label
