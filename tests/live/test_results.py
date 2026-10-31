"""Quarterly-results collector (fundamentals track, step 1), offline.

The listing and both XBRL files are real NSE downloads from 2026-10-31; the
network is a fake. What is under test:
- only Financials rows are kept, with NSE's broadcast time converted to UTC,
  and a revision dated by its own revision time
- the quarter's numbers come out exactly as filed, for a company and a bank
- a stored filing is never fetched or changed again, and the table refuses edits
- a failed download stores nothing, is recorded, and is retried next run
"""
from __future__ import annotations

import json
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from live import results, store
from live.ingest import PermanentFetchError, TransientFetchError

FIX = Path(__file__).parent / "fixtures"
LISTING = json.loads((FIX / "results_listing_hero.json").read_text(encoding="utf-8"))
HERO = (FIX / "results_hero_q2fy27.xml").read_bytes()
HDFC = (FIX / "results_hdfcbank_q2fy27.xml").read_bytes()
NOW = datetime(2026, 10, 31, 18, 0, tzinfo=timezone.utc)


def test_listing_keeps_financials_and_converts_nse_time_to_utc():
    listings, rejects = results.parse_listing(LISTING, "HEROMOTOCO")
    assert rejects == []
    assert {x.statement for x in listings} == {"Consolidated", "Standalone"}     # governance rows dropped
    latest = listings[0]
    assert (latest.quarter_end, latest.kind, latest.published_at) == \
        ("2026-09-30", "Original", "2026-10-28T15:38:16+00:00")                  # 21:08:16 IST


def test_a_revision_is_dated_by_its_own_revision_time():
    revision = next(x for x in results.parse_listing(LISTING, "HEROMOTOCO")[0] if x.kind == "Revision")
    assert (revision.quarter_end, revision.published_raw) == ("2025-06-30", "06-AUG-2025 19:29:32")


def test_a_row_for_another_company_is_rejected():
    payload = {"data": [dict(LISTING["data"][0], symbol="TVSMOTOR")], "totalCount": 1}
    listings, rejects = results.parse_listing(payload, "HEROMOTOCO")
    assert listings == [] and "symbol mismatch" in rejects[0].reason


def test_quarter_numbers_come_out_exactly_as_filed():
    facts, period = results.quarter_facts(results.parse_xbrl(HERO))
    assert period == {"start": "2026-07-01", "end": "2026-09-30"}
    assert (facts["RevenueFromOperations"], facts["ProfitLossForPeriod"],
            facts["BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations"]) == \
        ("149705000000", "18605500000", "92.99")


def test_breakdown_contexts_are_not_mixed_into_the_totals():
    parsed = results.parse_xbrl(HERO)
    assert "OneExpenses1D" not in parsed["periods"] and "OneD" in parsed["periods"]


def test_a_bank_files_a_different_taxonomy_and_it_is_kept_whole():
    facts, period = results.quarter_facts(results.parse_xbrl(HDFC))
    assert period["end"] == "2026-09-30"
    assert "RevenueFromOperations" not in facts
    assert (facts["InterestEarned"], facts["ProfitLossForThePeriod"]) == ("951316300000", "207000300000")


class FakeNSE:
    """Serves the Hero listing and XBRL; `broken` URLs fail."""

    def __init__(self, broken=()):
        self.calls, self.broken = [], set(broken)

    def __call__(self, url):
        self.calls.append(url)
        if any(b in url for b in self.broken):
            raise PermanentFetchError("HTTP 403")
        if "integrated-filing-results" in url:
            if "symbol=HEROMOTOCO" not in url:
                return json.dumps({"data": [], "totalCount": 0}).encode()
            return json.dumps(LISTING).encode()
        return HERO


def _run(tmp_path, fetch, symbols=("HEROMOTOCO",)):
    return results.run_once(list(symbols), tmp_path / "live.sqlite", fetch=fetch,
                            now=lambda: NOW, sleep=lambda s: None)


def test_collects_each_filing_once_and_never_changes_it(tmp_path):
    first = _run(tmp_path, FakeNSE())
    n = sum(1 for x in results.parse_listing(LISTING, "HEROMOTOCO")[0])
    assert (first["state"], first["inserted"], first["rejected"]) == ("ok", n, 0)

    conn = store.connect(tmp_path / "live.sqlite")
    row = conn.execute("SELECT * FROM results WHERE quarter_end = '2026-09-30' "
                       "AND statement = 'Consolidated'").fetchone()
    assert zlib.decompress(row["raw_xbrl"]) == HERO
    assert json.loads(row["facts_json"])["facts"]["OneD"]["RevenueFromOperations"] == "149705000000"

    fake = FakeNSE()
    again = _run(tmp_path, fake)
    assert (again["inserted"], again["duplicates"]) == (0, n)
    assert not any("xbrl/" in u for u in fake.calls)                      # nothing re-downloaded
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE results SET facts_json = '{}'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM results")


def test_a_failed_download_stores_nothing_and_is_retried(tmp_path):
    latest = LISTING["data"][0]["xbrl"].rsplit("/", 1)[-1]
    run = _run(tmp_path, FakeNSE(broken=[latest]))
    assert run["state"] == "ok" and run["rejected"] == 1 and "xbrl failed" in run["rejects_json"]
    conn = store.connect(tmp_path / "live.sqlite")
    assert conn.execute("SELECT COUNT(*) FROM results WHERE provider_ref LIKE ?", (f"%{latest}",)).fetchone()[0] == 0

    assert _run(tmp_path, FakeNSE())["inserted"] == 1


def test_one_dead_listing_is_a_reject_all_dead_is_a_failed_job(tmp_path):
    some = _run(tmp_path, FakeNSE(broken=["symbol=TCS"]), symbols=("HEROMOTOCO", "TCS"))
    assert some["state"] == "ok" and "listing failed" in some["rejects_json"]
    none = _run(tmp_path, FakeNSE(broken=["integrated-filing-results"]), symbols=("HEROMOTOCO", "TCS"))
    assert none["state"] == "failed" and none["error"] == "2 of 2 listings failed"


def test_transient_errors_are_retried_and_pages_are_followed():
    page1 = {"data": LISTING["data"][:2], "totalCount": 3}
    page2 = {"data": LISTING["data"][2:3], "totalCount": 3}
    seen, flaky = [], iter([TransientFetchError("HTTP 503")])

    def fetch(url):
        seen.append(url)
        if "page=2" in url:
            return json.dumps(page2).encode()
        err = next(flaky, None)
        if err:
            raise err
        return json.dumps(page1).encode()

    listings, _ = results.list_filings("HEROMOTOCO", fetch=fetch, sleep=lambda s: None)
    assert len(seen) == 3 and seen[-1].endswith("&page=2")
    assert len(listings) == 2            # row 3 is a Governance filing: listed, not kept


def test_symbols_with_an_ampersand_are_escaped():
    seen = []
    results.list_filings("M&M", fetch=lambda u: seen.append(u) or b'{"data": [], "totalCount": 0}',
                         sleep=lambda s: None)
    assert "symbol=M%26M&" in seen[0]
