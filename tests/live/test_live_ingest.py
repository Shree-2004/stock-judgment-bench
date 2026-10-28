"""M1 acceptance tests: the filing watcher, offline.

Three fixture polls stand in for the feed over a day — A, then B half an hour
later, then C after an overnight sleep. Each acceptance criterion from the plan
maps to a test below:

- two identical runs create zero duplicates      -> test_rerun_*
- a late filing keeps its real published_at       -> test_late_filing_*
- failures are visible and erase nothing          -> test_*_failure_*
- nothing invalid or future-dated gets in         -> test_parse_*, test_future_*
- turnover between polls is reported, not hidden  -> test_coverage_gap_*
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from live import store
from live.filings import parse_nse_rss, parse_pubdate, tracked_symbol
from live.ingest import PermanentFetchError, TransientFetchError, fetch_feed, run_once

FIX = Path(__file__).parent / "fixtures"
FEED_A = (FIX / "feed_a.xml").read_bytes()
FEED_B = (FIX / "feed_b.xml").read_bytes()
FEED_C = (FIX / "feed_turnover.xml").read_bytes()

T_A = datetime(2026, 10, 28, 12, 30, tzinfo=timezone.utc)   # 18:00 IST
T_B = datetime(2026, 10, 28, 13, 0, tzinfo=timezone.utc)    # 18:30 IST
T_C = datetime(2026, 10, 29, 6, 0, tzinfo=timezone.utc)     # 11:30 IST next day


def poll(db, body, at, **kw):
    return run_once(db, fetch=lambda: body, now=lambda: at, sleep=lambda s: None, **kw)


def filings(db) -> list[sqlite3.Row]:
    conn = store.connect(db, read_only=True)
    return conn.execute("SELECT * FROM filings ORDER BY published_at").fetchall()


@pytest.fixture
def db(tmp_path):
    return tmp_path / "live.sqlite"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parse_converts_ist_to_utc_in_both_date_shapes():
    assert parse_pubdate("28-Oct-2026 17:44:30").isoformat() == "2026-10-28T12:14:30+00:00"
    assert parse_pubdate("28-OCT-2026 17:20").isoformat() == "2026-10-28T11:50:00+00:00"
    assert parse_pubdate("yesterday-ish") is None


def test_parse_rejects_malformed_items_with_a_reason():
    records, rejected = parse_nse_rss(FEED_A)
    assert len(records) == 8
    assert sorted(r.reason for r in rejected) == [
        "missing company", "missing pubDate", "unparseable pubDate"]


def test_parse_accepts_items_without_a_link():
    records, _ = parse_nse_rss(FEED_A)
    notice = next(r for r in records if r.company.startswith("R.K. Fashion"))
    assert notice.url is None


def test_identity_distinguishes_same_link_at_different_seconds():
    records, _ = parse_nse_rss(FEED_A)
    sammaan = [r.provider_ref for r in records if r.company.startswith("Sammaan")]
    assert len(sammaan) == 3 and len(set(sammaan)) == 2


def test_tracked_symbol_is_exact_after_normalisation():
    assert tracked_symbol("Infosys Limited") == "INFY"
    assert tracked_symbol("INFOSYS LTD.") == "INFY"
    assert tracked_symbol("HDFC Bank Limited - NCD") is None
    assert tracked_symbol("Reliance Industrial Infrastructure Limited") is None


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------

def test_first_poll_stores_valid_items_once_and_logs_counts(db):
    run = poll(db, FEED_A, T_A)
    assert run["state"] == "ok"
    assert (run["fetched"], run["inserted"], run["duplicates"], run["rejected"]) == (11, 6, 1, 4)
    assert len(filings(db)) == 6


def test_rerun_of_identical_feed_inserts_nothing(db):
    poll(db, FEED_A, T_A)
    before = [tuple(r) for r in filings(db)]
    run = poll(db, FEED_A, T_B)
    assert (run["inserted"], run["duplicates"]) == (0, 7)
    assert [tuple(r) for r in filings(db)] == before   # timestamps and URLs untouched


def test_future_dated_items_are_rejected(db):
    run = poll(db, FEED_A, T_A)
    assert "future-dated" in run["rejects_json"]
    assert not any(r["company"].startswith("Fabtech") for r in filings(db))


def test_late_filing_keeps_exchange_time_and_records_when_it_was_seen(db):
    poll(db, FEED_A, T_A)
    run = poll(db, FEED_B, T_B)
    assert (run["inserted"], run["duplicates"]) == (2, 3)

    hdfc = next(r for r in filings(db) if r["company"] == "HDFC Bank Limited")
    assert hdfc["published_at"] == "2026-10-28T11:25:00+00:00"   # 16:55 IST, as filed
    assert hdfc["first_seen_at"] == T_B.isoformat()              # not backdated to A
    assert hdfc["published_at"] < hdfc["first_seen_at"]

    infy = next(r for r in filings(db) if r["company"] == "Infosys Limited")
    assert infy["first_seen_at"] == T_A.isoformat()              # unchanged by B


def test_filings_table_is_append_only(db):
    poll(db, FEED_A, T_A)
    conn = sqlite3.connect(db)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE filings SET subject = 'edited'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM filings")


def test_overlapping_polls_report_no_coverage_gap(db):
    first = poll(db, FEED_A, T_A)
    second = poll(db, FEED_B, T_B)
    assert first["coverage_gap_s"] is None     # nothing to compare against yet
    assert second["coverage_gap_s"] is None


def test_coverage_gap_is_measured_when_the_feed_turned_over(db):
    poll(db, FEED_A, T_A)
    poll(db, FEED_B, T_B)
    run = poll(db, FEED_C, T_C)
    # newest stored: TCS 18:20 IST. oldest in C: 10:00 IST next day.
    assert run["coverage_gap_s"] == 15 * 3600 + 40 * 60


# ---------------------------------------------------------------------------
# failure handling
# ---------------------------------------------------------------------------

def test_transient_failure_is_retried_and_noted(db):
    calls = iter([TransientFetchError("HTTP 503"), FEED_A])

    def flaky():
        nxt = next(calls)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    run = run_once(db, fetch=flaky, now=lambda: T_A, sleep=lambda s: None)
    assert (run["state"], run["attempts"], run["inserted"]) == ("ok", 2, 6)
    assert "503" in run["error"]


def test_permanent_failure_is_not_retried_and_erases_nothing(db):
    poll(db, FEED_A, T_A)
    attempts = []

    def blocked():
        attempts.append(1)
        raise PermanentFetchError("HTTP 403: 'Access Denied'")

    run = run_once(db, fetch=blocked, now=lambda: T_B, sleep=lambda s: None)
    assert (run["state"], len(attempts)) == ("failed", 1)
    assert "403" in run["error"]
    assert len(filings(db)) == 6
    assert store.last_ok_job(store.connect(db, read_only=True), "ingest_filings")["id"] == 1


def test_transient_failures_give_up_after_the_retry_limit(db):
    run = run_once(db, fetch=lambda: (_ for _ in ()).throw(TransientFetchError("timeout")),
                   now=lambda: T_A, sleep=lambda s: None)
    assert (run["state"], run["attempts"]) == ("failed", 3)


def test_malformed_xml_that_persists_is_a_failed_run_not_an_empty_one(db):
    run = poll(db, b"<html>maintenance page</htm", T_A)
    assert (run["state"], run["attempts"]) == ("failed", 3)
    assert run["error"].startswith("parse:") and run["inserted"] == 0


def test_one_malformed_body_is_retried(db):
    # Seen live: a truncated/garbled feed, then a clean one seconds later.
    bodies = iter([FEED_A[: len(FEED_A) // 2], FEED_A])
    run = run_once(db, fetch=lambda: next(bodies), now=lambda: T_A, sleep=lambda s: None)
    assert (run["state"], run["attempts"], run["inserted"]) == ("ok", 2, 6)
    assert run["error"].startswith("parse:")


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_real_feed_parses():
    records, rejected = parse_nse_rss(fetch_feed())
    assert len(records) > 50, "feed is unexpectedly small"
    assert len(rejected) < len(records) * 0.05, "rejection rate suggests a format change"
