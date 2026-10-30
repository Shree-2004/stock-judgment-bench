"""M3 acceptance tests: the prediction ledger, offline.

The model and market data are fakes; what is under test is the ledger's
contract:
- rows are append-only and hash-chained, and tampering is detected
- evidence is limited to filings this system had seen by prediction time
- failures are recorded rather than dropped, and reruns never touch successes
- a quota stop leaves no row for the stock it interrupted
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pytest

from live import ledger, store
from live.filings import FilingRecord, provider_ref
from live.ledger import (GENESIS, WARMUP_END, Prediction, append, append_chain_line, ensure_schema,
                         evidence_for, start_round, verify_chain)
from llm.client import DailyQuotaExhausted
from scripts.live_predict import run_round

SESSION = date(2026, 11, 2)
NOW = datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc)        # 18:30 IST, after the close
S = " |SUBJECT: "


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "live.sqlite")
    ensure_schema(c)
    return c


def _filing(company, subject, published, url=None):
    return FilingRecord("nse_rss", provider_ref(company, published, url, subject), company, subject, url,
                        published, "raw")


def _seed(conn, rows, first_seen):
    job = store.start_job(conn, "ingest_filings", "nse_rss", first_seen)
    store.insert_filings(conn, rows, first_seen, job)


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def test_evidence_is_point_in_time_and_uses_v1_selection(conn):
    seen_early = "2026-11-02T08:00:00+00:00"
    _seed(conn, [
        _filing("Infosys Limited", "Infosys has informed the Exchange regarding Acquisition of X" + S + "Acquisition",
                "2026-11-02T07:00:00+00:00"),
        _filing("INFOSYS LTD.", "Infosys Limited has submitted to the Exchange, the financial results" + S
                + "Outcome of Board Meeting", "2026-10-30T10:00:00+00:00"),
        _filing("Infosys Limited", "Copy of Newspaper Publication" + S + "Copy of Newspaper Publication",
                "2026-11-01T10:00:00+00:00"),                                  # routine -> dropped
        _filing("Infosys Limited", "Old news" + S + "Press Release", "2026-10-01T10:00:00+00:00"),  # > 21 days
        _filing("Infosys BPM Limited", "Not the same company" + S + "Acquisition", "2026-11-02T07:00:00+00:00"),
    ], seen_early)
    # Published before the prediction, but this system only saw it afterwards.
    _seed(conn, [_filing("Infosys Limited", "Late-seen order win worth Rs 900 crore from a bank" + S
                         + "General Updates", "2026-11-02T06:00:00+00:00")], "2026-11-02T14:00:00+00:00")

    bundle, kept = evidence_for(conn, "INFY", SESSION, NOW.isoformat())
    cats = [k["category"] for k in kept]
    assert cats == ["deal", "results"]                       # newest first, routine and old dropped
    assert bundle.dropped_as_noise == 1
    assert all(k["first_seen_at"] <= NOW.isoformat() for k in kept)
    assert "[1]" in bundle.as_prompt_block() and "Acquisition" in bundle.as_prompt_block()


def test_evidence_keeps_at_most_five_distinct_headlines(conn):
    rows = [_filing("Infosys Limited", f"Infosys order win number {i} worth Rs {i} crore" + S + "General Updates",
                    f"2026-11-01T0{i}:00:00+00:00") for i in range(8)]
    rows.append(_filing("Infosys Limited", "Infosys order win number 7 worth Rs 7 crore" + S + "General Updates",
                        "2026-11-01T09:30:00+00:00"))        # duplicate headline
    _seed(conn, rows, "2026-11-01T12:00:00+00:00")
    bundle, kept = evidence_for(conn, "INFY", SESSION, NOW.isoformat())
    assert len(kept) == 5 and bundle.dropped_as_duplicate == 1


def test_no_filings_is_an_explicit_empty_block(conn):
    bundle, kept = evidence_for(conn, "TCS", SESSION, NOW.isoformat())
    assert kept == [] and "no disclosures" in bundle.as_prompt_block()


# ---------------------------------------------------------------------------
# chain
# ---------------------------------------------------------------------------

def _pred(sym, **kw):
    return Prediction(symbol=sym, made_at=NOW.isoformat(), model="m", prompt_version="v1", evidence=[], **kw)


def test_rows_chain_from_genesis_and_verify(conn):
    rid = start_round(conn, SESSION, NOW.isoformat(), "m", "v1")
    a = append(conn, rid, SESSION, _pred("INFY", verdict="BULLISH"))
    b = append(conn, rid, SESSION, _pred("TCS", verdict="BEARISH"))
    assert a["prev_hash"] == GENESIS and b["prev_hash"] == a["entry_hash"]
    assert verify_chain(conn) == (True, 2, None)


def test_ledger_is_append_only(conn):
    rid = start_round(conn, SESSION, NOW.isoformat(), "m", "v1")
    append(conn, rid, SESSION, _pred("INFY", verdict="BULLISH"))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE predictions SET verdict = 'BEARISH'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM predictions")


def test_tampering_is_detected_even_past_the_triggers(conn):
    rid = start_round(conn, SESSION, NOW.isoformat(), "m", "v1")
    for sym in ("INFY", "TCS", "ITC"):
        append(conn, rid, SESSION, _pred(sym, verdict="NEUTRAL"))
    conn.execute("DROP TRIGGER predictions_no_update")               # someone with file access
    conn.execute("UPDATE predictions SET verdict = 'BULLISH' WHERE symbol = 'TCS'")
    assert verify_chain(conn) == (False, 2, "2")


def test_warmup_flag_follows_the_evidence_window(conn):
    rid = start_round(conn, SESSION, NOW.isoformat(), "m", "v1")
    early = append(conn, rid, SESSION, _pred("INFY"))
    later = append(conn, rid, WARMUP_END, _pred("INFY"))
    assert (early["warmup"], later["warmup"]) == (1, 0)


def test_chain_line_is_one_readable_line_per_round(tmp_path):
    path = tmp_path / "ledger" / "chain.txt"
    append_chain_line(path, SESSION, "ok", 41, "a" * 64, "2026-11-02T13:05:00+00:00")
    append_chain_line(path, date(2026, 11, 3), "partial", 70, "b" * 64, "2026-11-03T13:05:00+00:00")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[1].startswith("2026-11-03  partial") and "head=" + "b" * 64 in lines[1]


# ---------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------

@dataclass
class FakeSnap:
    price: float = 100.0

    def as_dict(self):
        return {"price": self.price, "as_of": SESSION.isoformat()}


@dataclass
class FakeBrief:
    verdict: str | None = "BULLISH"
    confidence: int | None = 3

    @property
    def parsed(self):
        return self.verdict is not None


@dataclass
class FakeResult:
    brief: FakeBrief
    raw_text: str = "VERDICT: BULLISH"

    def as_dict(self):
        return {"brief": {"verdict": self.brief.verdict}}


def _round(conn, symbols, generate, snapshot=lambda s, d: FakeSnap()):
    return run_round(conn, SESSION, symbols, now=lambda: NOW, snapshot_fn=snapshot, generate_fn=generate,
                     model="m", prompt_version="v1")


def test_full_round_records_every_stock(conn):
    out = _round(conn, ["INFY", "TCS", "ITC"], lambda snap, b: FakeResult(FakeBrief()))
    assert (out["state"], out["ok"], out["failed"]) == ("ok", 3, 0)
    assert verify_chain(conn)[0]


def test_failures_are_rows_and_reruns_only_retry_them(conn):
    def flaky(snap, bundle):
        if bundle.symbol == "TCS":
            raise RuntimeError("provider 500")
        return FakeResult(FakeBrief())

    first = _round(conn, ["INFY", "TCS", "ITC"], flaky)
    assert (first["state"], first["ok"], first["failed"]) == ("partial", 2, 1)
    err = conn.execute("SELECT error FROM predictions WHERE symbol = 'TCS'").fetchone()[0]
    assert "provider 500" in err

    second = _round(conn, ["INFY", "TCS", "ITC"], lambda snap, b: FakeResult(FakeBrief("BEARISH")))
    assert (second["state"], second["attempted"]) == ("ok", 1)
    rows = conn.execute("SELECT symbol, attempt, verdict FROM predictions ORDER BY seq").fetchall()
    assert [tuple(r) for r in rows] == [("INFY", 1, "BULLISH"), ("TCS", 1, None), ("ITC", 1, "BULLISH"),
                                        ("TCS", 2, "BEARISH")]


def test_unparseable_answer_is_a_failure_not_a_verdict(conn):
    out = _round(conn, ["INFY"], lambda snap, b: FakeResult(FakeBrief(verdict=None)))
    assert out["failed"] == 1
    assert conn.execute("SELECT error FROM predictions").fetchone()[0].startswith("parse:")


def test_quota_stop_leaves_no_row_for_the_interrupted_stock(conn):
    calls = []

    def quota_after_one(snap, bundle):
        calls.append(bundle.symbol)
        if len(calls) > 1:
            raise DailyQuotaExhausted("daily quota")
        return FakeResult(FakeBrief())

    out = _round(conn, ["INFY", "TCS", "ITC"], quota_after_one)
    assert out["state"] == "partial" and out["quota"]
    assert [r[0] for r in conn.execute("SELECT symbol FROM predictions")] == ["INFY"]


def test_market_data_failure_is_recorded_with_its_evidence(conn):
    def no_price(sym, d):
        raise ledger_error("no close for ITC")

    out = _round(conn, ["ITC"], lambda snap, b: FakeResult(FakeBrief()), snapshot=no_price)
    assert out["failed"] == 1
    row = conn.execute("SELECT error, evidence_json, price FROM predictions").fetchone()
    assert "no close" in row[0] and row[1] == "[]" and row[2] is None


def ledger_error(msg):
    from data.market_data import MarketDataError
    return MarketDataError(msg)


# ---------------------------------------------------------------------------
# tracks (second model, 2026-10-30)
# ---------------------------------------------------------------------------

def test_two_models_each_get_a_call_per_stock_on_one_chain(conn):
    rid = start_round(conn, SESSION, NOW.isoformat(), "m", "v1")
    a = append(conn, rid, SESSION, _pred("INFY", verdict="BULLISH"))
    b = append(conn, rid, SESSION, Prediction(symbol="INFY", made_at=NOW.isoformat(), model="other@low",
                                              prompt_version="v1", evidence=[], verdict="BEARISH"))
    assert (a["attempt"], b["attempt"]) == (1, 1) and b["prev_hash"] == a["entry_hash"]
    assert verify_chain(conn) == (True, 2, None)


def test_one_tracks_successes_do_not_skip_the_other_track(conn):
    _round(conn, ["INFY", "TCS"], lambda snap, b: FakeResult(FakeBrief()))
    out = run_round(conn, SESSION, ["INFY", "TCS"], now=lambda: NOW, snapshot_fn=lambda s, d: FakeSnap(),
                    generate_fn=lambda snap, b: FakeResult(FakeBrief("BEARISH")), model="other@low",
                    prompt_version="v1")
    assert (out["attempted"], out["ok"]) == (2, 2)


def test_migration_to_the_track_key_keeps_every_row_and_the_head(tmp_path):
    c = store.connect(tmp_path / "old.sqlite")
    c.executescript(ledger._SCHEMA.replace("UNIQUE (session_date, symbol, model, prompt_version, attempt)",
                                           ledger._OLD_UNIQUE))
    rid = start_round(c, SESSION, NOW.isoformat(), "m", "v1")
    for sym in ("INFY", "TCS", "ITC"):
        append(c, rid, SESSION, _pred(sym, verdict="NEUTRAL"))
    before = [tuple(r) for r in c.execute("SELECT seq, entry_hash FROM predictions ORDER BY seq")]
    head = ledger.head_hash(c)

    ensure_schema(c)                                                  # migrates

    sql = c.execute("SELECT sql FROM sqlite_master WHERE name = 'predictions'").fetchone()[0]
    assert ledger._OLD_UNIQUE not in sql and "model, prompt_version, attempt" in sql
    assert [tuple(r) for r in c.execute("SELECT seq, entry_hash FROM predictions ORDER BY seq")] == before
    assert ledger.head_hash(c) == head and verify_chain(c) == (True, 3, None)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        c.execute("DELETE FROM predictions")
    ensure_schema(c)                                                  # second call is a no-op
    assert ledger.head_hash(c) == head
