"""The static-site export: what the public page is allowed to show.

Offline: no prices are fetched. What is under test is that the export shows
the call that counts (the latest attempt), links each filing the model was
shown, and carries the chain check through rather than assuming it.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from live import store
from live.filings import FilingRecord, provider_ref
from live.ledger import Prediction, append, ensure_schema, start_round
from scripts import export_site

SESSION = date(2026, 11, 2)
NOW = datetime(2026, 11, 2, 13, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(export_site, "witness", lambda head: {"witnessed": True, "pushed": None})
    c = store.connect(tmp_path / "live.sqlite")
    ensure_schema(c)
    return c


def _pred(sym, **kw):
    return Prediction(symbol=sym, made_at=NOW.isoformat(), model="m", prompt_version="v1", **kw)


def test_export_shows_the_latest_attempt_and_links_evidence(conn):
    published, url = "2026-11-02T07:00:00+00:00", "https://x/1.pdf"
    ref = provider_ref("Infosys Limited", published, url, "s")
    job = store.start_job(conn, "ingest_filings", "nse_rss", "2026-11-02T08:00:00+00:00")
    store.insert_filings(conn, [FilingRecord("nse_rss", ref, "Infosys Limited", "s", url, published, "raw")],
                         "2026-11-02T08:00:00+00:00", job)
    rid = start_round(conn, SESSION, NOW.isoformat(), "m", "v1")
    append(conn, rid, SESSION, _pred("TCS", error="RuntimeError: provider 500", evidence=[]))
    append(conn, rid, SESSION, _pred("INFY", verdict="BULLISH", confidence=4, price=1146.5,
                                     evidence=[{"category": "deal", "headline": "h", "provider_ref": ref,
                                                "first_seen_at": "2026-11-02T08:00:00+00:00"}]))
    append(conn, rid, SESSION, _pred("TCS", verdict="BEARISH", confidence=2, evidence=[]))   # rerun

    out = export_site.export_ledger(conn)
    calls = out["tracks"]["m · v1"]["sessions"][SESSION.isoformat()]
    assert (calls["TCS"]["verdict"], calls["TCS"]["attempt"], calls["TCS"]["error"]) == ("BEARISH", 2, None)
    assert calls["INFY"]["evidence"][0]["url"] == "https://x/1.pdf"
    assert out["chain"]["ok"] and out["chain"]["rows"] == 3


def test_settled_calls_carry_their_outcome_with_v1s_hit_rule():
    assert export_site._outcome("BULLISH", ("2026-12-01", 3.0, 1.2))["right"] is True
    assert export_site._outcome("BEARISH", ("2026-12-01", 3.0, 1.2))["right"] is False
    assert export_site._outcome("BEARISH", ("2026-12-01", -3.0, 0.0))["right"] is False    # flat isn't a hit
    assert export_site._outcome("NEUTRAL", ("2026-12-01", 3.0, 1.2))["right"] is None
    assert export_site._outcome("BULLISH", None) is None                                  # not settled yet


def test_a_failed_site_refresh_is_reported_not_raised():
    from scripts.live_predict import refresh_site

    def boom():
        raise OSError("disk full")

    assert refresh_site(boom) == "failed: OSError: disk full"
    assert refresh_site(lambda: "41 calls") == "41 calls"


def test_export_of_an_empty_ledger_says_so(conn):
    out = export_site.export_ledger(conn)
    assert out["tracks"] == {} and out["chain"]["head"] is None
