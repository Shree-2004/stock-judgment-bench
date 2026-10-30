"""M4 acceptance tests: settlement and the pre-registered analysis, offline.

Prices and the NSE calendar are fakes. What is under test:
- a call settles only once its 21-session window has closed, and only once
- warm-up calls and calls made after the next open are excluded
- the primary series is the first eligible session and every 21st after it,
  with a missing session reported rather than replaced
- v1's own rows, replayed through M4's summariser, give v1's published result
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from evals.outcomes import Outcome, OutcomeError
from live import store
from live.ledger import Prediction, append, ensure_schema, start_round
from live.settle import (FORWARD, analyse, ensure_schema as settle_schema, forward_end, newey_west_se,
                         primary_schedule, settle_pending, summarise)

ROOT = Path(__file__).resolve().parents[2]


def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


CAL = _weekdays(date(2026, 10, 29), 120)
RETURNS = {"INFY": 4.0, "TCS": -2.0, "ITC": 1.0}


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "live.sqlite")
    ensure_schema(c)
    settle_schema(c)
    return c


def _call(conn, session: date, verdicts: dict[str, str], late=False):
    made = datetime.combine(session, datetime.min.time(), timezone.utc) + timedelta(hours=13)   # 18:30 IST
    if late:
        made += timedelta(hours=17)                                                          # 11:30 IST next day
    rid = start_round(conn, session, made.isoformat(), "m", "v1")
    for sym, v in verdicts.items():
        append(conn, rid, session, Prediction(symbol=sym, made_at=made.isoformat(), model="m", prompt_version="v1",
                                              verdict=v, confidence=3, evidence=[]))


def _outcome(sym, start, end):
    return Outcome(sym, start.isoformat(), end.isoformat(), 100.0, 100 + RETURNS[sym], RETURNS[sym], 0.5,
                   RETURNS[sym] - 0.5)


CALLS = {"INFY": "BULLISH", "TCS": "BEARISH", "ITC": "NEUTRAL"}      # perfectly ordered: rank corr 1.0


def test_a_call_settles_only_after_its_window_closes(conn):
    _call(conn, CAL[0], CALLS)
    early = settle_pending(conn, CAL[:FORWARD], outcome_fn=_outcome)
    assert (early["settled"], early["waiting"]) == (0, 3)

    seen = []
    out = settle_pending(conn, CAL, outcome_fn=lambda s, a, b: seen.append((s, a, b)) or _outcome(s, a, b))
    assert out["settled"] == 3 and seen[0][1:] == (CAL[0], CAL[FORWARD])
    assert forward_end(CAL[0], CAL) == CAL[21]


def test_settlement_happens_once_and_is_append_only(conn):
    _call(conn, CAL[0], CALLS)
    settle_pending(conn, CAL, outcome_fn=_outcome)
    assert settle_pending(conn, CAL, outcome_fn=_outcome)["settled"] == 0
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE outcomes SET forward_return_pct = 99")


def test_a_failed_price_lookup_writes_nothing_and_is_retried(conn):
    _call(conn, CAL[0], {"INFY": "BULLISH"})

    def broken(*a):
        raise OutcomeError("no history")

    assert settle_pending(conn, CAL, outcome_fn=broken)["failed"]
    assert conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] == 0
    assert settle_pending(conn, CAL, outcome_fn=_outcome)["settled"] == 1


def test_analysis_follows_the_preregistered_plan(conn):
    first = next(d for d in CAL if d >= date(2026, 11, 18))           # warm-up ends
    i = CAL.index(first)
    _call(conn, CAL[2], CALLS)                                        # warm-up: excluded
    _call(conn, first, CALLS)                                         # primary #1
    _call(conn, CAL[i + 1], CALLS, late=True)                         # Thu call made Fri 11:30: excluded
    _call(conn, CAL[i + 2], CALLS)                                    # Fri call; Sat is before Mon's open: kept
    # CAL[i + 21] (primary #2) is missing: the machine was off.
    _call(conn, CAL[i + 42], CALLS)                                   # primary #3
    settle_pending(conn, CAL, outcome_fn=_outcome)

    rep = analyse(conn, CAL)["m · v1"]
    assert rep["excluded"] == {"warmup": 3, "after_next_open": 3}
    assert rep["primary"]["scheduled"] == 3 and rep["primary"]["dates"] == 2
    assert rep["primary"]["missing"] == [CAL[i + 21].isoformat()]
    assert rep["primary"]["mean_rank_correlation"] == 1.0
    assert rep["secondary"]["dates"] == 3


def test_newey_west_without_lags_is_the_plain_standard_error():
    xs = [0.1, -0.05, 0.02, 0.07, -0.01]
    plain = statistics.pstdev(xs) / len(xs) ** 0.5
    assert newey_west_se(xs, lag=0) == pytest.approx(plain)
    assert newey_west_se(xs, lag=4) != pytest.approx(plain)


def test_primary_schedule_steps_21_sessions():
    assert primary_schedule(CAL[3], CAL)[:3] == [CAL[3], CAL[24], CAL[45]]


@pytest.mark.skipif(not (ROOT / "cache" / "briefs").exists(), reason="local brief/grid cache not present")
def test_v1_replayed_through_m4_gives_the_published_result():
    from data.universe import UNIVERSE
    from evals.grid import load_grid
    from evals.report import build_model_rows, load_briefs

    frozen = json.loads((ROOT / "results-final.json").read_text(encoding="utf-8"))
    cutoffs = [date.fromisoformat(c) for c in
               json.loads((ROOT / "tests" / "golden" / "v1_cutoffs.json").read_text(encoding="utf-8"))]
    grid = load_grid([c.ticker for c in UNIVERSE], cutoffs)
    briefs = load_briefs(frozen["prompt_version"], frozen["model"], {c.isoformat() for c in cutoffs})
    rows, _ = build_model_rows(briefs, grid)

    # v1's cutoffs are already non-overlapping, so all 90 are primary.
    rep = summarise("v1", rows, [c.isoformat() for c in cutoffs])
    assert (rep["primary"]["dates"], rep["primary"]["mean_rank_correlation"],
            rep["primary"]["rank_correlation_t"]) == (90, -0.0153, -0.65)


def test_each_track_is_settled_and_analysed_on_its_own(conn):
    first = next(d for d in CAL if d >= date(2026, 11, 18))
    _call(conn, first, CALLS)                                                    # model "m"
    made = (datetime.combine(first, datetime.min.time(), timezone.utc) + timedelta(hours=13)).isoformat()
    rid = start_round(conn, first, made, "other@low", "v1")
    for sym, v in {"INFY": "BEARISH", "TCS": "BULLISH", "ITC": "NEUTRAL"}.items():   # perfectly wrong
        append(conn, rid, first, Prediction(symbol=sym, made_at=made, model="other@low", prompt_version="v1",
                                            verdict=v, confidence=3, evidence=[]))
    assert settle_pending(conn, CAL, outcome_fn=_outcome)["settled"] == 6
    rep = analyse(conn, CAL)
    assert rep["m · v1"]["primary"]["mean_rank_correlation"] == 1.0
    assert rep["other@low · v1"]["primary"]["mean_rank_correlation"] == -1.0
