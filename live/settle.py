"""M4: settle the live ledger against what happened, then analyse it as
pre-registered (README, "Analysis plan, fixed 2026-10-29").

Settlement uses v1's answer key unchanged: `evals.outcomes.get_outcome`, the
adjusted total return from the close a call read to the close of the 21st NSE
session after it. Scoring uses v1's scorer unchanged: `evals.scoring.score_l3`
over (VERDICT_SCORE, forward return, verdict) rows, one observation per
session. A test replays v1's cached briefs through `summarise` and must get
v1's published −0.0153 / t −0.65.

A call is settled once, into the append-only `outcomes` table, when its
forward window has fully elapsed. Outcomes are derived from public prices, so
they are not hash-chained: anyone can recompute them. A failed lookup writes
nothing and is retried on the next run.

The analysis is computed on read:
- excluded: warm-up sessions, and any call made after the next session opened
- tracks: one per (model, prompt version), never pooled (plan item 6)
- primary: per-session rank correlation over non-overlapping sessions, the
  first non-warm-up session and every 21st NSE session after it. A scheduled
  session the ledger doesn't have (machine off) is reported as missing, never
  replaced by a neighbour, because picking the substitute would be a choice
  made after seeing data.
- secondary: every settled session, with a Newey-West standard error at lag
  20, since consecutive sessions' 21-day windows overlap by up to 20 days.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timezone

from data.windows import FORWARD_TRADING_DAYS
from evals.outcomes import OutcomeError, get_outcome
from evals.scoring import VERDICT_SCORE, score_l3
from live.filings import IST

FORWARD = FORWARD_TRADING_DAYS                       # 21, v1's horizon
NSE_OPEN = time(9, 15)
NW_LAG = FORWARD - 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    prediction_seq        INTEGER PRIMARY KEY REFERENCES predictions (seq),
    session_date          TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    forward_end           TEXT NOT NULL,
    start_price           REAL NOT NULL,
    end_price             REAL NOT NULL,
    forward_return_pct    REAL NOT NULL,
    benchmark_return_pct  REAL,
    excess_return_pct     REAL,
    settled_at            TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS outcomes_no_update BEFORE UPDATE ON outcomes
BEGIN SELECT RAISE(ABORT, 'outcomes are append-only'); END;
CREATE TRIGGER IF NOT EXISTS outcomes_no_delete BEFORE DELETE ON outcomes
BEGIN SELECT RAISE(ABORT, 'outcomes are append-only'); END;
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


@dataclass(frozen=True)
class Call:
    seq: int
    session: date
    symbol: str
    verdict: str
    made_at: str
    warmup: bool
    model: str
    prompt_version: str


def counting_calls(conn) -> list[Call]:
    """The call that counts per (session, stock): the latest attempt, if it
    produced a verdict. A failed latest attempt means no call that session."""
    rows = conn.execute(
        "SELECT p.seq, p.session_date, p.symbol, p.verdict, p.made_at, p.warmup, p.model, p.prompt_version "
        "FROM predictions p JOIN (SELECT session_date, symbol, MAX(attempt) AS a FROM predictions "
        "GROUP BY session_date, symbol) m ON p.session_date = m.session_date AND p.symbol = m.symbol "
        "AND p.attempt = m.a WHERE p.verdict IS NOT NULL ORDER BY p.session_date, p.symbol").fetchall()
    return [Call(r[0], date.fromisoformat(r[1]), r[2], r[3], r[4], bool(r[5]), r[6], r[7]) for r in rows]


def forward_end(session: date, calendar: list[date]) -> date | None:
    """The 21st NSE session after `session`, if it has already closed."""
    try:
        i = calendar.index(session)
    except ValueError:
        return None
    return calendar[i + FORWARD] if i + FORWARD < len(calendar) else None


# ---------------------------------------------------------------------------
# settlement
# ---------------------------------------------------------------------------

def settle_pending(conn, calendar: list[date], outcome_fn=None, now=None) -> dict:
    """Settle every counting call whose window has elapsed and isn't settled yet.

    `calendar` is NSE sessions up to the latest posted close (data.windows.
    nse_trading_days), so a forward end in it has a real closing price.
    """
    outcome_fn = outcome_fn or (lambda sym, start, end: get_outcome(f"{sym}.NS", start, end))
    now = now or (lambda: datetime.now(timezone.utc))
    ensure_schema(conn)
    done = {r[0] for r in conn.execute("SELECT prediction_seq FROM outcomes")}
    settled, failed, waiting = 0, [], 0
    for c in counting_calls(conn):
        if c.seq in done:
            continue
        end = forward_end(c.session, calendar)
        if end is None:
            waiting += 1
            continue
        try:
            o = outcome_fn(c.symbol, c.session, end)
        except OutcomeError as exc:
            failed.append(f"{c.symbol} {c.session}: {exc}")
            continue
        with conn:
            conn.execute(
                "INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (c.seq, c.session.isoformat(), c.symbol, end.isoformat(), o.start_price, o.end_price,
                 o.forward_return_pct, o.benchmark_return_pct, o.excess_return_pct,
                 now().isoformat(timespec="seconds")))
        settled += 1
    return {"settled": settled, "failed": failed, "waiting": waiting}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def newey_west_se(values: list[float], lag: int = NW_LAG) -> float | None:
    """Standard error of the mean with Bartlett-weighted autocovariances."""
    n = len(values)
    if n < 3:
        return None
    m = sum(values) / n
    d = [v - m for v in values]
    var = sum(x * x for x in d) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * sum(d[i] * d[i - k] for i in range(k, n)) / n
    return (var / n) ** 0.5 if var > 0 else None


def primary_schedule(first: date, calendar: list[date]) -> list[date]:
    """`first`, then every 21st NSE session after it."""
    return calendar[calendar.index(first)::FORWARD] if first in calendar else []


def summarise(label: str, by_session: dict[str, list[tuple[float, float, str]]],
              primary: list[str]) -> dict:
    """Primary and secondary results for one track. `by_session` has v1's row
    shape, so v1's own rows can be replayed through this function."""
    every = score_l3(label, by_session)
    prim = score_l3(label, {d: by_session[d] for d in primary if d in by_session})
    corr = [d.rank_correlation for d in every.dates if d.rank_correlation is not None]
    se = newey_west_se(corr)
    mean = sum(corr) / len(corr) if corr else None
    return {
        "primary": prim.summary() | {"scheduled": len(primary),
                                     "missing": sorted(d for d in primary if d not in by_session)},
        "secondary": every.summary() | {
            "newey_west_se": None if se is None else round(se, 4),
            "newey_west_t": None if se is None or mean is None else round(mean / se, 2)},
    }


def excluded_reason(call: Call, calendar: list[date]) -> str | None:
    if call.warmup:
        return "warmup"
    i = calendar.index(call.session) if call.session in calendar else None
    if i is None or i + 1 >= len(calendar):
        return "unknown_session"
    next_open = datetime.combine(calendar[i + 1], NSE_OPEN, tzinfo=IST)
    if datetime.fromisoformat(call.made_at) >= next_open:
        return "after_next_open"
    return None


def analyse(conn, calendar: list[date]) -> dict:
    ensure_schema(conn)
    outcomes = {r[0]: r[1] for r in conn.execute("SELECT prediction_seq, forward_return_pct FROM outcomes")}
    tracks: dict[tuple[str, str], dict] = {}
    for c in counting_calls(conn):
        t = tracks.setdefault((c.model, c.prompt_version),
                              {"rows": {}, "excluded": {}, "unsettled": 0, "first": None})
        why = excluded_reason(c, calendar)
        if why:
            t["excluded"][why] = t["excluded"].get(why, 0) + 1
            continue
        # The schedule anchors on the first eligible session, settled or not,
        # so a failed price lookup can't shift which sessions are primary.
        t["first"] = min(t["first"] or c.session, c.session)
        if c.seq not in outcomes:
            t["unsettled"] += 1
            continue
        t["rows"].setdefault(c.session.isoformat(), []).append(
            (VERDICT_SCORE[c.verdict], outcomes[c.seq], c.verdict))

    report = {}
    for (model, prompt), t in tracks.items():
        last = max(t["rows"]) if t["rows"] else None
        primary = ([d.isoformat() for d in primary_schedule(t["first"], calendar) if d.isoformat() <= last]
                   if last else [])
        report[f"{model} · {prompt}"] = summarise(model, t["rows"], primary) | {
            "excluded": t["excluded"], "unsettled": t["unsettled"]}
    return report
