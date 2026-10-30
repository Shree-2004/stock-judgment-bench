"""M3: the forward prediction ledger.

Each trading day after the close, one prediction per stock is appended for
the 41-name universe, using the v1 prompt and model and filings this system
had *seen* by the moment of prediction. The outcome won't exist for another
21 sessions. That makes the ledger the memorisation control v1 could not
have: no model can have trained on a future that hadn't happened.

**What makes it a ledger, not a log.**
- `predictions` is append-only (SQLite triggers abort UPDATE and DELETE).
- Every row carries `prev_hash` and `entry_hash = sha256(prev_hash + payload)`,
  so editing any past row breaks every hash after it. `verify_chain` recomputes
  the whole chain.
- After each round the head hash is appended to `ledger/chain.txt` and committed
  to git. Once pushed, rewriting history means rewriting a public repository.
  The SQLite file alone proves nothing; the commits are the witness.

**Rules fixed before the first outcome (2026-10-29):**
- Session: the latest ^NSEI daily row. If that isn't today in IST, there was
  no session (holiday, or prices not posted yet), and the round does not run.
  A prediction is never silently based on yesterday's close.
- Evidence: filings for the stock with `first_seen_at <= made_at` (when this
  system saw them, not when the exchange stamped them), published in the
  21 days before the session. Routine compliance and fund NAVs are dropped,
  and the newest 5 distinct headlines are kept, which is v1's selection rule.
- Warm-up: the filing store began 2026-10-28, so a session before
  `WARMUP_END` cannot have 21 days of evidence. Those rows are flagged
  `warmup = 1` and excluded from the primary analysis.
- A failure is a row with `error` set, never a missing row. A retry for the
  same session and stock is a new row with `attempt + 1`. M4 scores the
  latest successful attempt made before the next session opened.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from data.news_data import DisclosureBundle, Item, clean_text, normalise_headline
from live.classify import CLASSIFIER_VERSION, classify, split_subject
from live.filings import IST, NSE_NAMES, _norm

ROOT = Path(__file__).resolve().parent.parent
CHAIN_FILE = ROOT / "ledger" / "chain.txt"

GENESIS = "0" * 64
EVIDENCE_START = date(2026, 10, 28)                 # first poll of the filing store
LOOKBACK_DAYS = 21                                   # v1's evidence window
WARMUP_END = EVIDENCE_START + timedelta(days=LOOKBACK_DAYS)   # 2026-11-18
EVIDENCE_LIMIT = 5                                   # v1's DEFAULT_LIMIT
DROP_CATEGORIES = frozenset({"routine_compliance", "fund_nav"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_rounds (
    id                  INTEGER PRIMARY KEY,
    session_date        TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    state               TEXT NOT NULL CHECK (state IN ('running', 'ok', 'partial', 'failed')),
    model               TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    classifier_version  TEXT NOT NULL,
    n_ok                INTEGER NOT NULL DEFAULT 0,
    n_failed            INTEGER NOT NULL DEFAULT 0,
    head_hash           TEXT,
    error               TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    seq                 INTEGER PRIMARY KEY,
    round_id            INTEGER NOT NULL REFERENCES ledger_rounds (id),
    session_date        TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    attempt             INTEGER NOT NULL,
    made_at             TEXT NOT NULL,
    model               TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    classifier_version  TEXT NOT NULL,
    warmup              INTEGER NOT NULL,
    verdict             TEXT,
    confidence          INTEGER,
    price               REAL,
    snapshot_json       TEXT,
    evidence_json       TEXT NOT NULL,
    brief_json          TEXT,
    raw_text            TEXT,
    error               TEXT,
    prev_hash           TEXT NOT NULL,
    entry_hash          TEXT NOT NULL UNIQUE,
    UNIQUE (session_date, symbol, model, prompt_version, attempt)
);

CREATE TRIGGER IF NOT EXISTS predictions_no_update BEFORE UPDATE ON predictions
BEGIN SELECT RAISE(ABORT, 'predictions are append-only'); END;

CREATE TRIGGER IF NOT EXISTS predictions_no_delete BEFORE DELETE ON predictions
BEGIN SELECT RAISE(ABORT, 'predictions are append-only'); END;
"""
# Until 2026-10-30 a row was unique per (session, symbol, attempt), which
# allowed one model only. The second track needs the model in the key.
_OLD_UNIQUE = "UNIQUE (session_date, symbol, attempt)"

# The columns the hash covers, in a fixed order. Everything except seq (a
# storage detail) and entry_hash (the output).
HASHED = ("round_id", "session_date", "symbol", "attempt", "made_at", "model", "prompt_version",
          "classifier_version", "warmup", "verdict", "confidence", "price", "snapshot_json",
          "evidence_json", "brief_json", "raw_text", "error", "prev_hash")


def ensure_schema(conn: sqlite3.Connection) -> None:
    old = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'predictions'").fetchone()
    if old and _OLD_UNIQUE in old[0]:
        migrate_track_key(conn)
    conn.executescript(_SCHEMA)
    conn.commit()


def migrate_track_key(conn: sqlite3.Connection) -> None:
    """Rebuild `predictions` with the model in its unique key, copying every
    row verbatim (seq included).

    The hash covers row contents, not storage, so the chain must verify and
    the head must be unchanged afterwards. The head is what chain.txt and the
    pushed history witness. If either check fails, nothing is committed.
    """
    before_ok, before_n, _ = verify_chain(conn)
    before_head = head_hash(conn)
    if not before_ok:
        raise RuntimeError("refusing to migrate a ledger whose chain does not verify")
    new_table = _SCHEMA[_SCHEMA.index("CREATE TABLE IF NOT EXISTS predictions"):_SCHEMA.index("CREATE TRIGGER")]
    row_factory = conn.row_factory
    conn.execute("PRAGMA foreign_keys = OFF")          # must be set outside a transaction
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TRIGGER IF EXISTS predictions_no_update")
        conn.execute("DROP TRIGGER IF EXISTS predictions_no_delete")
        conn.execute(new_table.replace("IF NOT EXISTS predictions", "predictions_new"))
        conn.execute("INSERT INTO predictions_new SELECT * FROM predictions ORDER BY seq")
        conn.execute("DROP TABLE predictions")
        conn.execute("ALTER TABLE predictions_new RENAME TO predictions")
        ok, n, _ = verify_chain(conn)
        if not (ok and n == before_n and head_hash(conn) == before_head):
            raise RuntimeError(f"migration check failed: ok={ok} rows {n}/{before_n}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = row_factory


def entry_hash(row: dict) -> str:
    payload = json.dumps([row[k] for k in HASHED], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"{row['prev_hash']}\n{payload}".encode("utf-8")).hexdigest()


def head_hash(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT entry_hash FROM predictions ORDER BY seq DESC LIMIT 1").fetchone()
    return row[0] if row else GENESIS


# ---------------------------------------------------------------------------
# rounds and rows
# ---------------------------------------------------------------------------

def start_round(conn, session: date, started_at: str, model: str, prompt_version: str) -> int:
    cur = conn.execute(
        "INSERT INTO ledger_rounds (session_date, started_at, state, model, prompt_version, "
        "classifier_version) VALUES (?, ?, 'running', ?, ?, ?)",
        (session.isoformat(), started_at, model, prompt_version, CLASSIFIER_VERSION))
    conn.commit()
    return cur.lastrowid


def finish_round(conn, round_id: int, **fields) -> None:
    fields["head_hash"] = head_hash(conn)
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE ledger_rounds SET {cols} WHERE id = ?", (*fields.values(), round_id))
    conn.commit()


def done_symbols(conn, session: date, model: str, prompt_version: str) -> set[str]:
    """Stocks that already have a successful prediction for this session, on this track."""
    return {r[0] for r in conn.execute(
        "SELECT symbol FROM predictions WHERE session_date = ? AND model = ? AND prompt_version = ? "
        "AND error IS NULL", (session.isoformat(), model, prompt_version))}


def next_attempt(conn, session: date, symbol: str, model: str, prompt_version: str) -> int:
    row = conn.execute("SELECT MAX(attempt) FROM predictions WHERE session_date = ? AND symbol = ? "
                       "AND model = ? AND prompt_version = ?",
                       (session.isoformat(), symbol, model, prompt_version)).fetchone()
    return (row[0] or 0) + 1


@dataclass
class Prediction:
    symbol: str
    made_at: str
    model: str
    prompt_version: str
    evidence: list[dict]
    verdict: str | None = None
    confidence: int | None = None
    price: float | None = None
    snapshot: dict | None = None
    brief: dict | None = None
    raw_text: str | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)


def append(conn, round_id: int, session: date, p: Prediction) -> dict:
    """Chain one prediction onto the ledger. Atomic: the head is read and the
    row written in one transaction, so two writers cannot fork the chain."""
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        row = {
            "round_id": round_id,
            "session_date": session.isoformat(),
            "symbol": p.symbol,
            "attempt": next_attempt(conn, session, p.symbol, p.model, p.prompt_version),
            "made_at": p.made_at,
            "model": p.model,
            "prompt_version": p.prompt_version,
            "classifier_version": CLASSIFIER_VERSION,
            "warmup": int(session < WARMUP_END),
            "verdict": p.verdict,
            "confidence": p.confidence,
            "price": p.price,
            "snapshot_json": _dump(p.snapshot),
            "evidence_json": _dump(p.evidence),
            "brief_json": _dump(p.brief),
            "raw_text": p.raw_text,
            "error": p.error,
            "prev_hash": head_hash(conn),
        }
        row["entry_hash"] = entry_hash(row)
        cols = ", ".join(row)
        conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({', '.join('?' * len(row))})",
                     tuple(row.values()))
    return row


def _dump(obj) -> str | None:
    return None if obj is None else json.dumps(obj, ensure_ascii=False, sort_keys=True)


def verify_chain(conn) -> tuple[bool, int, str | None]:
    """Recompute every hash in order. Returns (ok, rows checked, first bad seq or None)."""
    prev = GENESIS
    n = 0
    conn.row_factory = sqlite3.Row
    for r in conn.execute("SELECT * FROM predictions ORDER BY seq"):
        row = dict(r)
        n += 1
        if row["prev_hash"] != prev or entry_hash(row) != row["entry_hash"]:
            return False, n, str(row["seq"])
        prev = row["entry_hash"]
    return True, n, None


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

def evidence_for(conn, symbol: str, session: date, made_at: str) -> tuple[DisclosureBundle, list[dict]]:
    """Filings this system had seen by `made_at`, for one stock, v1's selection rule.

    Returns the bundle the prompt renders and the provenance recorded in the
    ledger (one dict per filing shown to the model, in prompt order)."""
    names = {_norm(NSE_NAMES[symbol])}
    since = datetime.combine(session - timedelta(days=LOOKBACK_DAYS), datetime.min.time(), IST)
    rows = [r for r in conn.execute(
        "SELECT provider_ref, company, subject, url, published_at, first_seen_at FROM filings "
        "WHERE first_seen_at <= ? AND published_at >= ? ORDER BY published_at DESC",
        (made_at, since.astimezone(timezone.utc).isoformat()))
        if _norm(r[1]) in names or _alias_hit(r[1], symbol)]

    items, kept, seen = [], [], set()
    dropped_noise = dropped_dup = 0
    for ref, _company, subject, url, published_at, first_seen_at in rows:
        c = classify(subject, url)
        if c.category in DROP_CATEGORIES:
            dropped_noise += 1
            continue
        text, tag = split_subject(subject)
        headline = clean_text(text) or tag
        key = normalise_headline(headline)
        if not key or key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        if len(items) < EVIDENCE_LIMIT:
            filed_ist = datetime.fromisoformat(published_at).astimezone(IST).isoformat(timespec="seconds")
            items.append(Item(headline=headline, category=c.category, subcategory=tag, filed_at=filed_ist))
            kept.append({"provider_ref": ref, "published_at": published_at, "first_seen_at": first_seen_at,
                         "category": c.category, "rule": c.rule, "headline": headline})

    bundle = DisclosureBundle(symbol=symbol, cutoff=session.isoformat(), status="ok", items=items,
                              considered=len(rows), dropped_as_noise=dropped_noise,
                              dropped_as_duplicate=dropped_dup)
    return bundle, kept


def _alias_hit(company: str, symbol: str) -> bool:
    from live.filings import NSE_ALIASES
    return any(s == symbol and _norm(n) == _norm(company) for n, s in NSE_ALIASES.items())


# ---------------------------------------------------------------------------
# the git witness
# ---------------------------------------------------------------------------

def append_chain_line(path: Path, session: date, round_state: str, total_rows: int, head: str,
                      written_at: str) -> str:
    """One line per round: what git will witness. Plain text so a diff is readable."""
    line = f"{session.isoformat()}  {round_state:<7}  rows={total_rows:<6}  head={head}  at={written_at}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)
    return line
