"""SQLite store for live filings and the job log.

Plain `sqlite3`, no ORM: the rules that matter here — a UNIQUE key, and
triggers that make `filings` append-only — should be readable in the schema,
not inferred from model classes. The SQL is kept portable to Postgres for when
the store moves off this laptop.

The database is *primary data*, not a cache. The RSS feed cannot be backfilled,
so a deleted `live.sqlite` is a permanent hole in the record. It lives outside
`cache/` for that reason (and is gitignored by `*.sqlite`).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from live.filings import FilingRecord

DEFAULT_DB = Path(__file__).resolve().parent.parent / "store" / "live.sqlite"

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS job_runs (
    id              INTEGER PRIMARY KEY,
    job_type        TEXT NOT NULL,
    provider        TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    state           TEXT NOT NULL CHECK (state IN ('running', 'ok', 'failed')),
    attempts        INTEGER NOT NULL DEFAULT 0,
    fetched         INTEGER NOT NULL DEFAULT 0,
    inserted        INTEGER NOT NULL DEFAULT 0,
    duplicates      INTEGER NOT NULL DEFAULT 0,
    rejected        INTEGER NOT NULL DEFAULT 0,
    feed_oldest     TEXT,
    feed_newest     TEXT,
    coverage_gap_s  INTEGER,
    error           TEXT,
    rejects_json    TEXT
);

CREATE TABLE IF NOT EXISTS filings (
    id              INTEGER PRIMARY KEY,
    provider        TEXT NOT NULL,
    provider_ref    TEXT NOT NULL,
    company         TEXT NOT NULL,
    subject         TEXT NOT NULL,
    url             TEXT,
    published_at    TEXT NOT NULL,
    published_raw   TEXT NOT NULL,
    first_seen_at   TEXT NOT NULL,
    job_id          INTEGER NOT NULL REFERENCES job_runs (id),
    UNIQUE (provider, provider_ref)
);

CREATE INDEX IF NOT EXISTS ix_filings_company_published
    ON filings (company, published_at);

CREATE TRIGGER IF NOT EXISTS filings_no_update BEFORE UPDATE ON filings
BEGIN SELECT RAISE(ABORT, 'filings are append-only'); END;

CREATE TRIGGER IF NOT EXISTS filings_no_delete BEFORE DELETE ON filings
BEGIN SELECT RAISE(ABORT, 'filings are append-only'); END;
"""


def connect(path: Path | str = DEFAULT_DB, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.executescript(_SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
            conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
            conn.commit()
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def start_job(conn: sqlite3.Connection, job_type: str, provider: str, started_at: str) -> int:
    cur = conn.execute(
        "INSERT INTO job_runs (job_type, provider, started_at, state) VALUES (?, ?, ?, 'running')",
        (job_type, provider, started_at),
    )
    conn.commit()
    return cur.lastrowid


def finish_job(conn: sqlite3.Connection, job_id: int, **fields) -> None:
    rejects = fields.pop("rejects", None)
    if rejects is not None:
        fields["rejects_json"] = json.dumps([r.__dict__ for r in rejects], ensure_ascii=False)
    cols = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE job_runs SET {cols} WHERE id = ?", (*fields.values(), job_id))
    conn.commit()


def insert_filings(
    conn: sqlite3.Connection, records: list[FilingRecord], first_seen_at: str, job_id: int,
) -> int:
    """All-or-nothing. Returns how many were new; the rest were already stored
    and are left exactly as they were, first_seen_at included."""
    inserted = 0
    with conn:
        for r in records:
            cur = conn.execute(
                """INSERT INTO filings (provider, provider_ref, company, subject, url,
                                        published_at, published_raw, first_seen_at, job_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (provider, provider_ref) DO NOTHING""",
                (r.provider, r.provider_ref, r.company, r.subject, r.url,
                 r.published_at, r.published_raw, first_seen_at, job_id),
            )
            inserted += cur.rowcount
    return inserted


def newest_published(conn: sqlite3.Connection, provider: str) -> str | None:
    return conn.execute(
        "SELECT MAX(published_at) FROM filings WHERE provider = ?", (provider,)
    ).fetchone()[0]


def last_ok_job(conn: sqlite3.Connection, job_type: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM job_runs WHERE job_type = ? AND state = 'ok' ORDER BY id DESC LIMIT 1",
        (job_type,),
    ).fetchone()
