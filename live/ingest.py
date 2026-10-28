"""One poll of the NSE announcements feed: fetch, parse, store, log.

Every run writes a `job_runs` row, success or not, so a dead poller shows up as
a stale "last ok" rather than as a suspiciously quiet market.

**Coverage gaps.** The feed is a rolling window of a few hours. If the oldest
item in this poll is newer than the newest filing already stored, the window
turned over completely since the last good poll and filings in between may
have been lost. That is recorded as `coverage_gap_s`, not hidden — a laptop
that slept through an afternoon has a hole in its record, and the record says
so. A quiet market cannot fake a gap: with nothing new published, the feed's
oldest item is still one we have.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Callable

import requests

from live import store
from live.filings import PROVIDER_NSE_RSS, Rejected, parse_nse_rss

FEED_URL = "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"
# Honest, but without a URL: measured 2026-10-28, the same request with a
# "+https://..." contact link in the UA hangs until timeout, without one it
# answers in 0.1s. A browser UA is not needed and is not used.
USER_AGENT = "stock-judgment-bench/1.0 (research)"
JOB_TYPE = "ingest_filings"

MAX_ATTEMPTS = 3
BACKOFF_S = (2.0, 4.0)
# Clocks disagree a little; a filing "published" more than this far past our
# own observation time is malformed, not early.
FUTURE_TOLERANCE = timedelta(minutes=10)


class TransientFetchError(RuntimeError):
    """Worth retrying: timeouts, connection resets, 429, 5xx."""


class PermanentFetchError(RuntimeError):
    """Not worth retrying: 403 and other refusals. Retrying a block is rude
    and does not work."""


def fetch_feed(url: str = FEED_URL, timeout: float = 20.0) -> bytes:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:   # includes a download cut off mid-body
        raise TransientFetchError(f"{type(exc).__name__}: {str(exc)[:160]}") from exc
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientFetchError(f"HTTP {resp.status_code}")
    if not resp.ok:
        raise PermanentFetchError(f"HTTP {resp.status_code}: {resp.text[:160]!r}")
    return resp.content


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def run_once(
    db_path=store.DEFAULT_DB,
    fetch: Callable[[], bytes] = fetch_feed,
    now: Callable[[], datetime] = _utcnow,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Returns the finished job_runs row as a dict."""
    conn = store.connect(db_path)
    try:
        job_id = store.start_job(conn, JOB_TYPE, PROVIDER_NSE_RSS, _iso(now()))
        try:
            return _poll(conn, job_id, fetch, now, sleep)
        except Exception as exc:
            # Never leave a run looking like it is still going.
            store.finish_job(conn, job_id, state="failed",
                             error=f"crash: {type(exc).__name__}: {str(exc)[:160]}",
                             finished_at=_iso(now()))
            raise
    finally:
        conn.close()


def _poll(conn, job_id: int, fetch, now, sleep) -> dict:
    previous_newest = store.newest_published(conn, PROVIDER_NSE_RSS)

    # Parsing sits inside the retry loop: the live feed has served malformed XML
    # once and then a clean copy seconds later (2026-10-28), so a bad body is
    # treated like a timeout. If it stays bad, the run fails — never "empty".
    parsed, attempts, error = None, 0, None
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            body = fetch()
            observed = now()
            parsed = parse_nse_rss(body)
            break
        except PermanentFetchError as exc:
            error = f"permanent: {exc}"
            break
        except (TransientFetchError, ET.ParseError) as exc:
            kind = "parse" if isinstance(exc, ET.ParseError) else "transient"
            error = f"{kind}: {exc}"
            if attempts < MAX_ATTEMPTS:
                sleep(BACKOFF_S[min(attempts - 1, len(BACKOFF_S) - 1)])

    if parsed is None:
        store.finish_job(conn, job_id, state="failed", attempts=attempts,
                         error=error, finished_at=_iso(now()))
        return _row(conn, job_id)
    records, rejects = parsed

    limit = _iso(observed + FUTURE_TOLERANCE)
    future = [r for r in records if r.published_at > limit]
    records = [r for r in records if r.published_at <= limit]
    rejects += [Rejected("future-dated", r.company, r.published_raw) for r in future]

    inserted = store.insert_filings(conn, records, _iso(observed), job_id)

    oldest = min((r.published_at for r in records), default=None)
    newest = max((r.published_at for r in records), default=None)
    gap = None
    if previous_newest and oldest and oldest > previous_newest:
        gap = int((datetime.fromisoformat(oldest)
                   - datetime.fromisoformat(previous_newest)).total_seconds())

    store.finish_job(
        conn, job_id, state="ok", attempts=attempts,
        fetched=len(records) + len(rejects), inserted=inserted,
        duplicates=len(records) - inserted, rejected=len(rejects),
        feed_oldest=oldest, feed_newest=newest, coverage_gap_s=gap,
        rejects=rejects, error=error, finished_at=_iso(now()),
    )
    return _row(conn, job_id)


def _row(conn, job_id: int) -> dict:
    return dict(conn.execute("SELECT * FROM job_runs WHERE id = ?", (job_id,)).fetchone())
