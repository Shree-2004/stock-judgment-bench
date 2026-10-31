"""Quarterly results from NSE's Integrated Filing (Financials), stored as filed.

Step 1 of the fundamentals track (README plan): this module only *collects*.
Nothing here is shown to a model yet, and the running tracks do not read it.

**Source.** Since the quarter ended March 2025, SEBI's Integrated Filing
replaced the old results format, and NSE's per-company listing at
`/api/integrated-filing-results` stopped being mirrored in the older
`/api/corporates-financial-results` (it ends at Feb 2025 — checked 2026-10-31).
Each listed filing links an XBRL file: structured numbers, no PDF parsing.
Measured 2026-10-31, both answer a plain request with the same honest
User-Agent the RSS poller uses.

**What is stored.** The raw XBRL (zlib-compressed, with its sha256) and every
plain, non-dimensional fact in it, grouped by context ("OneD" = the quarter,
"FourD" = year to date, ...). Banks and NBFCs file a different taxonomy —
`InterestEarned` and `ProfitLossForThePeriod` rather than
`RevenueFromOperations` and `ProfitLossForPeriod` — so choosing which fact
means "revenue" is left to step 2. Storing all of them means that choice can
be made, and changed, without re-fetching anything.

**Point in time.** `published_at` is NSE's broadcast time (a revision's own
`revised_Date`; a revision has no broadcast time). A filing first stored by a
backfill has a `first_seen_at` long after it was published; NSE's timestamp is
what says when it was public. A revision is a new row: the original is never
replaced, so the record keeps what was public on each night.

**Failures.** A listing or XBRL download that fails is a reject on the job,
never an empty result, and since nothing was stored it is retried next run.
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable
from urllib.parse import quote

import requests

from live import store
from live.filings import IST, Rejected
from live.ingest import (BACKOFF_S, FUTURE_TOLERANCE, MAX_ATTEMPTS, USER_AGENT, PermanentFetchError,
                         TransientFetchError, _iso, _utcnow)

PROVIDER = "nse_integrated_filing"
JOB_TYPE = "ingest_results"
LISTING_URL = "https://www.nseindia.com/api/integrated-filing-results?index=equities&symbol={symbol}&period=Quarterly"
FINANCIALS = "Integrated Filing- Financials"      # the other type, "- Governance", has no numbers
STATEMENTS = {"Consolidated", "Standalone"}
MAX_PAGES = 5           # 20 rows a page; page 1 alone reaches back about six quarters
PAUSE_S = 1.0           # between requests: one polite client, not a crawler
MAX_TEXT = 200          # longer text facts (notes, remarks) stay in the raw XBRL only

_XBRLI = "{http://www.xbrl.org/2003/instance}"


@dataclass(frozen=True)
class Listing:
    symbol: str
    xbrl_url: str
    seq_id: str | None
    kind: str               # "Original" or "Revision"
    statement: str          # "Consolidated" or "Standalone"
    quarter_end: str        # ISO date
    published_at: str       # UTC ISO: NSE broadcast time, or the revision's own time
    published_raw: str


def _nse_time(raw: str) -> str:
    """'28-Oct-2026 21:08:16' (IST; revisions write the month in capitals) -> UTC ISO."""
    dt = datetime.strptime(raw.strip(), "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
    return _iso(dt.astimezone(timezone.utc))


def parse_listing(payload: dict, symbol: str) -> tuple[list[Listing], list[Rejected]]:
    listings, rejects = [], []
    for row in payload.get("data") or []:
        if row.get("type") != FINANCIALS:
            continue
        if row.get("symbol") != symbol:
            # A wrong-company row must never be stored under ours.
            rejects.append(Rejected(f"symbol mismatch: {row.get('symbol')!r}", symbol, None))
            continue
        raw_time = row.get("broadcast_Date") or (row.get("revised_Date") if row.get("type_Sub") == "Revision" else None)
        url, statement = row.get("xbrl"), row.get("consolidated")
        if not url or statement not in STATEMENTS or not raw_time or not row.get("qe_Date"):
            rejects.append(Rejected("incomplete listing row", symbol, raw_time))
            continue
        try:
            published = _nse_time(raw_time)
            quarter_end = datetime.strptime(row["qe_Date"], "%d-%b-%Y").date().isoformat()
        except ValueError:
            rejects.append(Rejected("unparseable date", symbol, raw_time))
            continue
        listings.append(Listing(symbol, url, row.get("seq_Id"), row.get("type_Sub") or "Original",
                                statement, quarter_end, published, raw_time))
    return listings, rejects


def parse_xbrl(content: bytes) -> dict:
    """{"periods": {ctx: {start, end} | {instant}}, "facts": {ctx: {name: value}}},
    plain contexts only. Values stay strings, exactly as filed."""
    root = ET.fromstring(content)
    periods = {}
    for ctx in root.iter(f"{_XBRLI}context"):
        if ctx.find(f".//{_XBRLI}scenario") is not None or ctx.find(f".//{_XBRLI}segment") is not None:
            continue                                  # a breakdown (one expense line, one segment)
        p = ctx.find(f"{_XBRLI}period")
        if p is None:
            continue
        instant = p.findtext(f"{_XBRLI}instant")
        periods[ctx.get("id")] = ({"instant": instant} if instant else
                                  {"start": p.findtext(f"{_XBRLI}startDate"), "end": p.findtext(f"{_XBRLI}endDate")})
    facts: dict[str, dict[str, str]] = {}
    for el in root:
        ctx = el.get("contextRef")
        value = (el.text or "").strip()
        if ctx not in periods or not value or len(value) > MAX_TEXT:
            continue
        facts.setdefault(ctx, {})[el.tag.rsplit("}", 1)[-1]] = value
    return {"periods": periods, "facts": facts}


def quarter_facts(parsed: dict) -> tuple[dict, dict]:
    """The facts for the three-month period the filing reports, and its dates."""
    best = None
    for ctx, p in parsed["periods"].items():
        if "start" not in p or ctx not in parsed["facts"]:
            continue
        days = (date.fromisoformat(p["end"]) - date.fromisoformat(p["start"])).days
        if 80 <= days <= 100 and (best is None or p["end"] > parsed["periods"][best]["end"]):
            best = ctx
    return (parsed["facts"][best], parsed["periods"][best]) if best else ({}, {})


def fetch(url: str, timeout: float = 30.0) -> bytes:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        raise TransientFetchError(f"{type(exc).__name__}: {str(exc)[:160]}") from exc
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientFetchError(f"HTTP {resp.status_code}")
    if not resp.ok:
        raise PermanentFetchError(f"HTTP {resp.status_code}: {resp.text[:160]!r}")
    return resp.content


def _retrying(get, sleep):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return get()
        except (TransientFetchError, ET.ParseError, json.JSONDecodeError):
            if attempt == MAX_ATTEMPTS:
                raise
            sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])


def list_filings(symbol: str, fetch=fetch, sleep=time.sleep) -> tuple[list[Listing], list[Rejected]]:
    base = LISTING_URL.format(symbol=quote(symbol, safe=""))
    listings, rejects, rows = [], [], 0
    for page in range(1, MAX_PAGES + 1):
        payload = _retrying(lambda: json.loads(fetch(base if page == 1 else f"{base}&page={page}")), sleep)
        got, bad = parse_listing(payload, symbol)
        listings += [x for x in got if x.xbrl_url not in {y.xbrl_url for y in listings}]   # pages may overlap
        rejects += bad
        rows += len(payload.get("data") or [])
        if not payload.get("data") or rows >= int(payload.get("totalCount") or 0):
            break
        sleep(PAUSE_S)
    return listings, rejects


def run_once(
    symbols: list[str],
    db_path=store.DEFAULT_DB,
    fetch: Callable[[str], bytes] = fetch,
    now: Callable[[], datetime] = _utcnow,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """One pass over `symbols`. Returns the finished job_runs row as a dict."""
    conn = store.connect(db_path)
    try:
        job_id = store.start_job(conn, JOB_TYPE, PROVIDER, _iso(now()))
        try:
            return _collect(conn, job_id, symbols, fetch, now, sleep)
        except Exception as exc:
            store.finish_job(conn, job_id, state="failed",
                             error=f"crash: {type(exc).__name__}: {str(exc)[:160]}", finished_at=_iso(now()))
            raise
    finally:
        conn.close()


def _collect(conn, job_id, symbols, fetch, now, sleep) -> dict:
    known = {r[0] for r in conn.execute("SELECT provider_ref FROM results")}
    rejects: list[Rejected] = []
    listed = inserted = dups = dead = 0
    for i, symbol in enumerate(symbols):
        if i:
            sleep(PAUSE_S)
        try:
            listings, bad = list_filings(symbol, fetch, sleep)
        except Exception as exc:
            rejects.append(Rejected(f"listing failed: {type(exc).__name__}: {str(exc)[:120]}", symbol, None))
            dead += 1
            continue
        rejects += bad
        listed += len(listings)
        for item in listings:
            if item.xbrl_url in known:
                dups += 1
                continue
            sleep(PAUSE_S)
            try:
                raw = _retrying(lambda: fetch(item.xbrl_url), sleep)
                parsed = parse_xbrl(raw)
            except Exception as exc:
                rejects.append(Rejected(f"xbrl failed: {type(exc).__name__}: {str(exc)[:120]}",
                                        symbol, item.published_raw))
                continue
            observed = now()
            if item.published_at > _iso(observed + FUTURE_TOLERANCE):
                rejects.append(Rejected("future-dated", symbol, item.published_raw))
                continue
            inserted += store.insert_result(conn, item, parsed, raw, _iso(observed), job_id)
            known.add(item.xbrl_url)

    state = "failed" if symbols and dead == len(symbols) else "ok"
    store.finish_job(conn, job_id, state=state, fetched=listed, inserted=inserted,
                     duplicates=dups, rejected=len(rejects), rejects=rejects,
                     error=f"{dead} of {len(symbols)} listings failed" if dead else None,
                     finished_at=_iso(now()))
    return dict(conn.execute("SELECT * FROM job_runs WHERE id = ?", (job_id,)).fetchone())


def summary(run: dict) -> str:
    return (f"{run['state']} listed={run['fetched']} inserted={run['inserted']} rejected={run['rejected']}"
            + (f" error={run['error']!r}" if run["error"] else ""))
