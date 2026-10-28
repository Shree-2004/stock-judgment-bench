"""Normalised filing records, parsed from NSE's announcements RSS feed.

**Why NSE and not the BSE connector the benchmark used.** On 2026-10-28 every
BSE API request — historical windows included — returned an Akamai "Access
Denied" page. That is an edge block, not a bug, and getting around a bot block
is not something this project does. NSE publishes an official RSS feed that
answers an honestly-identified client, so it is the live source; BSE stays the
historical one.

**What the feed is actually like** (measured on a 1,055-item pull, 28 Oct 2026):

- It is a rolling window, not an archive. On a results-season evening it held
  only ~6 hours. Anything that scrolls off between polls is gone for good —
  there is no backfill — so the poller stores *every* item exchange-wide and
  records coverage gaps instead of pretending they cannot happen.
- There is no stable ID. 38 links repeat (one PDF eight times, at different
  seconds), and 27 items have no link at all (price-movement notices, ETF
  NAVs). Identity is therefore a hash of the fields that define the item.
- `pubDate` comes in two shapes (`28-Oct-2026 17:44:30` and
  `28-OCT-2026 18:56`) and is occasionally missing. Times are IST without an
  offset; `lastBuildDate` confirms +0530.
- Order is mostly newest-first, with occasional inversions.

The company is stored as the exchange printed it. Mapping it to a symbol
happens at read time (`tracked_symbol`), so a wrong mapping can be fixed
without touching an append-only row.
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
PROVIDER_NSE_RSS = "nse_rss"

_DATE_FORMATS = ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M")

# NSE's printed company names for the M1 watch list. Checked against the feed
# for Infosys; the rest follow the same "<Registered name> Limited" pattern and
# are verified by `scripts.live_ingest --unmatched` once each one files.
NSE_NAMES: dict[str, str] = {
    "RELIANCE": "Reliance Industries Limited",
    "TCS": "Tata Consultancy Services Limited",
    "INFY": "Infosys Limited",
    "HDFCBANK": "HDFC Bank Limited",
    "ICICIBANK": "ICICI Bank Limited",
}
DEFAULT_SYMBOLS = tuple(NSE_NAMES)


@dataclass(frozen=True)
class FilingRecord:
    provider: str
    provider_ref: str        # sha256 of the identity fields; the dedupe key
    company: str             # as the exchange printed it
    subject: str
    url: str | None          # absent on some notices, by design of the feed
    published_at: str        # UTC ISO-8601, converted from exchange-local time
    published_raw: str       # the string the exchange sent, kept for audit


@dataclass(frozen=True)
class Rejected:
    reason: str
    company: str | None
    published_raw: str | None


def _text(item: ET.Element, tag: str) -> str:
    return (item.findtext(tag) or "").strip()


def parse_pubdate(raw: str) -> datetime | None:
    """IST wall-clock -> aware UTC. `.title()` folds `OCT` into `Oct`."""
    text = raw.strip().title()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=IST).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def provider_ref(company: str, published_at: str, url: str | None, subject: str) -> str:
    key = "\x1f".join((company, published_at, url or "", subject))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def parse_nse_rss(xml_bytes: bytes) -> tuple[list[FilingRecord], list[Rejected]]:
    """Every item becomes a record or a rejection — nothing is silently dropped.

    Exact repeats within one feed are *not* collapsed here; they share a
    provider_ref and the store's UNIQUE constraint absorbs them, so they are
    counted as duplicates like any other repeat.
    """
    root = ET.fromstring(xml_bytes)
    records: list[FilingRecord] = []
    rejected: list[Rejected] = []

    for item in root.findall("./channel/item"):
        company = _text(item, "title")
        raw = _text(item, "pubDate")
        if not company:
            rejected.append(Rejected("missing company", None, raw or None))
            continue
        if not raw:
            rejected.append(Rejected("missing pubDate", company, None))
            continue
        published = parse_pubdate(raw)
        if published is None:
            rejected.append(Rejected("unparseable pubDate", company, raw))
            continue

        published_at = published.isoformat(timespec="seconds")
        url = _text(item, "link") or None
        subject = _text(item, "description")
        records.append(FilingRecord(
            provider=PROVIDER_NSE_RSS,
            provider_ref=provider_ref(company, published_at, url, subject),
            company=company,
            subject=subject,
            url=url,
            published_at=published_at,
            published_raw=raw,
        ))
    return records, rejected


def _norm(name: str) -> str:
    name = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    name = re.sub(r"\b(limited|ltd)\b", " ", name)
    return " ".join(name.split())


_BY_NORM = {_norm(n): s for s, n in NSE_NAMES.items()}


def tracked_symbol(company: str) -> str | None:
    """Exact match after normalisation. Not a prefix match: "HDFC Bank Limited"
    must not swallow some future "HDFC Bank Limited - NCD" listing."""
    return _BY_NORM.get(_norm(company))
