"""BSE corporate announcements, cached on disk.

This replaces GDELT as the benchmark's information channel. GDELT's Doc API
rate-limits cumulatively per IP — it served the spike fine at 6s spacing and
then refused everything for the rest of the session — which makes a
~3,800-window bulk fetch unreliable at any pacing. BSE answers in 0.1-0.6s with
no throttling observed, and reaches back past 2020.

The trade is deliberate. Disclosures are not press coverage: they are what the
company itself told the exchange, timestamped by the exchange. For a benchmark
about whether a model's stated reasons are real, that is the stronger source —
a filing time is unambiguous, so there is no crawl-date drift to buffer against
and no question of whether an item was public before the cutoff.

Same cache discipline as before: one JSON file per (symbol, cutoff), written
atomically, skipped if present, resumable. Failures are recorded as "failed",
never as "no disclosures" — an unfetched window and a genuinely quiet month
must not look alike.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

LOOKBACK_DAYS = 21          # one benchmark window of context
CACHE_ROOT = Path(__file__).resolve().parent.parent / "cache" / "bse"

# BSE is fast but it is someone else's unofficial endpoint; stay polite.
MIN_INTERVAL_S = 0.35

_rate_lock = threading.Lock()
_last_call = 0.0
_client = None
_client_lock = threading.Lock()


class DisclosureUnavailableError(RuntimeError):
    """BSE was asked and would not answer.

    Distinct from "answered with nothing" on purpose: a refused fetch scored as
    an empty window would look like a company that filed nothing that month.
    """


@dataclass
class Disclosure:
    headline: str
    category: str | None = None
    subcategory: str | None = None
    filed_at: str | None = None          # ISO-8601
    attachment: str | None = None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class CachedWindow:
    symbol: str
    scrip_code: str
    cutoff: str
    window_start: str
    window_end: str
    status: str                          # "ok" | "empty" | "failed"
    disclosures: list[dict] = field(default_factory=list)
    error: str | None = None
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "CachedWindow":
        known = {k: d.get(k) for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in known.items() if v is not None})


def _client_instance():
    global _client
    with _client_lock:
        if _client is None:
            from bse import BSE

            _client = BSE(download_folder=str(CACHE_ROOT.parent))
        return _client


def _throttle() -> None:
    global _last_call
    with _rate_lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _parse_filed_at(row: dict) -> str | None:
    """BSE exposes several date fields with inconsistent formats."""
    for key in ("NEWS_DT", "DissemDT", "News_submission_dt", "DT_TM"):
        raw = row.get(key)
        if not raw:
            continue
        text = str(raw).strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(text[:26], fmt).isoformat(timespec="seconds")
            except ValueError:
                continue
    return None


def parse_row(row: dict) -> Disclosure | None:
    """BSE splits the text across two fields inconsistently.

    NEWSSUB is sometimes the regulation boilerplate and HEADLINE the substance
    ("Allotment of 3302544 Equity Shares pursuant to exercise of Warrants"),
    and sometimes the reverse. Taking the longer of the two gets the informative
    one far more often than trusting either field's position.
    """
    sub = (row.get("NEWSSUB") or "").strip()
    head = (row.get("HEADLINE") or "").strip()
    headline = head if len(head) > len(sub) else sub
    if not headline:
        return None
    return Disclosure(
        headline=headline,
        category=(row.get("CATEGORYNAME") or "").strip() or None,
        subcategory=(row.get("SUBCATNAME") or "").strip() or None,
        filed_at=_parse_filed_at(row),
        attachment=(row.get("ATTACHMENTNAME") or "").strip() or None,
    )


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def fetch_window(
    scrip_code: str,
    window_start: datetime,
    window_end: datetime,
    max_attempts: int = 3,
) -> list[Disclosure]:
    """Announcements for one scrip over one window. Raises rather than empty."""
    last_error = "unknown"
    for attempt in range(max_attempts):
        _throttle()
        try:
            res = _client_instance().announcements(
                scripcode=str(scrip_code),
                from_date=window_start,
                to_date=window_end,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(1.0 * (attempt + 1))
            continue

        rows = (res or {}).get("Table") or []
        out = [d for d in (parse_row(r) for r in rows) if d is not None]
        return out

    raise DisclosureUnavailableError(
        f"scrip {scrip_code} {window_start:%Y-%m-%d}..{window_end:%Y-%m-%d}: {last_error}"
    )


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def cache_path(symbol: str, cutoff: date | str) -> Path:
    cutoff_s = cutoff if isinstance(cutoff, str) else cutoff.isoformat()
    safe = symbol.replace("&", "_AND_").replace("/", "_")
    return CACHE_ROOT / safe / f"{cutoff_s}.json"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_cached(symbol: str, cutoff: date | str) -> CachedWindow | None:
    path = cache_path(symbol, cutoff)
    if not path.exists():
        return None
    try:
        return CachedWindow.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return None


def fetch_and_cache(
    symbol: str,
    scrip_code: str,
    cutoff: date,
    lookback_days: int = LOOKBACK_DAYS,
    refetch_failed: bool = False,
) -> CachedWindow:
    """Fill one cache cell; return immediately if already filled.

    The window ends at the cutoff *date boundary*, so a filing made during the
    cutoff session itself is excluded — a brief written as of that morning
    could not have seen it.
    """
    existing = load_cached(symbol, cutoff)
    if existing and (existing.status != "failed" or not refetch_failed):
        return existing

    end = datetime.combine(cutoff, datetime.min.time())
    start = end - timedelta(days=lookback_days)

    try:
        items = fetch_window(scrip_code, start, end)
        status = "ok" if items else "empty"
        error = None
    except DisclosureUnavailableError as exc:
        items, status, error = [], "failed", str(exc)

    entry = CachedWindow(
        symbol=symbol,
        scrip_code=str(scrip_code),
        cutoff=cutoff.isoformat(),
        window_start=start.isoformat(timespec="seconds"),
        window_end=end.isoformat(timespec="seconds"),
        status=status,
        disclosures=[d.as_dict() for d in items],
        error=error,
    )
    _write_atomic(cache_path(symbol, cutoff), entry.as_dict())
    return entry


def cache_stats() -> dict:
    counts = {"ok": 0, "empty": 0, "failed": 0, "unreadable": 0}
    total = 0
    if not CACHE_ROOT.exists():
        return {"files": 0, **counts, "disclosures": 0}
    files = list(CACHE_ROOT.rglob("*.json"))
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            counts["unreadable"] += 1
            continue
        status = d.get("status", "unreadable")
        counts[status] = counts.get(status, 0) + 1
        total += len(d.get("disclosures") or [])
    return {"files": len(files), **counts, "disclosures": total}
