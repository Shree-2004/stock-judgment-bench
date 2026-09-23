"""Disk cache for point-in-time snapshots.

A snapshot is a pure function of (ticker, as_of): the past does not change, so
the same inputs must always give the same figures. That makes it cacheable, and
caching it buys three things that matter here.

**No two jobs fight over yfinance.** Throttling has already corrupted one full
baseline run — the grid builder and the brief runner both want the same
snapshots, and a shared cache means the second one costs nothing.

**Re-runs are free.** The `v2_prose` ablation needs the same 3,690 snapshots the
`v1_structured` run used. Without a cache that is 3,690 more network calls for
data that cannot have changed.

**The benchmark becomes reproducible.** Scoring reads frozen inputs rather than
whatever yfinance returns today. A benchmark whose inputs drift between runs is
not measuring anything stable.

Retries live here too, because a throttled fetch raising `MarketDataError` is
indistinguishable from "this symbol has no data" at the call site, and the first
baseline run proved how much damage that confusion does when it is silent.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

from data.market_data import MarketDataError, MarketSnapshot, get_market_snapshot

CACHE_ROOT = Path(__file__).resolve().parent.parent / "cache" / "snapshots"


def cache_path(ticker: str, as_of: date | str) -> Path:
    as_of_s = as_of if isinstance(as_of, str) else as_of.isoformat()
    safe = ticker.replace("&", "_AND_").replace("/", "_")
    return CACHE_ROOT / safe / f"{as_of_s}.json"


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


def load_snapshot(ticker: str, as_of: date | str) -> MarketSnapshot | None:
    path = cache_path(ticker, as_of)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("_error"):
            return None
        known = {k: data.get(k) for k in MarketSnapshot.__dataclass_fields__}
        return MarketSnapshot(**known)
    except (ValueError, OSError, TypeError):
        return None


def get_snapshot_cached(
    ticker: str,
    as_of: date,
    include_name: bool = False,
    max_attempts: int = 4,
    base_delay: float = 2.0,
) -> MarketSnapshot:
    """Cached snapshot, fetching with backoff on a miss.

    Raises MarketDataError only after every retry has failed, so a throttled
    window is not mistaken for a symbol that does not exist.
    """
    cached = load_snapshot(ticker, as_of)
    if cached is not None:
        return cached

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            snap = get_market_snapshot(ticker, as_of, include_name=include_name)
            payload = snap.as_dict()
            payload["_cached_at"] = datetime.now().isoformat(timespec="seconds")
            _write_atomic(cache_path(ticker, as_of), payload)
            return snap
        except MarketDataError as exc:
            last_error = exc
            time.sleep(base_delay * (2 ** attempt))
        except Exception as exc:                       # noqa: BLE001
            last_error = exc
            time.sleep(base_delay * (2 ** attempt))

    raise MarketDataError(
        f"{ticker} @ {as_of}: failed after {max_attempts} attempts — {last_error}"
    )


def cache_stats() -> dict:
    if not CACHE_ROOT.exists():
        return {"files": 0}
    return {"files": sum(1 for _ in CACHE_ROOT.rglob("*.json"))}
