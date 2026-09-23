"""The benchmark grid: one cached row per (ticker, window).

Built after the first full baseline run came back with 1,826 of 3,690 pairs and
the rest silently skipped. yfinance had started refusing under sustained load —
the same failure GDELT produced, arriving from a different direction.

Two things made that worse than a slow run:

**The missingness was not random.** Tickers are iterated in a fixed order
within each window, so when throttling began mid-window it always dropped the
same tail of the universe. A baseline computed on that subset is measuring a
biased slice, not the benchmark.

**It failed quietly.** Every exception was caught and counted, so the run
completed, printed a tidy table, and the numbers looked publishable. A result
that survives half its data going missing is exactly the kind of thing this
project exists to catch.

So: fetch once with retries, cache to disk, and score from the frozen grid.
Scoring must never touch the network — a benchmark whose inputs change between
runs is not a benchmark. `load_grid` refuses to return an incomplete grid
unless explicitly told to.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from data.market_data import MarketDataError
from data.snapshot_cache import get_snapshot_cached
from evals.outcomes import OutcomeError, get_outcome

GRID_ROOT = Path(__file__).resolve().parent.parent / "cache" / "grid"


class GridIncompleteError(RuntimeError):
    """The grid has holes. Scoring on it would produce a biased result."""


@dataclass
class GridRow:
    ticker: str
    cutoff: str
    forward_end: str
    change_pct_1mo: float | None
    price: float | None
    forward_return: float | None
    excess_return: float | None
    status: str                      # "ok" | "failed"
    error: str | None = None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "GridRow":
        known = {k: d.get(k) for k in cls.__dataclass_fields__}
        return cls(**known)


def cell_path(ticker: str, cutoff: date | str) -> Path:
    cutoff_s = cutoff if isinstance(cutoff, str) else cutoff.isoformat()
    safe = ticker.replace("&", "_AND_").replace("/", "_")
    return GRID_ROOT / safe / f"{cutoff_s}.json"


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


def load_cell(ticker: str, cutoff: date | str) -> GridRow | None:
    path = cell_path(ticker, cutoff)
    if not path.exists():
        return None
    try:
        return GridRow.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError, TypeError):
        return None


def fetch_cell(
    ticker: str,
    cutoff: date,
    forward_end: date,
    max_attempts: int = 4,
    base_delay: float = 2.0,
    refetch_failed: bool = False,
) -> GridRow:
    """Fill one grid cell, retrying through throttling rather than skipping."""
    existing = load_cell(ticker, cutoff)
    if existing and (existing.status != "failed" or not refetch_failed):
        return existing

    last_error = "unknown"
    for attempt in range(max_attempts):
        try:
            snap = get_snapshot_cached(ticker, cutoff, include_name=False)
            out = get_outcome(ticker, cutoff, forward_end)
            row = GridRow(
                ticker=ticker,
                cutoff=cutoff.isoformat(),
                forward_end=forward_end.isoformat(),
                change_pct_1mo=snap.change_pct_1mo,
                price=snap.price,
                forward_return=out.forward_return_pct,
                excess_return=out.excess_return_pct,
                status="ok",
            )
            _write_atomic(cell_path(ticker, cutoff), row.as_dict())
            return row
        except (MarketDataError, OutcomeError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            # Throttling looks like a data error from up here, so back off
            # rather than concluding the data does not exist.
            time.sleep(base_delay * (2 ** attempt))
        except Exception as exc:                      # noqa: BLE001
            last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            time.sleep(base_delay * (2 ** attempt))

    row = GridRow(
        ticker=ticker, cutoff=cutoff.isoformat(), forward_end=forward_end.isoformat(),
        change_pct_1mo=None, price=None, forward_return=None, excess_return=None,
        status="failed", error=last_error,
    )
    _write_atomic(cell_path(ticker, cutoff), row.as_dict())
    return row


def grid_stats(tickers: list[str], cutoffs: list[date]) -> dict:
    total = len(tickers) * len(cutoffs)
    ok = failed = missing = 0
    for t in tickers:
        for c in cutoffs:
            cell = load_cell(t, c)
            if cell is None:
                missing += 1
            elif cell.status == "ok":
                ok += 1
            else:
                failed += 1
    return {
        "expected": total, "ok": ok, "failed": failed, "missing": missing,
        "complete": missing == 0 and failed == 0,
        "coverage": round(ok / total, 4) if total else 0.0,
    }


def load_grid(
    tickers: list[str],
    cutoffs: list[date],
    allow_incomplete: bool = False,
    min_names_per_date: int = 3,
) -> dict[str, list[dict]]:
    """Frozen grid as {cutoff: [row, ...]}. Never touches the network.

    Raises GridIncompleteError unless the grid is whole, because a partial grid
    is how the first run produced numbers that looked fine and were not.
    """
    stats = grid_stats(tickers, cutoffs)
    if not stats["complete"] and not allow_incomplete:
        raise GridIncompleteError(
            f"grid has holes: {stats['ok']}/{stats['expected']} ok, "
            f"{stats['failed']} failed, {stats['missing']} missing. "
            f"Run: python -m scripts.build_grid --retry-failed"
        )

    by_date: dict[str, list[dict]] = {}
    for c in cutoffs:
        rows = []
        for t in tickers:
            cell = load_cell(t, c)
            if cell is None or cell.status != "ok" or cell.forward_return is None:
                continue
            rows.append(cell.as_dict())
        if len(rows) >= min_names_per_date:
            by_date[c.isoformat()] = rows
    return by_date
