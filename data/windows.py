"""Benchmark cutoff dates, derived from the NSE trading calendar.

No holiday library is needed and none should be used: `^NSEI` history *is* the
NSE trading calendar. Days the index did not trade are days the market was
shut, including the irregular ones (Diwali Muhurat sessions, unscheduled
closures) that a generic calendar package gets wrong.

Windows are **non-overlapping**. Overlapping forward windows share price moves,
which would inflate the apparent sample size — the exact error this benchmark
exists to avoid making. 21 trading days per window, one cutoff per window.

The last cutoff must have a full forward window that has already happened,
otherwise its outcome cannot be scored.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import pandas as pd
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

NIFTY = "^NSEI"
FORWARD_TRADING_DAYS = 21

# GDELT's usable depth. Probed 2026-09-13: 5 and 7 years back both returned
# full result sets for Indian-source company queries.
DEFAULT_START = date(2019, 1, 1)

# The forward_end of the 90th window — the last outcome the v1 run scored.
# Without this pin, `end` defaulted to today and every 21 trading days the
# benchmark silently grew a window: by 2026-10-27 it expected 91, the report
# refused to run, and a scheduled run_briefs had tried to fill the new cell.
# A frozen benchmark is a fixed set of dates. Pass `end` explicitly to extend.
BENCHMARK_END = date(2026, 9, 1)


class CalendarError(RuntimeError):
    """Raised when the NSE calendar can't be established."""


@dataclass(frozen=True)
class Window:
    cutoff: date          # the brief is written as of this date
    forward_end: date     # 21 trading days later; the outcome is measured here

    def as_dict(self) -> dict:
        return {"cutoff": self.cutoff.isoformat(),
                "forward_end": self.forward_end.isoformat()}


@lru_cache(maxsize=4)
def nse_trading_days(start: date, end: date) -> tuple[date, ...]:
    """Every date NSE actually traded in [start, end], from Nifty 50 history."""
    try:
        hist = yf.Ticker(NIFTY).history(
            start=start, end=end + pd.Timedelta(days=1), auto_adjust=False
        )
    except Exception as exc:
        raise CalendarError(f"Could not fetch {NIFTY} history: {exc}") from exc

    if hist is None or hist.empty:
        raise CalendarError(f"{NIFTY} returned no history for {start}..{end}.")

    idx = pd.DatetimeIndex(hist.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return tuple(d.date() for d in idx if start <= d.date() <= end)


def build_windows(
    start: date = DEFAULT_START,
    end: date | None = None,
    forward_days: int = FORWARD_TRADING_DAYS,
) -> list[Window]:
    """Non-overlapping cutoffs, each with a fully-elapsed forward window.

    A cutoff is kept only if `forward_days` trading days after it are already
    in the past — an unscored window is not a sample.
    """
    end = end or BENCHMARK_END
    days = nse_trading_days(start, end)
    if len(days) < forward_days + 1:
        raise CalendarError(
            f"Only {len(days)} trading days in {start}..{end}; "
            f"need more than {forward_days}."
        )

    windows: list[Window] = []
    i = 0
    while i + forward_days < len(days):
        windows.append(Window(cutoff=days[i], forward_end=days[i + forward_days]))
        i += forward_days
    return windows


def summarise(windows: list[Window]) -> str:
    if not windows:
        return "no windows"
    return (
        f"{len(windows)} non-overlapping windows, "
        f"{windows[0].cutoff} .. {windows[-1].cutoff} "
        f"(last outcome measured {windows[-1].forward_end})"
    )
