"""Point-in-time market data for NSE equities.

The whole benchmark rests on one property: a snapshot built for a past date
must contain nothing that was unknowable on that date. Two things fight that.

1. `Ticker.info` and `Ticker.fast_info` are always *now*. AAPL's 52-week high
   read today is $344.57; reconstructed as of 2025-11-03 it is $271.40. Handing
   the model the former tells it the stock sits 21% below its high when it was
   sitting near it — an inverted signal, delivered as fact. So nothing
   time-varying is ever read from those. Everything comes from `history()`.

2. yfinance back-adjusts `Close` for splits that happened *after* the date you
   asked for. RELIANCE.NS closed at Rs 3,020.64 on 2024-06-03; yfinance stores
   Rs 1,510.32, because of a 1:2 split on 2024-10-28. Reporting the stored
   number would quote a price that was never quoted, and the price level itself
   would encode the later split. So closes are un-adjusted by the cumulative
   ratio of splits occurring strictly after `as_of` — which matters a lot on
   NSE, where splits and bonus issues are frequent (NESTLEIND 10x in 2024 and
   2x in 2025; IRCTC 5x; RELIANCE 2x).

Trading holidays need no calendar: `history()` returns only real trading rows,
so "21 trading days" is just 21 rows.

Everything the LLM is allowed to talk about comes out of `figures()`, a flat
{label: number} dict. That dict is the ground truth the L1 grounding check
scores against, so a figure that is not in it cannot be verified.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import lru_cache

import pandas as pd
import yfinance as yf

# yfinance logs "No data found, symbol may be delisted" straight to stderr for
# typos; we raise MarketDataError with a better message instead.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

TRADING_DAYS_YEAR = 252
TRADING_DAYS_MONTH = 21
# 252 trading days needs ~370 calendar days; pad for holiday clusters.
_LOOKBACK_CALENDAR_DAYS = 430


class MarketDataError(RuntimeError):
    """Raised when a symbol can't be resolved or lacks usable history."""


@dataclass
class MarketSnapshot:
    ticker: str
    company_name: str | None
    currency: str
    as_of: str                      # the date this snapshot is true for
    price: float
    previous_close: float | None
    change_pct_1d: float | None
    change_pct_1mo: float | None
    market_cap: float | None
    week52_high: float | None
    week52_low: float | None
    pct_off_52w_high: float | None
    avg_volume_1mo: float | None
    split_factor_applied: float     # 1.0 unless splits happened after as_of
    trading_days_used: int
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def figures(self) -> dict[str, float]:
        """Label -> number. The ground-truth set for the L1 grounding check.

        Deliberately excludes company_name, currency and the provenance fields:
        those are context, not claims to be verified.
        """
        candidates = {
            "price": self.price,
            "previous_close": self.previous_close,
            "change_pct_1d": self.change_pct_1d,
            "change_pct_1mo": self.change_pct_1mo,
            "market_cap": self.market_cap,
            "week52_high": self.week52_high,
            "week52_low": self.week52_low,
            "pct_off_52w_high": self.pct_off_52w_high,
            "avg_volume_1mo": self.avg_volume_1mo,
        }
        return {k: float(v) for k, v in candidates.items() if v is not None}

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _num(value) -> float | None:
    """Coerce yfinance's mixed bag (None, nan, numpy scalars, strings) to float."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return out


def _pct_change(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return (new - old) / old * 100.0


def normalise_symbol(ticker: str) -> str:
    """Bare names get .NS. yfinance returns nothing for .BO, so NSE only."""
    symbol = ticker.strip().upper()
    if not symbol:
        raise MarketDataError("No ticker given.")
    return symbol if "." in symbol else f"{symbol}.NS"


def _coerce_as_of(as_of: str | date | datetime | None) -> date:
    if as_of is None:
        return datetime.now(timezone.utc).date()
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    try:
        return datetime.strptime(str(as_of)[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise MarketDataError(f"as_of must be YYYY-MM-DD, got {as_of!r}") from exc


def _naive_dates(index: pd.Index) -> pd.Index:
    """NSE history comes back tz-aware (Asia/Kolkata); compare on plain dates."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def post_cutoff_split_factor(splits: pd.Series | None, as_of: date) -> float:
    """Cumulative ratio of splits occurring strictly AFTER as_of.

    yfinance has already divided historical closes by this. Multiplying it back
    recovers the price as actually quoted on as_of.
    """
    if splits is None or len(splits) == 0:
        return 1.0
    idx = _naive_dates(splits.index)
    factor = 1.0
    for stamp, ratio in zip(idx, splits.values):
        if stamp.date() > as_of:
            r = _num(ratio)
            if r and r > 0:
                factor *= r
    return factor


@lru_cache(maxsize=512)
def _company_name(symbol: str) -> str | None:
    """The one field taken from `info`.

    A company's name is not a time-varying market signal, so it cannot leak a
    price. Cached because the batch pipeline resolves the same names repeatedly
    and `info` is a slow scrape.
    """
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return None
    return info.get("longName") or info.get("shortName")


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------

def get_market_snapshot(
    ticker: str,
    as_of: str | date | datetime | None = None,
    include_name: bool = True,
) -> MarketSnapshot:
    """Everything knowable about `ticker` on `as_of`, and nothing more.

    Args:
        ticker:   NSE symbol; a bare name gets .NS appended.
        as_of:    YYYY-MM-DD. Defaults to today. Rows after this date are
                  dropped before any figure is computed.
        include_name: resolve the company name via `info` (one cached call).

    Raises:
        MarketDataError: unknown symbol, or too little history before as_of.
    """
    symbol = normalise_symbol(ticker)
    cutoff = _coerce_as_of(as_of)

    tk = yf.Ticker(symbol)
    start = cutoff - pd.Timedelta(days=_LOOKBACK_CALENDAR_DAYS)
    # end is exclusive in yfinance; +1 day so the cutoff itself is included.
    end = cutoff + pd.Timedelta(days=1)

    try:
        hist = tk.history(start=start, end=end, auto_adjust=False)
    except Exception as exc:
        raise MarketDataError(f"History fetch failed for {symbol}: {exc}") from exc

    if hist is None or hist.empty:
        raise MarketDataError(
            f"No price history for '{symbol}' before {cutoff}. "
            f"Check the symbol (NSE names need the .NS suffix)."
        )

    # Belt and braces: drop anything at or after the cutoff even though `end`
    # should have. This is the first leak guard, not a formality.
    hist = hist.copy()
    hist.index = _naive_dates(hist.index)
    hist = hist[hist.index.date <= cutoff]

    closes = hist["Close"].dropna()
    if len(closes) < 2:
        raise MarketDataError(
            f"Only {len(closes)} trading day(s) for '{symbol}' before {cutoff}; "
            f"need at least 2. Symbol may not have listed yet."
        )

    # Un-adjust for splits that happened after the cutoff, so the price is what
    # was actually quoted on the day rather than what yfinance stores now.
    try:
        factor = post_cutoff_split_factor(tk.splits, cutoff)
    except Exception:
        factor = 1.0
    if factor != 1.0:
        closes = closes * factor

    price = _num(closes.iloc[-1])
    if price is None:
        raise MarketDataError(f"Unusable close for '{symbol}' at {cutoff}.")

    prev_close = _num(closes.iloc[-2]) if len(closes) >= 2 else None

    month_window = closes.tail(TRADING_DAYS_MONTH + 1)
    change_1mo = (
        _pct_change(price, _num(month_window.iloc[0]))
        if len(month_window) >= TRADING_DAYS_MONTH + 1
        else None
    )

    year_window = closes.tail(TRADING_DAYS_YEAR)
    high52 = _num(year_window.max())
    low52 = _num(year_window.min())

    volumes = hist["Volume"].dropna().tail(TRADING_DAYS_MONTH)
    avg_volume = _num(volumes.mean()) if len(volumes) else None

    return MarketSnapshot(
        ticker=symbol,
        company_name=_company_name(symbol) if include_name else None,
        currency="INR" if symbol.endswith((".NS", ".BO")) else "USD",
        as_of=cutoff.isoformat(),
        price=price,
        previous_close=prev_close,
        change_pct_1d=_pct_change(price, prev_close),
        change_pct_1mo=change_1mo,
        market_cap=_market_cap_at(tk, cutoff, price, factor),
        week52_high=high52,
        week52_low=low52,
        pct_off_52w_high=_pct_change(price, high52),
        avg_volume_1mo=avg_volume,
        split_factor_applied=factor,
        trading_days_used=len(closes),
    )


def _market_cap_at(tk: yf.Ticker, cutoff: date, price: float, factor: float) -> float | None:
    """Shares outstanding as of the cutoff, times the price as quoted then.

    `get_shares_full` returns the actual historical count, which on `cutoff` is
    the pre-split count — consistent with the un-adjusted price, so the product
    is the market cap as it stood. Returns None rather than guessing when the
    series is unavailable, which happens for smaller names.
    """
    try:
        # 400 days rather than 180: the series is sparse and irregular, and a
        # short window silently loses market cap for recent cutoffs.
        shares = tk.get_shares_full(
            start=cutoff - pd.Timedelta(days=400),
            end=cutoff + pd.Timedelta(days=1),
        )
    except Exception:
        return None
    if shares is None or len(shares) == 0:
        return None

    idx = _naive_dates(shares.index)
    eligible = [v for stamp, v in zip(idx, shares.values) if stamp.date() <= cutoff]
    if not eligible:
        return None
    count = _num(eligible[-1])
    if count is None or count <= 0:
        return None
    return count * price
