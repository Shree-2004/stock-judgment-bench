"""Forward returns — the answer key.

One subtlety decides whether these numbers mean anything.

**Use `Adj Close`, not `Close`.** A 21-trading-day window frequently contains an
ex-dividend date, where the price drops by the dividend without anyone losing
money. Raw `Close` records that as a real decline. `Adj Close` carries the
cumulative dividend-and-split adjustment, so the *ratio* between two dates
inside the window is total return. On Indian large caps — where 1-3% yields are
routine and dividends cluster around results season — scoring on raw closes
would systematically punish exactly the names that just paid out.

**No un-adjustment here, unlike `market_data`.** That module un-adjusts splits
because it reports an absolute price that had to match what was quoted. A
return is a ratio between two points in one adjustment basis, and a uniform
factor cancels. Applying the split correction here would be wrong.

Returns are also computed market-relative against `^NSEI`, because the raw
number mostly measures whether the Indian market went up that month.
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


class OutcomeError(RuntimeError):
    """The forward window could not be priced."""


@dataclass(frozen=True)
class Outcome:
    ticker: str
    cutoff: str
    forward_end: str
    start_price: float
    end_price: float
    forward_return_pct: float
    benchmark_return_pct: float | None
    excess_return_pct: float | None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _naive(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


@lru_cache(maxsize=256)
def _adj_closes(ticker: str, start: date, end: date) -> pd.Series:
    """Dividend- and split-adjusted closes, indexed by plain date."""
    try:
        hist = yf.Ticker(ticker).history(
            start=start, end=end + pd.Timedelta(days=5), auto_adjust=False
        )
    except Exception as exc:
        raise OutcomeError(f"history failed for {ticker}: {exc}") from exc
    if hist is None or hist.empty:
        raise OutcomeError(f"no history for {ticker} in {start}..{end}")

    column = "Adj Close" if "Adj Close" in hist.columns else "Close"
    series = hist[column].dropna()
    series.index = _naive(series.index)
    return series


def _price_on_or_before(series: pd.Series, when: date) -> tuple[date, float] | None:
    eligible = series[series.index.date <= when]
    if eligible.empty:
        return None
    return eligible.index[-1].date(), float(eligible.iloc[-1])


def get_outcome(ticker: str, cutoff: date, forward_end: date) -> Outcome:
    """Total return from `cutoff` to `forward_end`, and versus the Nifty."""
    series = _adj_closes(ticker, cutoff - pd.Timedelta(days=10), forward_end)

    start = _price_on_or_before(series, cutoff)
    end = _price_on_or_before(series, forward_end)
    if start is None or end is None:
        raise OutcomeError(f"{ticker}: missing price at {cutoff} or {forward_end}")
    if start[0] == end[0]:
        raise OutcomeError(f"{ticker}: start and end resolve to the same day {start[0]}")

    start_price, end_price = start[1], end[1]
    if start_price <= 0:
        raise OutcomeError(f"{ticker}: non-positive start price")
    forward = (end_price / start_price - 1) * 100.0

    benchmark = excess = None
    try:
        bench = _adj_closes(NIFTY, cutoff - pd.Timedelta(days=10), forward_end)
        b_start = _price_on_or_before(bench, cutoff)
        b_end = _price_on_or_before(bench, forward_end)
        if b_start and b_end and b_start[1] > 0:
            benchmark = (b_end[1] / b_start[1] - 1) * 100.0
            excess = forward - benchmark
    except OutcomeError:
        pass

    return Outcome(
        ticker=ticker,
        cutoff=cutoff.isoformat(),
        forward_end=forward_end.isoformat(),
        start_price=start_price,
        end_price=end_price,
        forward_return_pct=forward,
        benchmark_return_pct=benchmark,
        excess_return_pct=excess,
    )
