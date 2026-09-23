"""Tests for benchmark window construction.

The non-overlap property is the one that matters statistically: overlapping
forward windows share price moves, so treating them as independent samples
inflates the apparent sample size. That is the error this benchmark exists to
avoid making, so it gets a test rather than a comment.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from data import windows as W
from data.universe import UNIVERSE, get, tickers


# ---------------------------------------------------------------------------
# window construction (offline, via a synthetic calendar)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_calendar(monkeypatch):
    """200 consecutive weekdays, standing in for the NSE calendar."""
    days: list[date] = []
    d = date(2024, 1, 1)
    while len(days) < 200:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    monkeypatch.setattr(W, "nse_trading_days", lambda start, end: tuple(days))
    return days


def test_windows_do_not_overlap(fake_calendar):
    ws = W.build_windows(start=date(2024, 1, 1), end=date(2024, 12, 31))
    for earlier, later in zip(ws, ws[1:]):
        assert earlier.forward_end <= later.cutoff, (
            f"window ending {earlier.forward_end} overlaps the one starting {later.cutoff}"
        )


def test_forward_window_is_exactly_21_trading_days(fake_calendar):
    ws = W.build_windows(start=date(2024, 1, 1), end=date(2024, 12, 31))
    for w in ws:
        i = fake_calendar.index(w.cutoff)
        j = fake_calendar.index(w.forward_end)
        assert j - i == W.FORWARD_TRADING_DAYS


def test_every_window_has_a_fully_elapsed_forward_period(fake_calendar):
    """An unscored window is not a sample, so it must not be emitted."""
    ws = W.build_windows(start=date(2024, 1, 1), end=date(2024, 12, 31))
    last_tradable = fake_calendar[-1]
    for w in ws:
        assert w.forward_end <= last_tradable


def test_window_count_matches_the_calendar(fake_calendar):
    ws = W.build_windows(start=date(2024, 1, 1), end=date(2024, 12, 31))
    # 200 days, stepping 21, keeping only cutoffs with 21 days left after them.
    assert len(ws) == (len(fake_calendar) - 1) // W.FORWARD_TRADING_DAYS


def test_too_short_a_calendar_raises_rather_than_returning_nothing(monkeypatch):
    monkeypatch.setattr(W, "nse_trading_days",
                        lambda start, end: tuple(date(2024, 1, 1) + timedelta(days=i)
                                                 for i in range(5)))
    with pytest.raises(W.CalendarError):
        W.build_windows(start=date(2024, 1, 1), end=date(2024, 1, 10))


def test_cutoffs_are_strictly_increasing(fake_calendar):
    ws = W.build_windows(start=date(2024, 1, 1), end=date(2024, 12, 31))
    cutoffs = [w.cutoff for w in ws]
    assert cutoffs == sorted(cutoffs)
    assert len(set(cutoffs)) == len(cutoffs)


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------

def test_universe_symbols_are_unique():
    symbols = [c.symbol for c in UNIVERSE]
    assert len(set(symbols)) == len(symbols)


def test_every_company_has_a_quoted_india_scoped_query():
    for c in UNIVERSE:
        assert c.gdelt_query.count('"') == 2, f"{c.symbol}: name must be quoted"
        assert "sourcecountry:india" in c.gdelt_query, f"{c.symbol}: not India-scoped"


def test_tickers_get_the_nse_suffix():
    assert all(t.endswith(".NS") for t in tickers())


@pytest.mark.parametrize("key", ["RELIANCE", "reliance", "RELIANCE.NS"])
def test_lookup_accepts_symbol_or_ticker(key):
    assert get(key).symbol == "RELIANCE"


def test_lookup_rejects_names_outside_the_universe():
    # The universe is frozen on purpose; silently accepting extras would let
    # survivorship bias creep back in.
    with pytest.raises(KeyError):
        get("ZOMATO")


# ---------------------------------------------------------------------------
# live calendar (network)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_real_nse_calendar_excludes_weekends_and_holidays():
    days = W.nse_trading_days(date(2024, 1, 1), date(2024, 12, 31))
    assert 230 < len(days) < 260, f"{len(days)} trading days in 2024 is implausible"
    assert all(d.weekday() < 5 for d in days), "a weekend slipped into the calendar"
    # Republic Day 2024 (26 Jan) and Independence Day (15 Aug) were closures.
    assert date(2024, 1, 26) not in days
    assert date(2024, 8, 15) not in days


@pytest.mark.network
def test_real_windows_are_non_overlapping_and_plausible():
    ws = W.build_windows()
    assert len(ws) > 60, "too few windows for the outcome test to have any power"
    for earlier, later in zip(ws, ws[1:]):
        assert earlier.forward_end <= later.cutoff
