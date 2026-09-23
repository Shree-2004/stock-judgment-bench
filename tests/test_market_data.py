"""Tests for the point-in-time data layer.

The leak tests are the point of this file. Everything else is scaffolding that
makes them trustworthy. Run the fast ones with:  pytest -m "not network"
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from data.market_data import (
    MarketDataError,
    MarketSnapshot,
    _coerce_as_of,
    _num,
    _pct_change,
    get_market_snapshot,
    normalise_symbol,
    post_cutoff_split_factor,
)

GOLDEN = Path(__file__).parent / "golden" / "snapshots.json"


# ---------------------------------------------------------------------------
# offline: normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [(1.5, 1.5), ("2.25", 2.25), (3, 3.0), (None, None), ("", None),
     ("n/a", None), (float("nan"), None), (float("inf"), None), ([], None)],
)
def test_num_coerces_yfinances_mixed_bag(raw, expected):
    assert _num(raw) == expected


def test_pct_change_basic():
    assert _pct_change(110.0, 100.0) == pytest.approx(10.0)
    assert _pct_change(90.0, 100.0) == pytest.approx(-10.0)


@pytest.mark.parametrize("new, old", [(None, 100.0), (100.0, None), (100.0, 0)])
def test_pct_change_guards_against_missing_and_zero(new, old):
    assert _pct_change(new, old) is None


@pytest.mark.parametrize(
    "raw, expected",
    [("RELIANCE", "RELIANCE.NS"), ("reliance", "RELIANCE.NS"),
     (" tcs ", "TCS.NS"), ("RELIANCE.NS", "RELIANCE.NS"), ("AAPL.X", "AAPL.X")],
)
def test_bare_names_get_the_nse_suffix(raw, expected):
    assert normalise_symbol(raw) == expected


def test_empty_ticker_rejected():
    with pytest.raises(MarketDataError):
        normalise_symbol("   ")


@pytest.mark.parametrize(
    "raw, expected",
    [("2024-06-03", date(2024, 6, 3)),
     (date(2024, 6, 3), date(2024, 6, 3)),
     (datetime(2024, 6, 3, 15, 30), date(2024, 6, 3)),
     ("2024-06-03T00:00:00Z", date(2024, 6, 3))],
)
def test_as_of_accepts_the_usual_shapes(raw, expected):
    assert _coerce_as_of(raw) == expected


def test_as_of_rejects_garbage():
    with pytest.raises(MarketDataError):
        _coerce_as_of("last tuesday")


# ---------------------------------------------------------------------------
# offline: split un-adjustment — the NSE-specific correctness bit
# ---------------------------------------------------------------------------

def _splits(pairs) -> pd.Series:
    return pd.Series(
        [r for _, r in pairs],
        index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in pairs]),
    )


def test_no_splits_means_no_adjustment():
    assert post_cutoff_split_factor(None, date(2024, 6, 3)) == 1.0
    assert post_cutoff_split_factor(_splits([]), date(2024, 6, 3)) == 1.0


def test_split_before_the_cutoff_is_already_priced_in():
    # yfinance has back-adjusted it; touching it again would double-count.
    s = _splits([("2024-01-05", 10.0)])
    assert post_cutoff_split_factor(s, date(2024, 6, 3)) == 1.0


def test_split_after_the_cutoff_is_undone():
    # RELIANCE 1:2 on 2024-10-28, snapshot taken 2024-06-03.
    s = _splits([("2024-10-28", 2.0)])
    assert post_cutoff_split_factor(s, date(2024, 6, 3)) == 2.0


def test_multiple_post_cutoff_splits_compound():
    # NESTLEIND: 10x on 2024-01-05 then 2x on 2025-08-08, seen from 2024-01-02.
    s = _splits([("2024-01-05", 10.0), ("2025-08-08", 2.0)])
    assert post_cutoff_split_factor(s, date(2024, 1, 2)) == 20.0


def test_split_exactly_on_the_cutoff_is_not_undone():
    s = _splits([("2024-06-03", 2.0)])
    assert post_cutoff_split_factor(s, date(2024, 6, 3)) == 1.0


def test_tz_aware_split_index_is_handled():
    # NSE history and splits come back in Asia/Kolkata.
    s = pd.Series([2.0], index=pd.DatetimeIndex([pd.Timestamp("2024-10-28 09:15:00", tz="Asia/Kolkata")]))
    assert post_cutoff_split_factor(s, date(2024, 6, 3)) == 2.0


def test_nonsense_split_ratios_are_ignored():
    assert post_cutoff_split_factor(_splits([("2025-01-01", 0.0)]), date(2024, 1, 1)) == 1.0


# ---------------------------------------------------------------------------
# offline: the figures() contract
# ---------------------------------------------------------------------------

def _snapshot(**overrides) -> MarketSnapshot:
    base = dict(
        ticker="TEST.NS", company_name="Test Ltd", currency="INR",
        as_of="2024-06-03", price=100.0, previous_close=98.0,
        change_pct_1d=2.04, change_pct_1mo=None, market_cap=1_000_000.0,
        week52_high=120.0, week52_low=80.0, pct_off_52w_high=-16.67,
        avg_volume_1mo=None, split_factor_applied=1.0, trading_days_used=250,
    )
    base.update(overrides)
    return MarketSnapshot(**base)


def test_figures_drops_missing_values_and_returns_floats():
    figures = _snapshot().figures()
    assert "change_pct_1mo" not in figures      # None is omitted, not zeroed
    assert "avg_volume_1mo" not in figures
    assert figures["price"] == 100.0
    assert all(isinstance(v, float) and not math.isnan(v) for v in figures.values())


def test_figures_excludes_context_and_provenance():
    # Only claims to be verified belong here. A name or a split factor is not
    # something the model asserts about the market.
    figures = _snapshot(change_pct_1mo=1.0, avg_volume_1mo=5.0).figures()
    assert set(figures) == {
        "price", "previous_close", "change_pct_1d", "change_pct_1mo",
        "market_cap", "week52_high", "week52_low", "pct_off_52w_high",
        "avg_volume_1mo",
    }
    for banned in ("ticker", "company_name", "currency", "as_of",
                   "split_factor_applied", "trading_days_used", "fetched_at"):
        assert banned not in figures


def test_as_dict_round_trips_every_field():
    snap = _snapshot()
    assert MarketSnapshot(**snap.as_dict()) == snap


# ---------------------------------------------------------------------------
# LEAK TESTS — the reason this module exists
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_leak_1_no_row_after_the_cutoff_is_used():
    """The snapshot must describe the cutoff day, not a later one."""
    snap = get_market_snapshot("RELIANCE", "2024-06-03", include_name=False)
    assert snap.as_of == "2024-06-03"
    # 2024-06-04 was a brutal -7% day for Reliance. If any of it bled in, the
    # 1d change would not match the 2024-06-03 close.
    assert snap.change_pct_1d == pytest.approx(5.5876, abs=0.01)
    assert snap.price == pytest.approx(3020.65, abs=0.5)


@pytest.mark.network
def test_leak_2_reconstructed_52w_high_differs_from_todays_info():
    """`info` is always now. Reconstructing from history must not match it.

    If these ever agree, someone has wired `info`/`fast_info` back in.
    """
    import yfinance as yf

    snap = get_market_snapshot("TCS", "2025-11-03", include_name=False)
    live_high = (yf.Ticker("TCS.NS").info or {}).get("fiftyTwoWeekHigh")

    assert snap.week52_high is not None
    assert live_high, "info gave no 52w high; test cannot discriminate"
    assert abs(snap.week52_high - float(live_high)) > 1.0, (
        f"point-in-time 52w high {snap.week52_high} matches today's {live_high} "
        f"— the snapshot is leaking current data"
    )


@pytest.mark.network
def test_leak_3_golden_snapshots_have_not_moved():
    """The past does not change. A diff here is a leak or a vendor revision."""
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))

    for key, want in expected.items():
        ticker, as_of = key.split("@")
        got = get_market_snapshot(ticker, as_of).as_dict()
        got.pop("fetched_at")

        for field, want_value in want.items():
            got_value = got[field]
            if isinstance(want_value, float):
                assert got_value == pytest.approx(want_value, rel=1e-6), (
                    f"{key}.{field}: {got_value} != {want_value}"
                )
            else:
                assert got_value == want_value, f"{key}.{field}: {got_value} != {want_value}"


@pytest.mark.network
def test_leak_4_a_future_split_is_undone_in_the_reported_price():
    """RELIANCE split 1:2 on 2024-10-28; a June snapshot must predate it."""
    snap = get_market_snapshot("RELIANCE", "2024-06-03", include_name=False)
    assert snap.split_factor_applied == 2.0
    # The raw stored close is ~1510; the actually-quoted price was ~3021.
    assert snap.price > 2500, "post-cutoff split was not undone"


@pytest.mark.network
def test_leak_5_two_cutoffs_disagree_as_they_should():
    """A snapshot is a function of its date, not of when it was computed."""
    early = get_market_snapshot("TCS", "2025-01-02", include_name=False)
    late = get_market_snapshot("TCS", "2025-11-03", include_name=False)
    assert early.price != late.price
    assert early.week52_high != late.week52_high


# ---------------------------------------------------------------------------
# network: general sanity
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_live_fetch_returns_sane_nse_data():
    snap = get_market_snapshot("reliance", "2025-11-03")
    assert snap.ticker == "RELIANCE.NS"
    assert snap.currency == "INR"
    assert snap.price > 0
    assert snap.week52_low <= snap.price <= snap.week52_high
    assert snap.trading_days_used > 200
    assert "price" in snap.figures()


@pytest.mark.network
def test_nonexistent_symbol_rejected():
    with pytest.raises(MarketDataError):
        get_market_snapshot("ZZZZNOTREAL", "2025-11-03")


@pytest.mark.network
def test_cutoff_before_listing_is_rejected_not_guessed():
    # Zomato listed in July 2021; a 2015 cutoff has no history to describe.
    with pytest.raises(MarketDataError):
        get_market_snapshot("ZOMATO", "2015-01-02", include_name=False)
