"""Debugging tool for the point-in-time data layer.

Not the product — the product is the benchmark. This exists so you can eyeball
any (ticker, date) pair by hand and sanity-check that the numbers are what the
market actually showed that day.

    python -m app.cli RELIANCE --as-of 2024-06-03
    python -m app.cli TCS --as-of 2025-11-03 --json
"""
from __future__ import annotations

import argparse
import json
import sys

from data.market_data import MarketDataError, MarketSnapshot, get_market_snapshot


def _force_utf8() -> None:
    """Windows consoles default to cp1252, which mangles ₹ and the separators.

    Switching the console code page as well as the stream encoding is what
    actually fixes it — reconfiguring the stream alone just writes UTF-8 bytes
    into a cp1252 console and you get mojibake instead.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _sym(currency: str | None) -> str:
    return {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(currency or "", "")


def _fmt_money(value: float | None, currency: str | None) -> str:
    if value is None:
        return "n/a"
    s = _sym(currency)
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e7, "Cr"), (1e5, "L")):
        if abs(value) >= cutoff:
            return f"{s}{value / cutoff:,.2f}{suffix}"
    return f"{s}{value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


def render_snapshot(snap: MarketSnapshot) -> str:
    name = snap.company_name or snap.ticker
    header = f"{name} ({snap.ticker})  —  as of {snap.as_of}"
    lines = [
        header,
        "=" * len(header),
        f"  price         {_sym(snap.currency)}{snap.price:,.2f}  "
        f"({_fmt_pct(snap.change_pct_1d)} 1d, {_fmt_pct(snap.change_pct_1mo)} 1mo)",
        f"  prev close    {_sym(snap.currency)}{snap.previous_close:,.2f}"
        if snap.previous_close is not None else "  prev close    n/a",
        f"  market cap    {_fmt_money(snap.market_cap, snap.currency)}",
        f"  52-week range {_sym(snap.currency)}{snap.week52_low:,.2f} – "
        f"{_sym(snap.currency)}{snap.week52_high:,.2f}  "
        f"({_fmt_pct(snap.pct_off_52w_high)} vs high)",
        f"  avg volume    {_fmt_money(snap.avg_volume_1mo, None)}  (1mo)",
    ]
    if snap.split_factor_applied != 1.0:
        lines.append(
            f"  split adj     ×{snap.split_factor_applied:g} un-adjusted "
            f"(splits occurred after {snap.as_of}; this is the price as quoted then)"
        )
    lines.append(f"  history       {snap.trading_days_used} trading days")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Inspect the point-in-time snapshot for one NSE ticker.",
    )
    parser.add_argument("ticker", help="e.g. RELIANCE, TCS, HDFCBANK (.NS is implied)")
    parser.add_argument(
        "--as-of", default=None, metavar="YYYY-MM-DD",
        help="the date the snapshot should be true for. Default: today.",
    )
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    parser.add_argument("--no-name", action="store_true",
                        help="skip the company-name lookup (one less network call)")
    args = parser.parse_args(argv)

    try:
        snap = get_market_snapshot(args.ticker, args.as_of, include_name=not args.no_name)
    except MarketDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(snap.as_dict(), indent=2, ensure_ascii=False))
        return 0

    print(render_snapshot(snap))
    if args.as_of:
        print("\n  (news for past dates arrives in step 2 — GDELT cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
