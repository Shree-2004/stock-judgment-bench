"""What a model would know about a company's quarterly results on a given night.

Step 2 of the fundamentals track (README plan item 8). Built from the
append-only `results` table (live/results.py); nothing here fetches, and no
running track reads it yet.

**Point in time.** A filing counts only if NSE published it at or before
`as_of` (the moment the call is made). A revision replaces the original only
from its own publication time. Hero MotoCorp's Sep-2026 results, broadcast at
21:08 IST on 28 Oct, are invisible to that evening's 18:30 call and visible to
the next one.

**One statement series.** Consolidated if its latest quarter is at least as
recent as the standalone one, else standalone. Year-on-year and four-quarter
figures never mix the two.

**Which fact is "revenue".** Companies file `RevenueFromOperations`; banks file
`InterestEarned` and no revenue line. The label says which one was used, so a
bank's interest income is never presented as "revenue".

**Derived quarters.** A few filings carry only year-to-date figures (Coal India,
Sep 2026: half year only). The quarter is then this filing's year-to-date minus
the previous quarter's, both as filed, and the block says it was derived. If
either is missing the quarter is left out, never guessed.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date

from live.results import quarter_facts

CRORE = 1e7
# First match wins. The label travels with the number.
REVENUE = (("Revenue from operations", "RevenueFromOperations"), ("Interest earned", "InterestEarned"))
PROFIT = ("ProfitLossForPeriod", "ProfitLossForThePeriod")
EPS = ("BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
       "BasicEarningsPerShareBeforeExtraordinaryItems",
       "BasicEarningsLossPerShareFromContinuingOperations")


@dataclass(frozen=True)
class Quarter:
    end: date
    published_at: str
    revenue_label: str | None
    revenue: float | None           # ₹, as filed
    profit: float | None            # ₹, profit for the period
    eps: float | None               # ₹ per share, basic
    derived: bool = False           # from year-to-date figures

    @property
    def label(self) -> str:
        """Indian fiscal quarter: Jun -> Q1 ... Mar -> Q4; FY27 ends March 2027."""
        q = {6: 1, 9: 2, 12: 3, 3: 4}[self.end.month]
        fy = self.end.year + (0 if self.end.month == 3 else 1)
        return f"Q{q} FY{fy % 100:02d}"


@dataclass(frozen=True)
class Fundamentals:
    symbol: str
    statement: str | None
    quarters: tuple[Quarter, ...]   # newest first
    close: float | None = None

    @property
    def latest(self) -> Quarter | None:
        return self.quarters[0] if self.quarters else None

    def _year_ago(self, q: Quarter) -> Quarter | None:
        target = _shift_year(q.end)
        return next((x for x in self.quarters if x.end == target), None)

    def yoy(self, attr: str) -> float | None:
        q = self.latest
        prev = self._year_ago(q) if q else None
        now, then = (getattr(q, attr), getattr(prev, attr)) if prev else (None, None)
        if now is None or then is None or then <= 0:
            return None             # growth off a loss or zero is not a number worth printing
        return (now / then - 1) * 100

    def eps_4q(self) -> float | None:
        if len(self.quarters) < 4 or self.quarters[3].end != _shift_year(self.quarters[0].end, quarters=3):
            return None             # the last four quarters must be consecutive
        values = [q.eps for q in self.quarters[:4]]
        return None if any(v is None for v in values) else sum(values)

    def pe(self) -> float | None:
        eps = self.eps_4q()
        return self.close / eps if self.close and eps and eps > 0 else None

    def figures(self) -> dict[str, float]:
        """Labelled figures a brief may cite (same contract as MARKET DATA)."""
        q = self.latest
        if q is None:
            return {}
        out = {}
        if q.revenue is not None:
            out[f"{q.revenue_label}, latest quarter (Rs cr)"] = round(q.revenue / CRORE, 2)
        if q.profit is not None:
            out["Net profit, latest quarter (Rs cr)"] = round(q.profit / CRORE, 2)
        if q.eps is not None:
            out["EPS, latest quarter (Rs)"] = round(q.eps, 2)
        for name, value in ((f"{q.revenue_label} growth YoY (%)", self.yoy("revenue")),
                            ("Net profit growth YoY (%)", self.yoy("profit")),
                            ("EPS, last 4 quarters (Rs)", self.eps_4q()),
                            ("P/E on last 4 quarters' EPS", self.pe())):
            if value is not None:
                out[name] = round(value, 2)
        return out

    def as_prompt_block(self, as_of: date) -> str:
        q = self.latest
        if q is None:
            return "(no quarterly results filed with NSE before this date)"
        published = date.fromisoformat(q.published_at[:10])
        lines = [f"Latest results: {q.label} (quarter ended {q.end:%d %b %Y}), {self.statement.lower()}, "
                 f"published {published:%d %b %Y} ({(as_of - published).days} days before {as_of:%d %b %Y})."
                 + (" Quarter derived from year-to-date figures as filed." if q.derived else "")]
        lines += [f"- {label}: {value:,.2f}" for label, value in self.figures().items()]
        prev = self._year_ago(q)
        if prev is None:
            lines.append("Same quarter a year earlier: not available, so no year-on-year figures.")
        return "\n".join(lines)


def _shift_year(d: date, quarters: int = 4) -> date:
    """The quarter-end `quarters` quarters before `d`."""
    month, year = d.month - 3 * quarters, d.year
    while month <= 0:
        month, year = month + 12, year - 1
    return date(year, month, 31 if month in (3, 12) else 30)


def _first(facts: dict, names) -> float | None:
    for n in names:
        if n in facts:
            try:
                return float(facts[n])
            except ValueError:
                return None
    return None


def _metrics(facts: dict) -> tuple[str | None, float | None, float | None, float | None]:
    label, revenue = next(((lab, _first(facts, (tag,))) for lab, tag in REVENUE if tag in facts), (None, None))
    return label, revenue, _first(facts, PROFIT), _first(facts, EPS)


def _ytd(parsed: dict, end: date) -> dict:
    """Facts for the period from the fiscal-year start (1 April) to `end`."""
    fy_start = date(end.year if end.month > 3 else end.year - 1, 4, 1).isoformat()
    for ctx, p in parsed["periods"].items():
        if p.get("start") == fy_start and p.get("end") == end.isoformat() and ctx in parsed["facts"]:
            return parsed["facts"][ctx]
    return {}


def load(conn: sqlite3.Connection, symbol: str, as_of_utc: str, close: float | None = None,
         keep: int = 8) -> Fundamentals:
    """`as_of_utc` is an ISO timestamp; only filings published by then count."""
    rows = conn.execute("""SELECT statement, quarter_end, published_at, facts_json FROM results
                           WHERE symbol = ? AND published_at <= ? ORDER BY published_at""",
                        (symbol, as_of_utc)).fetchall()
    latest: dict[tuple[str, str], tuple[str, dict]] = {}
    for statement, quarter_end, published_at, facts_json in rows:     # later publication wins
        latest[(statement, quarter_end)] = (published_at, json.loads(facts_json))
    newest = {s: max((qe for st, qe in latest if st == s), default="") for s in ("Consolidated", "Standalone")}
    if not any(newest.values()):
        return Fundamentals(symbol, None, (), close)
    statement = "Consolidated" if newest["Consolidated"] >= newest["Standalone"] else "Standalone"

    filings = {date.fromisoformat(qe): v for (st, qe), v in latest.items() if st == statement}
    quarters = []
    for end in sorted(filings, reverse=True):
        published_at, parsed = filings[end]
        facts, _ = quarter_facts(parsed)
        derived = False
        if not facts:
            prev_end = _shift_year(end, quarters=1)
            here, before = _ytd(parsed, end), (_ytd(filings[prev_end][1], prev_end) if prev_end in filings else {})
            if end.month == 6 or not here or not before:
                continue                                       # can't derive: leave it out
            a, b = _metrics(here), _metrics(before)
            if a[0] != b[0]:
                continue
            sub = [None if x is None or y is None else x - y for x, y in zip(a[1:], b[1:])]
            quarters.append(Quarter(end, published_at, a[0], *sub, derived=True))
            continue
        quarters.append(Quarter(end, published_at, *_metrics(facts)))
    return Fundamentals(symbol, statement, tuple(quarters[:keep]), close)
