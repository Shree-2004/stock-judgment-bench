"""L3: does a set of verdicts predict anything?

**Everything is scored within a date, never pooled across dates.** Briefs
sharing a cutoff are not independent observations — they all ride the same
market-wide move. Pooling 3,690 briefs and computing a t-statistic would give a
confident-looking number backed by ~90 real degrees of freedom, which is the
single easiest way to manufacture a result that is not there.

So each cutoff produces **one** observation, and those ~90 observations are the
sample.

Two metrics, both computed, one pre-registered as primary:

- **Rank correlation (PRIMARY).** Spearman correlation between the verdict
  score and forward return across the names on that date. Always defined as
  long as two distinct verdicts exist, and it uses the whole cross-section
  rather than the tails.

- **Long-short spread (SECONDARY).** Mean return of BULLISH names minus mean of
  BEARISH names. More intuitive, but undefined on any date where the model was
  bullish on everything — and in the probe it was bullish 10 of 16 times, so
  that will happen. A metric that quietly vanishes on the dates where the model
  was most one-sided is not a metric to lead with.

Both are market-neutral by construction: ranking within a date removes the
common market move, which is what makes "always bullish" a non-strategy here
rather than a strong baseline.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

VERDICT_SCORE = {"BEARISH": -1.0, "NEUTRAL": 0.0, "BULLISH": 1.0}


@dataclass
class DateResult:
    cutoff: str
    n: int
    rank_correlation: float | None
    long_short_spread: float | None
    n_bullish: int
    n_bearish: int
    n_neutral: int

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class L3Result:
    label: str
    dates: list[DateResult] = field(default_factory=list)

    # --- primary ---
    @property
    def mean_rank_correlation(self) -> float | None:
        values = [d.rank_correlation for d in self.dates if d.rank_correlation is not None]
        return sum(values) / len(values) if values else None

    @property
    def rank_correlation_t(self) -> float | None:
        """One-sample t across dates. n is the number of DATES, not briefs."""
        values = [d.rank_correlation for d in self.dates if d.rank_correlation is not None]
        if len(values) < 3:
            return None
        sd = statistics.stdev(values)
        if sd == 0:
            return None
        return (sum(values) / len(values)) / (sd / len(values) ** 0.5)

    # --- secondary ---
    @property
    def mean_spread(self) -> float | None:
        values = [d.long_short_spread for d in self.dates if d.long_short_spread is not None]
        return sum(values) / len(values) if values else None

    @property
    def dates_with_spread(self) -> int:
        return sum(1 for d in self.dates if d.long_short_spread is not None)

    @property
    def hit_rate(self) -> float | None:
        """Share of dates where the spread was positive. Sign test fodder."""
        values = [d.long_short_spread for d in self.dates if d.long_short_spread is not None]
        return sum(1 for v in values if v > 0) / len(values) if values else None

    def summary(self) -> dict:
        return {
            "label": self.label,
            "dates": len(self.dates),
            "mean_rank_correlation": _round(self.mean_rank_correlation, 4),
            "rank_correlation_t": _round(self.rank_correlation_t, 2),
            "mean_spread_pct": _round(self.mean_spread, 3),
            "dates_with_spread": self.dates_with_spread,
            "spread_hit_rate": _round(self.hit_rate, 3),
        }


def _round(value, places):
    return None if value is None else round(value, places)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties (very common in verdicts) are handled properly."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of ranks. None when either side is constant."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None          # no variation to correlate — not a zero result
    return num / (dx * dy)


def score_one_date(
    cutoff: str,
    scores: list[float],
    returns: list[float],
    verdicts: list[str] | None = None,
) -> DateResult:
    """One cutoff -> one observation."""
    verdicts = verdicts or []
    bulls = [r for v, r in zip(verdicts, returns) if v == "BULLISH"]
    bears = [r for v, r in zip(verdicts, returns) if v == "BEARISH"]

    spread = None
    if bulls and bears:
        spread = sum(bulls) / len(bulls) - sum(bears) / len(bears)

    return DateResult(
        cutoff=cutoff,
        n=len(scores),
        rank_correlation=spearman(scores, returns),
        long_short_spread=spread,
        n_bullish=len(bulls),
        n_bearish=len(bears),
        n_neutral=sum(1 for v in verdicts if v == "NEUTRAL"),
    )


def score_l3(
    label: str,
    by_date: dict[str, list[tuple[float, float, str | None]]],
) -> L3Result:
    """Args: {cutoff: [(verdict_score, forward_return, verdict), ...]}."""
    result = L3Result(label=label)
    for cutoff in sorted(by_date):
        rows = by_date[cutoff]
        if len(rows) < 3:
            continue
        result.dates.append(
            score_one_date(
                cutoff,
                [r[0] for r in rows],
                [r[1] for r in rows],
                [r[2] or "" for r in rows],
            )
        )
    return result
