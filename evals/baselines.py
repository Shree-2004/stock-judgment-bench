"""Baselines. Without these the model's number means nothing.

Four strategies, scored through exactly the same pipeline as the model so the
comparison is like-for-like:

- **always_bullish** — the honesty check. Under cross-sectional scoring it
  produces a constant score vector, so its rank correlation is undefined and
  its spread has no short leg. That is the *point*: scoring within a date makes
  market drift unexploitable, and a metric that rewarded this would be
  measuring the Indian bull market rather than any judgement.

- **random** — seeded, so it is reproducible. This is the null.

- **momentum** — 1-month trailing return, the classic cheap signal. The live
  probe showed the model's verdicts tracking recent momentum closely
  (bearish on HDFCBANK −3.05% and BAJFINANCE −4.17%, bullish on the strong
  movers), so this is the baseline that matters most. If the model merely
  re-reads momentum, it should score like momentum, and the benchmark's answer
  is "no independent judgement".

- **reversal** — momentum inverted. Included because mean-reversion at a
  one-month horizon is a real documented effect, so if momentum scores
  *negatively* that is a finding rather than noise, and having both makes the
  sign interpretable.

All of these use only information available at the cutoff — momentum comes from
the same point-in-time snapshot the model is shown, not from a fresh lookup.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from evals.scoring import VERDICT_SCORE


@dataclass(frozen=True)
class Baseline:
    name: str
    describe: str
    score_fn: Callable[[dict, random.Random], float]

    def verdict(self, score: float) -> str:
        if score > 0:
            return "BULLISH"
        if score < 0:
            return "BEARISH"
        return "NEUTRAL"


def _always_bullish(row: dict, rng: random.Random) -> float:
    return VERDICT_SCORE["BULLISH"]


def _random(row: dict, rng: random.Random) -> float:
    return rng.choice([-1.0, 0.0, 1.0])


def _momentum(row: dict, rng: random.Random) -> float:
    """Trailing 1-month return, as shown to the model in its snapshot."""
    value = row.get("change_pct_1mo")
    return float(value) if value is not None else 0.0


def _reversal(row: dict, rng: random.Random) -> float:
    return -_momentum(row, rng)


BASELINES: tuple[Baseline, ...] = (
    Baseline("always_bullish", "BULLISH on everything", _always_bullish),
    Baseline("random", "seeded coin flip", _random),
    Baseline("momentum", "trailing 1-month return", _momentum),
    Baseline("reversal", "negative trailing 1-month return", _reversal),
)

BY_NAME = {b.name: b for b in BASELINES}


def build_baseline_rows(
    snapshots_by_date: dict[str, list[dict]],
    seed: int = 20260913,
) -> dict[str, dict[str, list[tuple[float, float, str]]]]:
    """Run every baseline over the same (date, ticker) grid the model sees.

    Args:
        snapshots_by_date: {cutoff: [{change_pct_1mo, forward_return, ...}, ...]}

    Returns:
        {baseline_name: {cutoff: [(score, forward_return, verdict), ...]}}
    """
    out: dict[str, dict[str, list[tuple[float, float, str]]]] = {
        b.name: {} for b in BASELINES
    }
    for baseline in BASELINES:
        rng = random.Random(seed)
        for cutoff in sorted(snapshots_by_date):
            rows = []
            for row in snapshots_by_date[cutoff]:
                score = baseline.score_fn(row, rng)
                rows.append((score, row["forward_return"], baseline.verdict(score)))
            out[baseline.name][cutoff] = rows
    return out
