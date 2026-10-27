"""Assemble the benchmark result from frozen artefacts.

Reads only what is already on disk — briefs, grid, disclosure cache. No network,
no model calls. That matters for a benchmark: running the report twice on the
same artefacts must give the same answer, and it must be possible to check the
answer without an API key.

What it refuses to do:

- **Score L3 on briefs whose grid rows are missing.** A verdict with no forward
  return is not a data point, and quietly dropping it would reintroduce the
  biased-subset problem that invalidated the first baseline run.
- **Report tier-2 attribution.** Measurement showed the overlap heuristic flags
  correct paraphrases as unsupported, so only tier 1 (citation validity) is a
  result. See `evals/attribution.py`.
- **Pool briefs across dates for L3.** One cutoff is one observation.

The model is scored through exactly the same `score_l3` path as every baseline,
so the comparison is like-for-like rather than two code paths that happen to
produce similar-looking numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from data.news_data import get_disclosures
from evals.attribution import check_attribution
from evals.scoring import VERDICT_SCORE, score_l3
from llm.brief import Driver

BRIEFS_ROOT = Path(__file__).resolve().parent.parent / "cache" / "briefs"


@dataclass
class BriefRecord:
    ticker: str
    symbol: str
    cutoff: str
    verdict: str | None
    confidence: int | None
    parsed: bool
    l1_score: float | None
    l1_clean: bool
    l1_wrong: int
    l1_unverifiable: int
    drivers: list = field(default_factory=list)
    error: str | None = None


def load_briefs(
    prompt_version: str, model: str, cutoffs: set[str] | None = None,
) -> list[BriefRecord]:
    """`cutoffs` restricts to the benchmark's dates. Without it, any stray file
    on disk — e.g. a cell for a window outside the frozen set — is counted."""
    from scripts.run_briefs import model_slug
    root = BRIEFS_ROOT / model_slug(model) / prompt_version
    out: list[BriefRecord] = []
    if not root.exists():
        return out

    for path in sorted(root.rglob("*.json")):
        if cutoffs is not None and path.stem not in cutoffs:
            continue
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue

        ticker = d.get("ticker", "")
        if d.get("error"):
            out.append(BriefRecord(
                ticker=ticker, symbol=ticker.replace(".NS", ""),
                cutoff=d.get("cutoff", ""), verdict=None, confidence=None,
                parsed=False, l1_score=None, l1_clean=False, l1_wrong=0,
                l1_unverifiable=0, error=d["error"],
            ))
            continue

        brief = d.get("brief") or {}
        l1 = d.get("l1") or {}
        out.append(BriefRecord(
            ticker=ticker,
            symbol=ticker.replace(".NS", ""),
            cutoff=d.get("cutoff", ""),
            verdict=brief.get("verdict"),
            confidence=brief.get("confidence"),
            parsed=bool(brief.get("parsed")),
            l1_score=l1.get("score"),
            l1_clean=bool(l1.get("clean")),
            l1_wrong=int(l1.get("wrong") or 0),
            l1_unverifiable=int(l1.get("unverifiable") or 0),
            drivers=[Driver(x.get("claim", ""), x.get("citation"))
                     for x in (brief.get("drivers") or [])],
        ))
    return out


def summarise_l1(briefs: list[BriefRecord]) -> dict:
    usable = [b for b in briefs if b.error is None]
    scored = [b.l1_score for b in usable if b.l1_score is not None]
    return {
        "briefs": len(usable),
        "failed": sum(1 for b in briefs if b.error),
        "unparsed": sum(1 for b in usable if not b.parsed),
        "mean_grounding": round(sum(scored) / len(scored), 4) if scored else None,
        "clean_rate": round(sum(1 for b in usable if b.l1_clean) / len(usable), 4)
        if usable else None,
        "briefs_with_a_wrong_figure": sum(1 for b in usable if b.l1_wrong > 0),
        "briefs_with_an_invented_label": sum(1 for b in usable if b.l1_unverifiable > 0),
        "briefs_citing_nothing_checkable": sum(
            1 for b in usable if b.l1_score is None and b.parsed
        ),
    }


def summarise_l2(briefs: list[BriefRecord], disclosure_limit: int = 5) -> dict:
    """Tier 1 only — exact citation validity. See attribution.py for why."""
    claims = fabricated = uncited = 0
    briefs_checked = briefs_clean = 0

    for b in briefs:
        if b.error is not None or not b.drivers:
            continue
        try:
            bundle = get_disclosures(b.symbol, date.fromisoformat(b.cutoff),
                                     limit=disclosure_limit)
        except Exception:
            continue
        result = check_attribution(b.drivers, bundle.items)
        claims += result.claims
        fabricated += result.fabricated
        uncited += result.uncited
        briefs_checked += 1
        briefs_clean += int(result.tier1_clean)

    return {
        "briefs_checked": briefs_checked,
        "claims": claims,
        "fabricated_citations": fabricated,
        "uncited_claims": uncited,
        "citation_validity": round((claims - fabricated - uncited) / claims, 4)
        if claims else None,
        "brief_clean_rate": round(briefs_clean / briefs_checked, 4)
        if briefs_checked else None,
    }


def build_model_rows(
    briefs: list[BriefRecord],
    grid_by_date: dict[str, list[dict]],
) -> tuple[dict[str, list[tuple[float, float, str]]], dict]:
    """Join verdicts to forward returns. Returns (rows, join_stats)."""
    returns_by_key = {
        (row["cutoff"], row["ticker"]): row["forward_return"]
        for rows in grid_by_date.values()
        for row in rows
        if row.get("forward_return") is not None
    }

    rows: dict[str, list[tuple[float, float, str]]] = {}
    joined = unmatched = unusable = 0

    for b in briefs:
        if b.error is not None or not b.parsed or b.verdict is None:
            unusable += 1
            continue
        forward = returns_by_key.get((b.cutoff, b.ticker))
        if forward is None:
            unmatched += 1
            continue
        rows.setdefault(b.cutoff, []).append(
            (VERDICT_SCORE.get(b.verdict, 0.0), forward, b.verdict)
        )
        joined += 1

    return rows, {
        "joined": joined,
        "no_grid_row": unmatched,
        "unparsed_or_failed": unusable,
    }


def build_report(
    prompt_version: str,
    model: str,
    grid_by_date: dict[str, list[dict]],
    baseline_summaries: dict | None = None,
    disclosure_limit: int = 5,
    cutoffs: list[date] | None = None,
) -> dict:
    wanted = {c.isoformat() for c in cutoffs} if cutoffs is not None else None
    briefs = load_briefs(prompt_version, model, wanted)
    model_rows, join_stats = build_model_rows(briefs, grid_by_date)
    l3 = score_l3(f"model:{prompt_version}", model_rows)

    verdicts: dict[str, int] = {}
    for b in briefs:
        if b.verdict:
            verdicts[b.verdict] = verdicts.get(b.verdict, 0) + 1

    return {
        "prompt_version": prompt_version,
        "model": model,
        "l1_grounding": summarise_l1(briefs),
        "l2_attribution_tier1": summarise_l2(briefs, disclosure_limit),
        "l3_outcome": l3.summary(),
        "l3_join": join_stats,
        "verdict_distribution": verdicts,
        "baselines": baseline_summaries or {},
    }
