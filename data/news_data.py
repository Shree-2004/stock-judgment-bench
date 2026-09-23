"""Disclosure selection from the BSE cache.

The selection rule is part of the experiment, not an implementation detail. A
21-day window holds 9-41 filings, a brief shows a handful, so "which five"
decides what the model can say. The rule is therefore fixed, deterministic, and
written down *before* any scoring run:

1. drop anything filed at or after the cutoff boundary,
2. drop administrative noise by category (below),
3. drop pure regulation boilerplate that carries no content,
4. de-duplicate on a normalised headline,
5. sort by filing time descending,
6. take the first N.

Explicitly **not** done: relevance or importance ranking. Scoring filings by how
market-moving they look is a judgement call that would need its own validation,
and a bad ranker would quietly become part of the result.

**Why no crawl-drift buffer.** GDELT needed a 24h safety margin because its
timestamps record when a crawler saw an article, not when it published. BSE
timestamps are exchange filing times — authoritative to the second — so a
strict "filed before the cutoff" comparison is exact, and costs no freshness.
This is the single biggest advantage of the disclosure channel over news.

**The noise filter.** Roughly a third of filings are administrative: lost share
certificates under Reg. 39(3), newspaper-publication intimations, and bare
"Disclosure under Regulation 30..." stubs whose content lives only in an
attached PDF. They are real filings but carry no information a brief could use,
and letting them crowd out the informative ones would weaken the model's input
for no reason. What is dropped is listed explicitly so the choice is auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable

from data.disclosures import LOOKBACK_DAYS, load_cached

DEFAULT_LIMIT = 5

# Subcategories that are purely administrative.
NOISE_SUBCATEGORIES = frozenset({
    "reg. 39 (3) - details of loss of certificate / duplicate certificate",
    "newspaper publication",
    "loss of share certificates",
})

# Headlines that are nothing but the regulation citation — the substance is in
# an attachment the benchmark does not read.
_BOILERPLATE = re.compile(
    r"^(disclosure|intimation|announcement|compliances)"
    r"[\s\-]*(under|of)?[\s\-]*(regulation|reg\.?)\s*\d+",
    re.IGNORECASE,
)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

# Some filings arrive as HTML fragments — bold tags, <br/>, whole <table>s.
# The cache keeps them verbatim (it should be faithful to the source); cleaning
# happens on read, so the rule can change without refetching 3,690 windows.
_TAG = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&(nbsp|amp|lt|gt|quot|#\d+);", re.IGNORECASE)
_ENTITIES = {"nbsp": " ", "amp": "&", "lt": "<", "gt": ">", "quot": '"'}


def clean_text(raw: str) -> str:
    """Strip HTML, decode the common entities, collapse whitespace.

    BSE headlines also carry doubled apostrophes from SQL escaping ('' for ')
    and hard line breaks from the filing form.
    """
    text = _TAG.sub(" ", raw or "")
    text = _ENTITY.sub(
        lambda m: _ENTITIES.get(m.group(1).lower(), " "), text
    )
    text = text.replace("''", "'")
    return _WS.sub(" ", text).strip()


class DisclosureCacheMiss(RuntimeError):
    """The (symbol, cutoff) cell has not been fetched yet.

    Raised rather than silently returning nothing: an unfetched window and a
    genuinely quiet month must never look the same.
    """


@dataclass
class Item:
    headline: str
    category: str | None = None
    subcategory: str | None = None
    filed_at: str | None = None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class DisclosureBundle:
    symbol: str
    cutoff: str
    status: str
    items: list[Item] = field(default_factory=list)
    considered: int = 0
    dropped_after_cutoff: int = 0
    dropped_as_noise: int = 0
    dropped_as_duplicate: int = 0

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["items"] = [i.as_dict() for i in self.items]
        return d

    def as_prompt_block(self) -> str:
        """Numbered filings. The numbers are the citation handles the L2
        attribution check resolves against."""
        if not self.items:
            return "(no disclosures filed in this window)"
        lines = []
        for n, it in enumerate(self.items, 1):
            stamp = (it.filed_at or "")[:10]
            tag = f"  [{it.category}]" if it.category else ""
            lines.append(f"[{n}] {stamp}{tag} {it.headline}")
        return "\n".join(lines)


def normalise_headline(headline: str) -> str:
    """Collapse near-identical filings onto one key."""
    return _WS.sub(" ", _PUNCT.sub(" ", (headline or "").lower())).strip()


def is_noise(item: Item) -> bool:
    """Administrative filings that carry no usable information."""
    if item.subcategory and item.subcategory.strip().lower() in NOISE_SUBCATEGORIES:
        return True
    return bool(_BOILERPLATE.match((item.headline or "").strip()))


def select_disclosures(
    rows: Iterable[dict],
    cutoff: date,
    limit: int = DEFAULT_LIMIT,
) -> tuple[list[Item], dict[str, int]]:
    """Apply the fixed selection rule.

    Returns (items, {"after_cutoff": n, "noise": n, "duplicate": n}).
    """
    boundary = datetime.combine(cutoff, datetime.min.time())
    dropped = {"after_cutoff": 0, "noise": 0, "duplicate": 0}

    seen: set[str] = set()
    kept: list[tuple[datetime, Item]] = []

    for raw in rows or []:
        item = Item(
            headline=clean_text(raw.get("headline") or ""),
            category=raw.get("category"),
            subcategory=raw.get("subcategory"),
            filed_at=raw.get("filed_at"),
        )
        if not item.headline or not item.filed_at:
            continue

        try:
            stamp = datetime.fromisoformat(item.filed_at)
        except ValueError:
            continue

        if stamp >= boundary:
            dropped["after_cutoff"] += 1
            continue
        if is_noise(item):
            dropped["noise"] += 1
            continue

        key = normalise_headline(item.headline)
        if not key or key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append((stamp, item))

    kept.sort(key=lambda pair: pair[0], reverse=True)
    return [i for _, i in kept[:limit]], dropped


def get_disclosures(
    symbol: str,
    cutoff: date,
    limit: int = DEFAULT_LIMIT,
) -> DisclosureBundle:
    """Filings for one (symbol, cutoff), read from the cache only.

    Raises:
        DisclosureCacheMiss: the cell has not been fetched yet.
    """
    cell = load_cached(symbol, cutoff)
    if cell is None:
        raise DisclosureCacheMiss(
            f"No cached BSE window for {symbol} @ {cutoff}. "
            f"Run: python -m scripts.fetch_disclosures"
        )

    items, dropped = select_disclosures(cell.disclosures, cutoff, limit=limit)
    return DisclosureBundle(
        symbol=symbol,
        cutoff=cutoff.isoformat(),
        status=cell.status,
        items=items,
        considered=len(cell.disclosures or []),
        dropped_after_cutoff=dropped["after_cutoff"],
        dropped_as_noise=dropped["noise"],
        dropped_as_duplicate=dropped["duplicate"],
    )


__all__ = [
    "DEFAULT_LIMIT",
    "clean_text",
    "LOOKBACK_DAYS",
    "NOISE_SUBCATEGORIES",
    "DisclosureBundle",
    "DisclosureCacheMiss",
    "Item",
    "get_disclosures",
    "is_noise",
    "normalise_headline",
    "select_disclosures",
]
