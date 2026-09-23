"""Parsing a model's answer into a checkable structure.

The parser is deliberately forgiving about *shape* and strict about *content*.
Models drift on formatting — markdown bullets, bold headers, a stray preamble,
`**VERDICT:**` instead of `VERDICT:` — and rejecting those would measure
instruction-following rather than grounding, which is not what this benchmark
is about. So cosmetic noise is absorbed.

What is never absorbed: a figure the model did not label, a citation that is
not an integer, or a verdict outside the three allowed values. Those become
`parse_errors`, which are reported rather than silently repaired, because
"the model produced something uncheckable" is itself a result worth counting.

A failed parse is not scored as a bad brief. It is scored as `parsed=False`,
kept separate — conflating "wrote nonsense" with "wrote a wrong number" would
corrupt exactly the signal L1 exists to measure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

VERDICTS = ("BULLISH", "BEARISH", "NEUTRAL")

# Strip markdown emphasis and list markers before matching section headers.
_DECORATION = re.compile(r"^[\s>*_#-]+|[\s*_]+$")
_SECTION = re.compile(
    r"^(VERDICT|CONFIDENCE|KEY_FIGURES|DRIVERS|BULL_CASE|BEAR_CASE)\s*:?\s*(.*)$",
    re.IGNORECASE,
)
# "- price: 3,020.65" / "price = 3020.65" / "* **price**: 3020.65"
_FIGURE = re.compile(
    r"^[\s\-*•]*\**\s*([A-Za-z_][A-Za-z0-9_]*)\**\s*[:=]\s*"
    r"\**\s*(-?[\d,]+(?:\.\d+)?)\s*\**\s*$"
)
# '"claim text" -> [2]'  / '- claim text -> filing 2' / 'claim (see [2])'
# The citation is the LAST integer on the line. Everything before it is the
# claim. Trailing brackets/punctuation are allowed because models wrap
# citations in every combination of (), [] and "see ...".
_DRIVER = re.compile(
    r"^(.*?)(?:->|—|–|-->|\bsee\b)?[\s(\[]*(?:filing\s*)?(\d+)[\s)\]\.]*$"
)

_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")


@dataclass
class Driver:
    claim: str
    citation: int | None            # 1-based filing index as the model gave it


@dataclass
class Brief:
    raw: str
    verdict: str | None = None
    confidence: int | None = None
    cited_figures: dict[str, float] = field(default_factory=dict)
    drivers: list[Driver] = field(default_factory=list)
    bull_case: str | None = None
    bear_case: str | None = None
    parse_errors: list[str] = field(default_factory=list)

    @property
    def parsed(self) -> bool:
        """Did we get the minimum needed to score anything at all?"""
        return self.verdict is not None

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "cited_figures": self.cited_figures,
            "drivers": [{"claim": d.claim, "citation": d.citation} for d in self.drivers],
            "bull_case": self.bull_case,
            "bear_case": self.bear_case,
            "parsed": self.parsed,
            "parse_errors": self.parse_errors,
        }


def _clean(line: str) -> str:
    return _DECORATION.sub("", line).strip()


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_brief(text: str) -> Brief:
    """Parse a structured brief. Never raises — failures land in parse_errors."""
    brief = Brief(raw=text or "")
    if not (text or "").strip():
        brief.parse_errors.append("empty response")
        return brief

    section: str | None = None
    buffers: dict[str, list[str]] = {"BULL_CASE": [], "BEAR_CASE": []}

    for raw_line in text.splitlines():
        line = _clean(raw_line)
        if not line:
            continue

        header = _SECTION.match(line)
        if header:
            section = header.group(1).upper()
            inline = header.group(2).strip()

            if section == "VERDICT" and inline:
                # Look for the verdict word anywhere in the line rather than
                # assuming it is the first token: "**VERDICT:** BULLISH" leaves
                # a stray "**" in front of it.
                found = [v for v in VERDICTS if re.search(rf"\b{v}\b", inline, re.I)]
                if len(found) == 1:
                    brief.verdict = found[0]
                else:
                    brief.parse_errors.append(f"unrecognised verdict: {inline[:40]!r}")
            elif section == "CONFIDENCE" and inline:
                m = _NUM.search(inline)
                value = int(float(m.group())) if m else None
                if value is not None and 1 <= value <= 5:
                    brief.confidence = value
                else:
                    brief.parse_errors.append(f"confidence out of range: {inline[:40]!r}")
            elif section in ("BULL_CASE", "BEAR_CASE") and inline:
                buffers[section].append(inline)
            elif section == "DRIVERS" and inline.lower() in ("none", "n/a", "-"):
                section = None          # explicit "no drivers" is valid
            continue

        if section == "KEY_FIGURES":
            m = _FIGURE.match(line)
            if not m:
                brief.parse_errors.append(f"unparseable figure line: {line[:60]!r}")
                continue
            label, value_text = m.group(1), m.group(2)
            value = _to_float(value_text)
            if value is None:
                brief.parse_errors.append(f"unparseable number for {label!r}: {value_text!r}")
                continue
            brief.cited_figures[label] = value

        elif section == "DRIVERS":
            body = line.lstrip("-*• ").strip()
            if not body or body.lower() in ("none", "n/a"):
                continue
            m = _DRIVER.match(body)
            if m and m.group(2):
                claim = m.group(1).strip().strip('"“”\'').rstrip("-–—→ ").strip()
                brief.drivers.append(Driver(claim=claim, citation=int(m.group(2))))
            else:
                # A claim with no citation is a finding, not a parse failure:
                # an uncited driver is precisely what L2 is looking for.
                brief.drivers.append(
                    Driver(claim=body.strip('"“”\''), citation=None)
                )

        elif section in ("BULL_CASE", "BEAR_CASE"):
            buffers[section].append(line)

    brief.bull_case = " ".join(buffers["BULL_CASE"]).strip() or None
    brief.bear_case = " ".join(buffers["BEAR_CASE"]).strip() or None

    if brief.verdict is None:
        brief.parse_errors.append("no VERDICT found")
    return brief
