"""Versioned prompt templates.

Prompts are versioned because step 6 compares them against each other on
identical (ticker, cutoff) pairs. A prompt that changes without its version
changing would silently mix two experiments, so `VERSION` is part of every
trace and every score.

**The structured contract is the whole point.** If the model answers in prose,
checking its numbers means asking another model what it claimed — an opinion
scoring an opinion. Forcing labelled figures and indexed filing citations makes
L1 and L2 exact string-and-arithmetic operations: free, deterministic, and not
themselves a source of error.

That has a cost worth stating up front: a rigid format probably suppresses
hallucination on its own, so v1's grounding scores are likely flattering
relative to how the same model behaves in free prose. `v2_prose` exists to
measure exactly that, and the comparison is a real finding rather than a
footnote.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    template: str

    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)


_CONTRACT = """\
Answer in EXACTLY this format. No preamble, no markdown, no extra sections.

VERDICT: BULLISH | BEARISH | NEUTRAL
CONFIDENCE: <integer 1-5>
KEY_FIGURES:
- <figure_label>: <number>
DRIVERS:
- "<short claim>" -> [<filing number>]
BULL_CASE: <one or two sentences>
BEAR_CASE: <one or two sentences>

Rules that will be checked mechanically:
- Every number under KEY_FIGURES must be copied from MARKET DATA, using the
  exact label shown there. Do not compute new figures. Do not round beyond
  two decimal places.
- Every DRIVERS line must cite one filing number that appears in DISCLOSURES.
  If a claim does not come from a listed filing, do not make the claim.
- If DISCLOSURES is empty, write "DRIVERS: none" and rely on price action only.
- VERDICT is your view of the NEXT 21 TRADING DAYS.\
"""

V1_STRUCTURED = Prompt(
    version="v1_structured",
    system=(
        "You are an equity analyst writing a short internal brief on an Indian "
        "(NSE) listed company. You have only the data given to you. You never "
        "use outside knowledge about what happened after the stated date, and "
        "you never state a figure that is not in the data provided."
    ),
    template="""\
Write a research brief for {company} ({ticker}) as of {as_of}.

You know nothing about events after {as_of}. Treat that date as today.

MARKET DATA (all figures as of {as_of}, in {currency}):
{figures_block}

DISCLOSURES filed with the exchange in the 21 days before {as_of}:
{disclosures_block}

{contract}
""",
)

# Same information, no format constraint — the control for "does the structure
# itself do the work?"
V2_PROSE = Prompt(
    version="v2_prose",
    system=V1_STRUCTURED.system,
    template="""\
Write a short research brief for {company} ({ticker}) as of {as_of}.

You know nothing about events after {as_of}. Treat that date as today.

MARKET DATA (all figures as of {as_of}, in {currency}):
{figures_block}

DISCLOSURES filed with the exchange in the 21 days before {as_of}:
{disclosures_block}

Cover what happened, why, the bull case and the bear case, and end with a
clear view on the next 21 trading days. Write naturally.
""",
)

PROMPTS: dict[str, Prompt] = {p.version: p for p in (V1_STRUCTURED, V2_PROSE)}
DEFAULT_VERSION = V1_STRUCTURED.version


def get(version: str = DEFAULT_VERSION) -> Prompt:
    if version not in PROMPTS:
        raise KeyError(f"Unknown prompt version {version!r}. Have: {sorted(PROMPTS)}")
    return PROMPTS[version]


def render_figures_block(figures: dict[str, float]) -> str:
    """The labelled figures the model is allowed to cite.

    Labels match `MarketSnapshot.figures()` exactly, because the L1 check looks
    up cited labels in that dict. A label the model invents cannot be verified,
    and is scored as unverifiable rather than silently ignored.
    """
    if not figures:
        return "(no market figures available)"
    return "\n".join(f"- {label}: {value:,.2f}" for label, value in sorted(figures.items()))


def build_prompt(
    snapshot,
    disclosures,
    version: str = DEFAULT_VERSION,
) -> tuple[Prompt, str]:
    """Returns (prompt, rendered_user_message)."""
    prompt = get(version)
    rendered = prompt.render(
        company=snapshot.company_name or snapshot.ticker,
        ticker=snapshot.ticker,
        as_of=snapshot.as_of,
        currency=snapshot.currency,
        figures_block=render_figures_block(snapshot.figures()),
        disclosures_block=disclosures.as_prompt_block(),
        contract=_CONTRACT,
    )
    return prompt, rendered
