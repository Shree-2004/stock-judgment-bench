"""L2: are the model's stated *reasons* real, or invented?

L1 catches fabricated numbers. This catches the failure L1 cannot see: a brief
whose figures are all correct and whose causal story is made up. "The stock rose
on strong demand" passes every numeric check ever written.

**Two tiers, deliberately separated, because they are not equally trustworthy.**

Tier 1 — the hard check. Exact, no judgement:
  - `fabricated_citation`: the model cited filing [7] when four were supplied.
    There is no interpretation under which that is correct.
  - `uncited`: a driver with no citation at all, when the contract required one.

Tier 2 — the soft check. A heuristic, and labelled as one:
  - `supported` / `weak_match`: does the claim's wording overlap the filing it
    cites? Claims are paraphrases ("Acquisition of Udhaiyam brand expands
    consumer products footprint" against a filing headed "Acquisition of
    Udhaiyam Dhall Brand"), so overlap is fuzzy by nature.

The tiers are never summed into one number. Tier 1 is a finding; tier 2 is a
flag for review until it has been validated against hand-labelled claims. A
grounding score that mixed an exact result with an uncalibrated heuristic would
be exactly the kind of authoritative-looking number this project exists to
distrust — the same mistake as SEBI's substring verifier, one level up.

MEASURED, 2026-09-14, on 60 claims from 20 real briefs
------------------------------------------------------
Tier 1: **0 fabricated citations, 0 uncited claims.** Every driver pointed at a
filing that had actually been supplied. Exact, and a genuine result.

Tier 2: **the heuristic does not work.** Hand-reading every claim it flagged as
`weak_match` showed all of them were in fact correct:

  overlap 0.00  "Acquisition of edtech and tech startups expands digital
                 portfolio"  vs  a filing about Embibe, a Reliance subsidiary.
                 Embibe *is* edtech. The claim is right; the word is not in the
                 filing.
  overlap 0.14  "Active participation in investor conferences enhances market
                 visibility"  vs  "the Company participated in the investor
                 conference..." — plainly the same event.
  overlap 0.17  "Acquisition of Den Networks expands telecom footprint"  vs  a
                 filing whose headline is truncated boilerplate. The claim
                 fails on the *filing's* wording, not its own.

Token overlap measures shared vocabulary, and a correct paraphrase routinely
shares none. On this sample the flag's precision against real errors is zero.

So `support_rate` is NOT a metric and must not be reported as one. The plan
said "deterministic first, escalate to a judge only if too crude"; this is the
evidence that it is too crude. L2's semantic half needs an entailment judge
("does this filing support this claim?") validated against hand labels — a
narrow, checkable question, unlike "is this brief good?".

Until then the honest L2 result is tier 1 alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words carried by the filing boilerplate and by any financial sentence; they
# inflate overlap without evidencing anything.
STOPWORDS = frozenset("""
a an the and or of for to in on at by with from as is are was were be been being
this that these those it its their there here which who whom what when where how
company companies limited ltd inc corporation corp india indian exchange board
regulation regulations disclosure disclosures announcement announcements filing
filed intimation pursuant under sebi lodr listing obligations requirements
please find attached herewith enclosed submit submitted information details
respect regard shri smt mr ms dear sir madam thank you we our us they them
has have had will shall may can could would should not no yes per cent percent
""".split())

_WORD = re.compile(r"[a-z0-9]+")

# Tuned to be permissive: tier 2 is a review flag, so a false "weak_match" is
# more costly than a false "supported". Validate before treating as a metric.
SUPPORT_THRESHOLD = 0.25


@dataclass
class ClaimCheck:
    claim: str
    citation: int | None
    status: str            # fabricated_citation | uncited | supported | weak_match
    tier: int              # 1 = exact, 2 = heuristic
    cited_headline: str | None = None
    overlap: float | None = None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class AttributionResult:
    checks: list[ClaimCheck] = field(default_factory=list)
    n_filings_supplied: int = 0

    @property
    def claims(self) -> int:
        return len(self.checks)

    def _count(self, status: str) -> int:
        return sum(1 for c in self.checks if c.status == status)

    @property
    def fabricated(self) -> int:
        return self._count("fabricated_citation")

    @property
    def uncited(self) -> int:
        return self._count("uncited")

    @property
    def supported(self) -> int:
        return self._count("supported")

    @property
    def weak(self) -> int:
        return self._count("weak_match")

    @property
    def tier1_clean(self) -> bool:
        """Exact result: every claim cited a filing that was actually supplied."""
        return self.fabricated == 0 and self.uncited == 0

    @property
    def citation_validity(self) -> float | None:
        """Tier 1 score. None when the model made no claims at all."""
        if not self.checks:
            return None
        valid = self.claims - self.fabricated - self.uncited
        return valid / self.claims

    @property
    def support_rate(self) -> float | None:
        """Tier 2, HEURISTIC — NOT A METRIC. See the module docstring.

        Measured on 60 real claims: every `weak_match` it produced was a
        correct claim that happened to paraphrase. Kept only so the threshold
        can be re-measured against hand labels later; never report it.
        """
        pool = self.supported + self.weak
        return self.supported / pool if pool else None

    @property
    def l2_score(self) -> float | None:
        """The honest L2 result: tier 1 only."""
        return self.citation_validity

    def as_dict(self) -> dict:
        return {
            "claims": self.claims,
            "n_filings_supplied": self.n_filings_supplied,
            "tier1": {
                "citation_validity": self.citation_validity,
                "fabricated_citation": self.fabricated,
                "uncited": self.uncited,
                "clean": self.tier1_clean,
            },
            "tier2_heuristic_NOT_A_METRIC": {
                "support_rate": self.support_rate,
                "supported": self.supported,
                "weak_match": self.weak,
                "threshold": SUPPORT_THRESHOLD,
            },
            "checks": [c.as_dict() for c in self.checks],
        }


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower())
            if len(w) > 2 and w not in STOPWORDS}


def overlap_ratio(claim: str, headline: str) -> float:
    """Share of the claim's content words that appear in the filing.

    Asymmetric on purpose: filings are long and claims are short, so Jaccard
    would punish a perfectly supported claim for being concise.
    """
    claim_words = content_words(claim)
    if not claim_words:
        return 0.0
    return len(claim_words & content_words(headline)) / len(claim_words)


def check_attribution(drivers, filings: list) -> AttributionResult:
    """Score one brief's drivers against the filings it was shown.

    Args:
        drivers: parsed `Driver` objects (claim, citation).
        filings: the `Item`s supplied, in the order they were numbered.
    """
    result = AttributionResult(n_filings_supplied=len(filings))

    for driver in drivers or []:
        citation = driver.citation

        if citation is None:
            result.checks.append(
                ClaimCheck(driver.claim, None, "uncited", tier=1)
            )
            continue

        # Citations are 1-based as presented to the model.
        if citation < 1 or citation > len(filings):
            result.checks.append(
                ClaimCheck(driver.claim, citation, "fabricated_citation", tier=1)
            )
            continue

        headline = getattr(filings[citation - 1], "headline", "") or ""
        ratio = overlap_ratio(driver.claim, headline)
        result.checks.append(
            ClaimCheck(
                claim=driver.claim,
                citation=citation,
                status="supported" if ratio >= SUPPORT_THRESHOLD else "weak_match",
                tier=2,
                cited_headline=headline[:200],
                overlap=round(ratio, 3),
            )
        )

    return result
