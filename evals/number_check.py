"""L1: are the numbers the model cited actually the numbers it was given?

This is the check the rest of the portfolio does not have. The SEBI RAG
project's verifier asks whether a figure appears *anywhere* in the source text
— a substring test. That has two failure modes this avoids:

**Rounding false-positives.** "Rs 20.4T" does not appear as a substring of
`20438050139042.25`, so an honest rounding is flagged as a hallucination.
Here, a cited value matches if it is within a relative tolerance, so rounding
passes and fabrication does not.

**Field-blind false-negatives.** If the model writes "P/E of 3020" and 3020
appears in the corpus as the *share price*, a substring test passes it. Here
every number is checked against the field it was claimed for, so a right number
attached to the wrong label fails — which is the more dangerous error, because
it reads as authoritative.

Three outcomes, deliberately distinct:
  - matched     : label exists, value agrees within tolerance
  - wrong       : label exists, value disagrees  -> a real hallucination
  - unverifiable: label not in the snapshot      -> nothing to check against

`unverifiable` is not folded into `wrong`. A model inventing a *label* is a
different failure from a model inventing a *value*, and a score that merges
them cannot tell you which is happening.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_TOLERANCE = 0.01      # 1% relative — covers ordinary rounding
ABSOLUTE_FLOOR = 1e-9         # guards division when a truth value is ~0


@dataclass
class FigureCheck:
    label: str
    cited: float
    truth: float | None
    status: str               # "matched" | "wrong" | "unverifiable"
    relative_error: float | None = None

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class NumberCheckResult:
    checks: list[FigureCheck] = field(default_factory=list)
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def cited(self) -> int:
        return len(self.checks)

    @property
    def matched(self) -> int:
        return sum(1 for c in self.checks if c.status == "matched")

    @property
    def wrong(self) -> int:
        return sum(1 for c in self.checks if c.status == "wrong")

    @property
    def unverifiable(self) -> int:
        return sum(1 for c in self.checks if c.status == "unverifiable")

    @property
    def score(self) -> float | None:
        """Share of *checkable* citations that were right.

        None when the model cited nothing checkable — an honest "no signal",
        not a zero. Scoring silence as failure would reward verbosity and
        punish a model that sensibly declined to quote figures.
        """
        checkable = self.matched + self.wrong
        if checkable == 0:
            return None
        return self.matched / checkable

    @property
    def clean(self) -> bool:
        """No fabricated values and no invented labels."""
        return self.wrong == 0 and self.unverifiable == 0

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "cited": self.cited,
            "matched": self.matched,
            "wrong": self.wrong,
            "unverifiable": self.unverifiable,
            "clean": self.clean,
            "tolerance": self.tolerance,
            "checks": [c.as_dict() for c in self.checks],
        }


def _decimals(value: float) -> int:
    """Decimal places the model actually wrote, capped at a sane maximum."""
    text = repr(float(value))
    if "e" in text or "E" in text or "." not in text:
        return 0
    return min(len(text.split(".", 1)[1]), 6)


def _agrees(cited: float, truth: float, tolerance: float) -> tuple[bool, float]:
    """Match on relative tolerance OR on correct rounding.

    Relative tolerance alone is wrong for small-magnitude figures, and the
    benchmark caught it in its own first run: a model told to round to two
    decimals wrote `change_pct_1d: 0.42` against a truth of 0.4248. That is a
    1.13% relative error — over a 1% tolerance — yet it is exactly the rounding
    the contract asked for. Scoring it as a hallucination would have measured
    arithmetic precision, not honesty, and would have biased every percentage
    field where values sit near zero.

    So a citation also matches when it equals the truth rounded to however many
    decimals the model wrote. Fabrication still fails: 1510.32 is not
    round(3020.65, 2) at any precision.
    """
    denominator = max(abs(truth), ABSOLUTE_FLOOR)
    error = abs(cited - truth) / denominator
    if error <= tolerance:
        return True, error
    if cited == round(truth, _decimals(cited)):
        return True, error
    return False, error


def check_numbers(
    cited_figures: dict[str, float],
    truth_figures: dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> NumberCheckResult:
    """Compare every cited figure against the snapshot it was drawn from.

    Args:
        cited_figures: {label: value} parsed from the brief.
        truth_figures: `MarketSnapshot.figures()` for the same (ticker, date).
        tolerance:     relative tolerance for a match.
    """
    result = NumberCheckResult(tolerance=tolerance)

    for label, cited in (cited_figures or {}).items():
        truth = (truth_figures or {}).get(label)
        if truth is None:
            result.checks.append(
                FigureCheck(label=label, cited=cited, truth=None, status="unverifiable")
            )
            continue
        ok, error = _agrees(cited, float(truth), tolerance)
        result.checks.append(
            FigureCheck(
                label=label,
                cited=cited,
                truth=float(truth),
                status="matched" if ok else "wrong",
                relative_error=error,
            )
        )

    return result
