"""Data models shared across the cavaquinho pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Labels(Enum):
    """NLI classification labels produced by the classifier stage."""

    VALUE_NEUTRAL = "neutral"
    VALUE_ENTAILMENT = "entailment"
    VALUE_CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class ClaimResult:
    """Verification result for a single atomic claim.

    Attributes:
        text: The extracted claim sentence.
        evidence: The context sentence selected as the most relevant
            counterpart during classification. When a contradiction is
            detected this is the sentence that contradicts the claim;
            otherwise it is the context sentence with the highest model
            confidence regardless of label.
        label: NLI label assigned by the classifier.
        score: Model confidence for the assigned label, in [0.0, 1.0].
            Rounded to 4 decimal places.
        reason: The contradicting evidence sentence when *label* is
            ``VALUE_CONTRADICTION``; ``None`` otherwise.
    """

    text: str
    evidence: str
    label: Labels
    score: float
    reason: str | None = None

    def __repr__(self) -> str:
        text = self.text if len(self.text) <= 60 else self.text[:57] + "..."
        return f"ClaimResult(label={self.label.value.upper()}, score={self.score}, text={text!r})"


@dataclass(frozen=True)
class ValidationResult:
    """Aggregated faithfulness result for a full response.

    Attributes:
        score: Weighted aggregate hallucination score in [0.0, 1.0].
            Higher values indicate more conflict with the provided context.
        is_hallucination: ``True`` when *score* exceeds the configured
            threshold.
        claims: Immutable sequence of per-claim results in extraction order.
        summary: Human-readable description of the result, localised to the
            configured language.
    """

    score: float
    is_hallucination: bool
    claims: tuple[ClaimResult, ...]
    summary: str

    def __repr__(self) -> str:
        flag = "HALLUCINATION" if self.is_hallucination else "OK"
        return f"ValidationResult(score={self.score}, {flag}, claims={len(self.claims)})"

    def __str__(self) -> str:
        return self.summary
