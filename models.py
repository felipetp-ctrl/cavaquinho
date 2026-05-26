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
        reason: The contradicting evidence sentence when *label* is
            ``VALUE_CONTRADICTION``; ``None`` otherwise.
    """

    text: str
    evidence: str
    label: Labels
    score: float
    reason: str | None = None


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
        threshold: The decision threshold that was applied to produce
            ``is_hallucination``.  Useful for auditing results produced with
            task-specific thresholds from :data:`~cavaquinho.config.THRESHOLDS`.
    """

    score: float
    is_hallucination: bool
    claims: tuple[ClaimResult, ...]
    summary: str
    threshold: float = 0.5
