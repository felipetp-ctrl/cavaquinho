from dataclasses import dataclass
from enum import Enum

class Labels(Enum):
    VALUE_NEUTRAL = "neutral"
    VALUE_ENTAILMENT = "entailment"
    VALUE_CONTRADICTION = "contradiction"

@dataclass(frozen=True)
class ClaimResult:
    text: str
    evidence: str
    label: Labels
    score: float
    reason: str | None = None

@dataclass(frozen=True)
class ValidationResult:
    score: float
    is_hallucination: bool
    claims: list[ClaimResult]
    summary: str