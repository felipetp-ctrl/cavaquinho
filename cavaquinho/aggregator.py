"""Aggregation of per-claim classification results into a final verdict."""

from __future__ import annotations

from typing import Optional

from .config import DEFAULT_THRESHOLD, SUPPORTED_LANGUAGES
from .models import ClaimResult, Labels, ValidationResult

# Per-language summary templates.
# Keys: "no_contradiction", "with_contradiction", "main_conflict",
#       "no_reason", "empty"
_TEMPLATES: dict[str, dict[str, str]] = {
    "english": {
        "no_contradiction": "No contradictions found across {total} verified claim(s). Score: {score:.2f}.",
        "with_contradiction": (
            "{n_contradictions} of {total} claim(s) contradict the provided context. "
            "Main conflict: '{claim_text}' — {reason_phrase}. Score: {score:.2f}."
        ),
        "reason_phrase": "contradicting evidence: '{reason}'",
        "no_reason": "no specific evidence identified",
        "empty": "No claims were found to verify.",
    },
    "portuguese": {
        "no_contradiction": (
            "Nenhuma contradição encontrada entre {total} afirmação(ões) verificada(s). Score: {score:.2f}."
        ),
        "with_contradiction": (
            "{n_contradictions} de {total} afirmação(ões) contradizem o contexto. "
            "Principal conflito: '{claim_text}' — {reason_phrase}. Score: {score:.2f}."
        ),
        "reason_phrase": "evidência contraditória: '{reason}'",
        "no_reason": "nenhuma evidência específica identificada",
        "empty": "Nenhuma afirmação foi encontrada para verificar.",
    },
}


class Aggregator:
    """Combines per-claim NLI results into a single :class:`~cavaquinho.models.ValidationResult`.

    **Scoring semantics** — the aggregator computes a *weighted mean* over all
    claims, where each claim contributes its NLI confidence score multiplied
    by a label weight:

    - ``VALUE_CONTRADICTION`` → 1.0 (full contribution)
    - ``VALUE_NEUTRAL``       → 0.5 (partial contribution)
    - ``VALUE_ENTAILMENT``    → 0.0 (no contribution)

    The denominator is the **total number of claims**, so a single
    contradiction in a long response will yield a proportionally low score.
    This is intentional: the library treats faithfulness as a property of the
    whole response — a response that is 90% correct and 10% contradictory is
    meaningfully different from one that is entirely contradictory.  Callers
    that need to flag *any* contradiction should inspect
    ``result.claims`` directly.

    Args:
        weights: Optional mapping of :class:`~cavaquinho.models.Labels` to
            their contribution weights.  Defaults to the three-class weights
            described above.
        threshold: Score threshold above which the response is considered a
            hallucination.  Must be in ``(0.0, 1.0)``.
        language: Localisation language for the generated summary string.
            Must be one of :data:`~cavaquinho.config.SUPPORTED_LANGUAGES`.

    Raises:
        ValueError: If *language* is not supported.
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        threshold: float = DEFAULT_THRESHOLD,
        language: str = "english",
    ):
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Choose from: {SUPPORTED_LANGUAGES}"
            )

        self.threshold = threshold
        self.language = language
        self.weights = weights if weights is not None else {
            Labels.VALUE_CONTRADICTION: 1.0,
            Labels.VALUE_NEUTRAL: 0.5,
            Labels.VALUE_ENTAILMENT: 0.0,
        }

    def aggregate(self, claims: list[ClaimResult]) -> ValidationResult:
        """Aggregate a list of claim results into a single validation verdict.

        Args:
            claims: Per-claim results produced by the classifier stage.

        Returns:
            A :class:`~cavaquinho.models.ValidationResult` with the computed
            score, hallucination flag, original claims, and a localised summary.
        """
        if not claims:
            return ValidationResult(
                score=0.0,
                is_hallucination=False,
                claims=(),
                summary=_TEMPLATES[self.language]["empty"],
            )

        contradiction_claims: list[ClaimResult] = []
        weighted_sum = 0.0

        for claim in claims:
            weighted_sum += claim.score * self.weights.get(claim.label, 0.0)
            if claim.label == Labels.VALUE_CONTRADICTION:
                contradiction_claims.append(claim)

        final_score = weighted_sum / len(claims)
        is_hallucination = final_score > self.threshold
        summary = self._build_summary(claims, contradiction_claims, final_score)

        return ValidationResult(
            score=round(final_score, 4),
            is_hallucination=is_hallucination,
            claims=tuple(claims),
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        claims: list[ClaimResult],
        contradictions: list[ClaimResult],
        score: float,
    ) -> str:
        t = _TEMPLATES[self.language]
        total = len(claims)
        n_contradictions = len(contradictions)

        if n_contradictions == 0:
            return t["no_contradiction"].format(total=total, score=score)

        most_severe = max(contradictions, key=lambda c: c.score)
        reason_phrase = (
            t["reason_phrase"].format(reason=most_severe.reason)
            if most_severe.reason
            else t["no_reason"]
        )
        return t["with_contradiction"].format(
            n_contradictions=n_contradictions,
            total=total,
            claim_text=most_severe.text,
            reason_phrase=reason_phrase,
            score=score,
        )
