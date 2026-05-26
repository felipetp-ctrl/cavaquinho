"""Aggregation of per-claim classification results into a final verdict."""

from __future__ import annotations

from typing import Optional

from .config import DEFAULT_THRESHOLD, SUPPORTED_LANGUAGES
from .models import ClaimResult, Labels, ValidationResult


class Aggregator:
    """Combines per-claim NLI results into a single :class:`~cavaquinho.models.ValidationResult`.

    The aggregation formula computes a weighted average of per-claim scores,
    where each label class carries a configurable weight:

    - ``VALUE_CONTRADICTION`` → 1.0  (full contribution to hallucination score)
    - ``VALUE_NEUTRAL``       → 0.5  (partial contribution)
    - ``VALUE_ENTAILMENT``    → 0.0  (no contribution)

    The final score is compared against *threshold* to produce the binary
    ``is_hallucination`` decision.

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
                summary=self._empty_summary(),
                threshold=self.threshold,
            )

        claim_avg: list[float] = []
        contradiction_claims: list[ClaimResult] = []

        for claim in claims:
            label = claim.label
            score = claim.score
            weighted_avg = score * self.weights.get(label, 0.0)
            claim_avg.append(weighted_avg)

            if label == Labels.VALUE_CONTRADICTION:
                contradiction_claims.append(claim)

        final_score = sum(claim_avg) / len(claim_avg)
        is_hallucination = final_score > self.threshold
        summary = self._build_summary(claims, contradiction_claims, final_score)

        return ValidationResult(
            score=round(final_score, 4),
            is_hallucination=is_hallucination,
            claims=tuple(claims),
            summary=summary,
            threshold=self.threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_summary(self) -> str:
        if self.language == "portuguese":
            return "Nenhuma afirmação foi encontrada para verificar."
        return "No claims were found to verify."

    def _build_summary(
        self,
        claims: list[ClaimResult],
        contradictions: list[ClaimResult],
        score: float,
    ) -> str:
        total = len(claims)
        n_contradictions = len(contradictions)

        if self.language == "portuguese":
            return self._build_summary_pt(total, n_contradictions, contradictions, score)
        return self._build_summary_en(total, n_contradictions, contradictions, score)

    def _build_summary_en(
        self,
        total: int,
        n_contradictions: int,
        contradictions: list[ClaimResult],
        score: float,
    ) -> str:
        if n_contradictions == 0:
            return (
                f"No contradictions found across {total} verified claim(s). "
                f"Score: {score:.2f}."
            )

        most_severe = max(contradictions, key=lambda c: c.score)
        reason_text = (
            f"contradicting evidence: '{most_severe.reason}'"
            if most_severe.reason
            else "no specific evidence identified"
        )
        return (
            f"{n_contradictions} of {total} claim(s) contradict the provided context. "
            f"Main conflict: '{most_severe.text}' — {reason_text}. "
            f"Score: {score:.2f}."
        )

    def _build_summary_pt(
        self,
        total: int,
        n_contradictions: int,
        contradictions: list[ClaimResult],
        score: float,
    ) -> str:
        if n_contradictions == 0:
            return (
                f"Nenhuma contradição encontrada entre {total} "
                f"afirmação(ões) verificada(s). Score: {score:.2f}."
            )

        most_severe = max(contradictions, key=lambda c: c.score)
        reason_text = (
            f"evidência contraditória: '{most_severe.reason}'"
            if most_severe.reason
            else "nenhuma evidência específica identificada"
        )
        return (
            f"{n_contradictions} de {total} afirmação(ões) contradizem o contexto. "
            f"Principal conflito: '{most_severe.text}' — {reason_text}. "
            f"Score: {score:.2f}."
        )
