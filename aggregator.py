from models import Labels, ClaimResult, ValidationResult
from typing import Optional


class Aggregator:
    SUPPORTED_LANGUAGES = ("en", "pt")

    def __init__(
        self,
        weights: Optional[dict] = None,
        threshold: float = 0.5,
        language: str = "en"
    ):
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Choose from: {self.SUPPORTED_LANGUAGES}"
            )

        self.threshold = threshold
        self.language = language
        self.weights = weights if weights is not None else {
            Labels.VALUE_CONTRADICTION: 1.0,
            Labels.VALUE_NEUTRAL: 0.5,
            Labels.VALUE_ENTAILMENT: 0.0,
        }

    def aggregate(self, claims: list[ClaimResult]) -> ValidationResult:
        if not claims:
            return ValidationResult(
                score=0.0,
                is_hallucination=False,
                claims=[],
                summary=self._empty_summary()
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
            claims=claims,
            summary=summary
        )

    def _empty_summary(self) -> str:
        if self.language == "pt":
            return "Nenhuma afirmação foi encontrada para verificar."
        return "No claims were found to verify."

    def _build_summary(
        self,
        claims: list[ClaimResult],
        contradictions: list[ClaimResult],
        score: float
    ) -> str:
        total = len(claims)
        n_contradictions = len(contradictions)

        if self.language == "pt":
            return self._build_summary_pt(total, n_contradictions, contradictions, score)
        return self._build_summary_en(total, n_contradictions, contradictions, score)

    def _build_summary_en(
        self,
        total: int,
        n_contradictions: int,
        contradictions: list[ClaimResult],
        score: float
    ) -> str:
        if n_contradictions == 0:
            return (
                f"No contradictions found across {total} verified claim(s). "
                f"Score: {score:.2f}."
            )

        most_severe = max(contradictions, key=lambda c: c.score)
        return (
            f"{n_contradictions} of {total} claim(s) contradict the provided context. "
            f"Main conflict: '{most_severe.text}' "
            f"contradicts evidence: '{most_severe.evidence}'. "
            f"Reason: {most_severe.reason or 'not specified'}. "
            f"Score: {score:.2f}."
        )

    def _build_summary_pt(
        self,
        total: int,
        n_contradictions: int,
        contradictions: list[ClaimResult],
        score: float
    ) -> str:
        if n_contradictions == 0:
            return (
                f"Nenhuma contradição encontrada entre {total} "
                f"afirmação(ões) verificada(s). Score: {score:.2f}."
            )

        most_severe = max(contradictions, key=lambda c: c.score)
        return (
            f"{n_contradictions} de {total} afirmação(ões) contradizem o contexto. "
            f"Principal conflito: '{most_severe.text}' "
            f"contradiz a evidência: '{most_severe.evidence}'. "
            f"Motivo: {most_severe.reason or 'não especificado'}. "
            f"Score: {score:.2f}."
        )