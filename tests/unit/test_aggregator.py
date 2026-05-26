"""Unit tests for cavaquinho.aggregator.Aggregator."""

import pytest
from cavaquinho.aggregator import Aggregator
from cavaquinho.models import ClaimResult, Labels, ValidationResult


def make_claim(label: Labels, score: float = 0.9, reason: str | None = None) -> ClaimResult:
    return ClaimResult(
        text="some claim",
        evidence="some evidence",
        label=label,
        score=score,
        reason=reason,
    )


class TestAggregatorInit:
    def test_default_threshold(self):
        a = Aggregator()
        assert a.threshold == 0.5

    def test_custom_threshold(self):
        a = Aggregator(threshold=0.7)
        assert a.threshold == 0.7

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            Aggregator(language="klingon")

    def test_supported_languages(self):
        Aggregator(language="english")
        Aggregator(language="portuguese")

    def test_custom_weights(self):
        weights = {Labels.VALUE_CONTRADICTION: 0.8, Labels.VALUE_NEUTRAL: 0.2, Labels.VALUE_ENTAILMENT: 0.0}
        a = Aggregator(weights=weights)
        assert a.weights[Labels.VALUE_CONTRADICTION] == 0.8


class TestAggregateEmpty:
    def test_empty_list_returns_zero_score(self):
        result = Aggregator().aggregate([])
        assert result.score == 0.0
        assert result.is_hallucination is False
        assert result.claims == ()

    def test_empty_list_summary_english(self):
        result = Aggregator(language="english").aggregate([])
        assert "No claims" in result.summary

    def test_empty_list_summary_portuguese(self):
        result = Aggregator(language="portuguese").aggregate([])
        assert "Nenhuma" in result.summary


class TestAggregateScoring:
    def test_all_entailment_score_is_zero(self):
        claims = [make_claim(Labels.VALUE_ENTAILMENT, score=0.99) for _ in range(3)]
        result = Aggregator().aggregate(claims)
        assert result.score == 0.0
        assert result.is_hallucination is False

    def test_all_contradiction_score_is_high(self):
        claims = [make_claim(Labels.VALUE_CONTRADICTION, score=1.0) for _ in range(3)]
        result = Aggregator().aggregate(claims)
        assert result.score == 1.0
        assert result.is_hallucination is True

    def test_neutral_contributes_half_weight(self):
        # single neutral claim with score=1.0 → weighted = 0.5 * 1.0 = 0.5
        claims = [make_claim(Labels.VALUE_NEUTRAL, score=1.0)]
        result = Aggregator(threshold=0.6).aggregate(claims)
        assert result.score == 0.5
        assert result.is_hallucination is False

    def test_threshold_boundary_above(self):
        claims = [make_claim(Labels.VALUE_CONTRADICTION, score=0.8)]
        # weighted score = 1.0 * 0.8 = 0.8 > threshold=0.5
        result = Aggregator(threshold=0.5).aggregate(claims)
        assert result.is_hallucination is True

    def test_threshold_boundary_below(self):
        claims = [make_claim(Labels.VALUE_CONTRADICTION, score=0.4)]
        # weighted score = 1.0 * 0.4 = 0.4 < threshold=0.5
        result = Aggregator(threshold=0.5).aggregate(claims)
        assert result.is_hallucination is False

    def test_score_is_rounded_to_4_decimals(self):
        claims = [make_claim(Labels.VALUE_NEUTRAL, score=1 / 3)]
        result = Aggregator().aggregate(claims)
        assert result.score == round(0.5 * (1 / 3), 4)

    def test_mixed_claims_average(self):
        claims = [
            make_claim(Labels.VALUE_CONTRADICTION, score=1.0),  # contributes 1.0
            make_claim(Labels.VALUE_ENTAILMENT, score=1.0),     # contributes 0.0
        ]
        result = Aggregator().aggregate(claims)
        assert result.score == 0.5  # (1.0 + 0.0) / 2

    def test_returns_tuple_of_claims(self):
        claims = [make_claim(Labels.VALUE_ENTAILMENT)]
        result = Aggregator().aggregate(claims)
        assert isinstance(result.claims, tuple)
        assert len(result.claims) == 1


class TestAggregateSummary:
    def test_no_contradiction_summary_english(self):
        claims = [make_claim(Labels.VALUE_ENTAILMENT)]
        result = Aggregator(language="english").aggregate(claims)
        assert "No contradictions" in result.summary
        assert "1" in result.summary

    def test_no_contradiction_summary_portuguese(self):
        claims = [make_claim(Labels.VALUE_ENTAILMENT)]
        result = Aggregator(language="portuguese").aggregate(claims)
        assert "Nenhuma contradição" in result.summary

    def test_contradiction_summary_english_includes_claim_text(self):
        claim = ClaimResult(
            text="The sky is green.",
            evidence="The sky is blue.",
            label=Labels.VALUE_CONTRADICTION,
            score=0.95,
            reason="The sky is blue.",
        )
        result = Aggregator(language="english").aggregate([claim])
        assert "The sky is green." in result.summary
        assert "The sky is blue." in result.summary

    def test_contradiction_summary_portuguese(self):
        claim = ClaimResult(
            text="O céu é verde.",
            evidence="O céu é azul.",
            label=Labels.VALUE_CONTRADICTION,
            score=0.9,
            reason="O céu é azul.",
        )
        result = Aggregator(language="portuguese").aggregate([claim])
        assert "O céu é verde." in result.summary

    def test_summary_no_reason_fallback_english(self):
        claim = ClaimResult(
            text="claim",
            evidence="evidence",
            label=Labels.VALUE_CONTRADICTION,
            score=0.9,
            reason=None,
        )
        result = Aggregator(language="english").aggregate([claim])
        assert "no specific evidence identified" in result.summary

    def test_summary_no_reason_fallback_portuguese(self):
        claim = ClaimResult(
            text="afirmação",
            evidence="evidência",
            label=Labels.VALUE_CONTRADICTION,
            score=0.9,
            reason=None,
        )
        result = Aggregator(language="portuguese").aggregate([claim])
        assert "nenhuma evidência específica identificada" in result.summary

    def test_most_severe_contradiction_used_in_summary(self):
        low = ClaimResult(text="low claim", evidence="e", label=Labels.VALUE_CONTRADICTION, score=0.3, reason="r")
        high = ClaimResult(text="high claim", evidence="e", label=Labels.VALUE_CONTRADICTION, score=0.9, reason="r")
        result = Aggregator().aggregate([low, high])
        assert "high claim" in result.summary
        assert "low claim" not in result.summary
