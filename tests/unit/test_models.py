"""Unit tests for cavaquinho.models."""

import pytest
from cavaquinho.models import ClaimResult, Labels, ValidationResult


class TestLabels:
    def test_values(self):
        assert Labels.VALUE_NEUTRAL.value == "neutral"
        assert Labels.VALUE_ENTAILMENT.value == "entailment"
        assert Labels.VALUE_CONTRADICTION.value == "contradiction"

    def test_enum_identity(self):
        assert Labels.VALUE_CONTRADICTION is Labels.VALUE_CONTRADICTION


class TestClaimResult:
    def test_basic_construction(self):
        cr = ClaimResult(
            text="Paris is the capital of France.",
            evidence="France's capital city is Paris.",
            label=Labels.VALUE_ENTAILMENT,
            score=0.95,
        )
        assert cr.text == "Paris is the capital of France."
        assert cr.label == Labels.VALUE_ENTAILMENT
        assert cr.score == 0.95
        assert cr.reason is None

    def test_reason_optional(self):
        cr = ClaimResult(
            text="claim",
            evidence="evidence",
            label=Labels.VALUE_CONTRADICTION,
            score=0.9,
            reason="contradicting evidence sentence",
        )
        assert cr.reason == "contradicting evidence sentence"

    def test_frozen(self):
        cr = ClaimResult(text="c", evidence="e", label=Labels.VALUE_NEUTRAL, score=0.5)
        with pytest.raises((AttributeError, TypeError)):
            cr.score = 0.1  # type: ignore[misc]

    def test_score_stored_as_given(self):
        cr = ClaimResult(text="c", evidence="e", label=Labels.VALUE_NEUTRAL, score=0.1234)
        assert cr.score == 0.1234


class TestValidationResult:
    def _make_claim(self, label=Labels.VALUE_ENTAILMENT, score=0.9):
        return ClaimResult(text="t", evidence="e", label=label, score=score)

    def test_construction(self):
        claim = self._make_claim()
        vr = ValidationResult(
            score=0.1,
            is_hallucination=False,
            claims=(claim,),
            summary="All good.",
        )
        assert vr.score == 0.1
        assert not vr.is_hallucination
        assert len(vr.claims) == 1
        assert vr.summary == "All good."

    def test_claims_is_tuple(self):
        vr = ValidationResult(score=0.0, is_hallucination=False, claims=(), summary="")
        assert isinstance(vr.claims, tuple)

    def test_frozen(self):
        vr = ValidationResult(score=0.0, is_hallucination=False, claims=(), summary="")
        with pytest.raises((AttributeError, TypeError)):
            vr.score = 1.0  # type: ignore[misc]

    def test_claims_tuple_is_immutable(self):
        vr = ValidationResult(score=0.0, is_hallucination=False, claims=(), summary="")
        with pytest.raises(AttributeError):
            vr.claims.append(self._make_claim())  # type: ignore[attr-defined]
