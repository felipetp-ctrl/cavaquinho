"""Unit tests for cavaquinho.core.caco — all model calls are mocked."""

import pytest
from unittest.mock import MagicMock, patch
from cavaquinho.core import caco
from cavaquinho.models import ClaimResult, Labels, ValidationResult


def _make_claim(label=Labels.VALUE_ENTAILMENT, score=0.9, text="claim", reason=None):
    return ClaimResult(text=text, evidence="evidence", label=label, score=score, reason=reason)


def _make_validator(claims_out=None, aggregate_out=None):
    """Build a caco instance with fully mocked extractor, classifier, and aggregator."""
    extractor = MagicMock()
    extractor.extract.return_value = ["claim one", "claim two"]

    classifier = MagicMock()
    default_claim = _make_claim()
    classifier.classify_batch.side_effect = lambda claims, context: [default_claim] * len(claims)

    aggregator = MagicMock()
    aggregator.aggregate.return_value = aggregate_out or ValidationResult(
        score=0.0,
        is_hallucination=False,
        claims=(_make_claim(),),
        summary="ok",
    )

    validator = caco(extractor=extractor, classifier=classifier, aggregator=aggregator)
    return validator, extractor, classifier, aggregator


class TestCacoInit:
    def test_defaults_are_set(self):
        with patch("cavaquinho.core.DeBERTaClassifier"), \
             patch("cavaquinho.core.RuleExtractor"), \
             patch("cavaquinho.core.Aggregator") as MockAgg:
            validator = caco(threshold=0.7, language="portuguese")
            assert validator.threshold == 0.7
            assert validator.language == "portuguese"
            # threshold must be forwarded to Aggregator
            MockAgg.assert_called_once_with(threshold=0.7, language="portuguese")

    def test_custom_components_accepted(self):
        ext, clf, agg = MagicMock(), MagicMock(), MagicMock()
        agg.aggregate.return_value = ValidationResult(
            score=0.0, is_hallucination=False, claims=(), summary=""
        )
        clf.classify_batch.return_value = []
        ext.extract.return_value = []
        v = caco(extractor=ext, classifier=clf, aggregator=agg)
        assert v.extractor is ext
        assert v.classifier is clf
        assert v.aggregator is agg


class TestCacoValidate:
    def test_empty_response_raises(self):
        v, *_ = _make_validator()
        with pytest.raises(ValueError, match="Empty response"):
            v.validate(response="   ", context="some context")

    def test_empty_context_raises(self):
        v, *_ = _make_validator()
        with pytest.raises(ValueError, match="Empty context"):
            v.validate(response="some response", context="   ")

    def test_pipeline_calls_extractor(self):
        v, ext, clf, agg = _make_validator()
        v.validate(response="Hello world.", context="Context here.")
        ext.extract.assert_called_once_with("Hello world.", "Context here.", None)

    def test_pipeline_passes_prompt(self):
        v, ext, clf, agg = _make_validator()
        v.validate(response="Hello.", context="Context.", prompt="user question")
        ext.extract.assert_called_once_with("Hello.", "Context.", "user question")

    def test_classifier_called_for_each_claim(self):
        v, ext, clf, agg = _make_validator()
        ext.extract.return_value = ["claim A", "claim B", "claim C"]
        v.validate(response="r", context="c")
        clf.classify_batch.assert_called_once_with(["claim A", "claim B", "claim C"], "c")

    def test_classifier_receives_full_context(self):
        v, ext, clf, agg = _make_validator()
        ext.extract.return_value = ["claim"]
        v.validate(response="r", context="the full context string")
        clf.classify_batch.assert_called_once_with(["claim"], "the full context string")

    def test_aggregator_receives_claim_results(self):
        v, ext, clf, agg = _make_validator()
        claim_result = _make_claim(label=Labels.VALUE_CONTRADICTION)
        clf.classify_batch.side_effect = lambda claims, ctx: [claim_result] * len(claims)
        ext.extract.return_value = ["c1", "c2"]
        v.validate(response="r", context="c")
        args, _ = agg.aggregate.call_args
        assert all(cr is claim_result for cr in args[0])

    def test_returns_validation_result(self):
        v, *_ = _make_validator()
        result = v.validate(response="response", context="context")
        assert isinstance(result, ValidationResult)

    def test_no_claims_extracted_still_calls_aggregator(self):
        v, ext, clf, agg = _make_validator()
        ext.extract.return_value = []
        clf.classify_batch.return_value = []
        agg.aggregate.return_value = ValidationResult(
            score=0.0, is_hallucination=False, claims=(), summary="No claims."
        )
        result = v.validate(response="r", context="c")
        agg.aggregate.assert_called_once_with([])
        assert result.claims == ()


class TestCacoThresholdPropagation:
    def test_threshold_forwarded_to_aggregator_constructor(self):
        """Regression test: threshold configured on caco must reach the Aggregator."""
        with patch("cavaquinho.core.DeBERTaClassifier"), \
             patch("cavaquinho.core.RuleExtractor"), \
             patch("cavaquinho.core.Aggregator") as MockAgg:
            MockAgg.return_value = MagicMock()
            MockAgg.return_value.aggregate.return_value = ValidationResult(
                score=0.0, is_hallucination=False, claims=(), summary=""
            )
            caco(threshold=0.9)
            _, kwargs = MockAgg.call_args
            assert kwargs.get("threshold") == 0.9
