"""Unit tests for cavaquinho extractors."""

import json
from unittest.mock import MagicMock

import pytest

from cavaquinho.extractor.llm_extractor import LLMExtractor
from cavaquinho.extractor.rule_extractor import RuleExtractor


class TestRuleExtractor:
    def test_single_sentence(self):
        ex = RuleExtractor()
        result = ex.extract("The sky is blue.")
        assert result == ["The sky is blue."]

    def test_multiple_sentences(self):
        ex = RuleExtractor()
        result = ex.extract("The sky is blue. Water is wet. Fire is hot.")
        assert len(result) == 3
        assert "The sky is blue." in result

    def test_strips_whitespace(self):
        ex = RuleExtractor()
        result = ex.extract("  Hello world.  ")
        assert all(s == s.strip() for s in result)

    def test_empty_string_returns_empty(self):
        ex = RuleExtractor()
        result = ex.extract("")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        ex = RuleExtractor()
        result = ex.extract("   \n\t  ")
        assert result == []

    def test_portuguese_language(self):
        ex = RuleExtractor(language="portuguese")
        result = ex.extract("O céu é azul. A água é molhada.")
        assert len(result) == 2

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            RuleExtractor(language="klingon")

    def test_context_and_prompt_ignored(self):
        ex = RuleExtractor()
        r1 = ex.extract("The sky is blue.", context="ignored context")
        r2 = ex.extract("The sky is blue.", context="different context", prompt="some prompt")
        assert r1 == r2

    def test_newline_separated_sentences(self):
        ex = RuleExtractor()
        result = ex.extract("First sentence.\nSecond sentence.")
        assert len(result) >= 1  # NLTK may or may not split on newlines; at least one claim


class TestLLMExtractor:
    def _make_llm_fn(self, response: str):
        return MagicMock(return_value=response)

    def test_parses_json_array(self):
        llm_fn = self._make_llm_fn('["Claim one.", "Claim two."]')
        ex = LLMExtractor(llm_fn=llm_fn)
        result = ex.extract("some response")
        assert result == ["Claim one.", "Claim two."]

    def test_strips_markdown_fences(self):
        llm_fn = self._make_llm_fn('```\n["Claim one."]\n```')
        ex = LLMExtractor(llm_fn=llm_fn)
        result = ex.extract("some response")
        assert result == ["Claim one."]

    def test_strips_json_fences(self):
        llm_fn = self._make_llm_fn('```json\n["Claim one."]\n```')
        ex = LLMExtractor(llm_fn=llm_fn)
        result = ex.extract("some response")
        assert result == ["Claim one."]

    def test_max_claims_truncation(self):
        claims = [f"Claim {i}." for i in range(30)]
        llm_fn = self._make_llm_fn(json.dumps(claims))
        ex = LLMExtractor(llm_fn=llm_fn, max_claims=5)
        result = ex.extract("some response")
        assert len(result) == 5

    def test_empty_strings_filtered(self):
        llm_fn = self._make_llm_fn('["Valid claim.", "", "  ", "Another claim."]')
        ex = LLMExtractor(llm_fn=llm_fn)
        result = ex.extract("some response")
        assert result == ["Valid claim.", "Another claim."]

    def test_falls_back_on_json_error(self):
        llm_fn = self._make_llm_fn("not valid json at all")
        ex = LLMExtractor(llm_fn=llm_fn)
        result = ex.extract("The sky is blue. Water is wet.")
        # fallback produces sentence-split result
        assert len(result) >= 1
        assert all(isinstance(c, str) for c in result)

    def test_falls_back_on_non_list_json(self):
        llm_fn = self._make_llm_fn('{"key": "value"}')
        ex = LLMExtractor(llm_fn=llm_fn)
        result = ex.extract("The sky is blue.")
        assert isinstance(result, list)

    def test_falls_back_on_llm_fn_exception(self):
        llm_fn = MagicMock(side_effect=RuntimeError("API error"))
        ex = LLMExtractor(llm_fn=llm_fn)
        result = ex.extract("The sky is blue.")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_fallback_logs_warning(self, caplog):
        import logging
        llm_fn = MagicMock(side_effect=RuntimeError("boom"))
        ex = LLMExtractor(llm_fn=llm_fn)
        with caplog.at_level(logging.WARNING, logger="cavaquinho.extractor.llm_extractor"):
            ex.extract("The sky is blue.")
        assert any("RuntimeError" in r.message for r in caplog.records)

    def test_prompt_template_contains_response(self):
        captured = []
        def llm_fn(p):
            captured.append(p)
            return '["claim"]'
        ex = LLMExtractor(llm_fn=llm_fn)
        ex.extract("My specific response text.")
        assert "My specific response text." in captured[0]

    def test_default_max_claims_is_20(self):
        ex = LLMExtractor(llm_fn=lambda p: "[]")
        assert ex.max_claims == 20


class TestDocumentedLimitations:
    """Regression tests that pin known limitations so they cannot silently break."""

    def test_implicit_negation_is_kept_as_single_claim(self):
        # RuleExtractor splits on sentence boundaries, not semantic negation.
        # "The LGPD was not enacted in 2015." must be kept whole, not split.
        ex = RuleExtractor()
        result = ex.extract("The LGPD was not enacted in 2015.")
        assert len(result) == 1
        assert "not" in result[0]

    def test_empty_context_does_not_affect_extraction(self):
        # Extraction is context-independent; empty context must not raise.
        ex = RuleExtractor()
        result = ex.extract("The sky is blue.", context="")
        assert result == ["The sky is blue."]

    def test_response_with_no_verifiable_claims_returns_list(self):
        # Even a response that is purely opinion must return a list (possibly empty).
        ex = RuleExtractor()
        result = ex.extract("I think this is nice.")
        assert isinstance(result, list)

    def test_llm_extractor_empty_response_returns_empty_list(self):
        ex = LLMExtractor(llm_fn=lambda p: "[]")
        result = ex.extract("  ")
        assert result == []
