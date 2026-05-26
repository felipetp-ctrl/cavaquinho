"""Sentence-boundary claim extractor using NLTK."""

from __future__ import annotations

import nltk
from nltk.tokenize import sent_tokenize

from ..config import SUPPORTED_LANGUAGES
from .base import ExtractorContract


class RuleExtractor(ExtractorContract):
    """Extracts claims by splitting the response into sentences with NLTK.

    Each sentence is treated as an independent, verifiable claim.  This is
    the default extractor — it requires no external API and has no
    per-request latency beyond the tokenisation itself.

    Note that sentence tokenisation is not the same as atomic claim
    decomposition: a single sentence may contain multiple verifiable facts.
    For higher-precision decomposition, use
    :class:`~cavaquinho.extractor.llm_extractor.LLMExtractor`.

    Args:
        language: Language passed to :func:`nltk.tokenize.sent_tokenize`.
            Must be one of :data:`~cavaquinho.config.SUPPORTED_LANGUAGES`.

    Raises:
        ValueError: If *language* is not supported.
    """

    def __init__(self, language: str = "english"):
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Choose from: {SUPPORTED_LANGUAGES}"
            )
        self.language = language
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    def extract(
        self, response: str, context: str = "", prompt: str | None = None
    ) -> list[str]:
        """Split *response* into sentences and return them as claims.

        Args:
            response: The LLM-generated text to split.
            context: Unused by this extractor; accepted for interface
                compatibility.
            prompt: Unused by this extractor; accepted for interface
                compatibility.

        Returns:
            Non-empty, stripped sentence strings.
        """
        sentences = sent_tokenize(response, language=self.language)
        return [s.strip() for s in sentences if s.strip()]
