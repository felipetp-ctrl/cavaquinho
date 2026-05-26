"""Abstract contract for claim extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ExtractorContract(ABC):
    """Interface that all claim extractors must implement.

    An extractor is responsible for decomposing a free-text response into a
    list of atomic, independently verifiable claim strings.  Each string
    should express a single assertion that can be evaluated against the
    provided context.

    Custom extractors can be passed to :class:`~cavaquinho.core.caco` without
    modifying any other part of the pipeline::

        from cavaquinho.extractor.base import ExtractorContract

        class MyExtractor(ExtractorContract):
            def extract(
                self, response: str, context: str, prompt: str | None = None
            ) -> list[str]:
                ...
    """

    @abstractmethod
    def extract(
        self, response: str, context: str, prompt: str | None = None
    ) -> list[str]:
        """Extract atomic claims from *response*.

        Args:
            response: The LLM-generated text to decompose.
            context: Retrieved context passed alongside the response.
                Rule-based extractors typically ignore this parameter;
                LLM-based extractors may use it to improve precision.
            prompt: Optional original user prompt.  May be used by
                advanced extractors to resolve anaphora or scope claims.

        Returns:
            A list of non-empty claim strings.  May be empty if the
            response contains no verifiable assertions.
        """
        ...
