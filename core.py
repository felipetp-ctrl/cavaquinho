"""Main entry point — the :class:`caco` validator."""

from __future__ import annotations

from .aggregator import Aggregator
from .classifier.deberta import DeBERTaClassifier
from .config import DEFAULT_LANGUAGE, DEFAULT_THRESHOLD, THRESHOLDS
from .extractor.rule_extractor import RuleExtractor
from .models import ValidationResult


class caco:
    """Faithfulness hallucination detector for LLM responses.

    Orchestrates three sequential stages:

    1. **Extraction** — decomposes the response into atomic claim strings.
    2. **Classification** — runs NLI inference on each claim against the
       provided context (concurrently via a thread pool).
    3. **Aggregation** — combines per-claim results into a single
       :class:`~cavaquinho.models.ValidationResult`.

    Each stage is independently replaceable via the *extractor*,
    *classifier*, and *aggregator* parameters.  See
    :class:`~cavaquinho.extractor.base.ExtractorContract` and
    :class:`~cavaquinho.classifier.base.ClassifierContract` for the
    interfaces required by custom implementations.

    Args:
        extractor: Claim extractor instance.  Defaults to
            :class:`~cavaquinho.extractor.rule_extractor.RuleExtractor`.
        classifier: NLI classifier instance.  Defaults to
            :class:`~cavaquinho.classifier.deberta.DeBERTaClassifier`.
            Requires ``pip install "cavaquinho[nli]"``.
        aggregator: Aggregator instance.  Defaults to
            :class:`~cavaquinho.aggregator.Aggregator`.
        threshold: Hallucination score threshold in ``(0.0, 1.0)``.
            Responses whose aggregated score exceeds this value are flagged
            as hallucinations.  Defaults to ``0.5``.
        language: Pipeline language.  Passed to all default components.
            Must be one of :data:`~cavaquinho.config.SUPPORTED_LANGUAGES`.

    Example::

        import cavaquinho as caco

        validator = caco.caco()
        result = validator.validate(
            response="The LGPD was enacted in 2015.",
            context="The LGPD was enacted on August 14, 2018.",
        )
        print(result.is_hallucination)  # True
    """

    def __init__(
        self,
        extractor=None,
        classifier=None,
        aggregator=None,
        threshold: float = DEFAULT_THRESHOLD,
        language: str = DEFAULT_LANGUAGE,
    ):
        self.extractor = extractor or RuleExtractor(language=language)
        self.classifier = classifier or DeBERTaClassifier(language=language)
        self.aggregator = aggregator or Aggregator(threshold=threshold, language=language)
        self.threshold = threshold
        self.language = language

    def validate(
        self,
        response: str,
        context: str,
        prompt: str | None = None,
        query: str | None = None,
    ) -> ValidationResult:
        """Validate whether *response* is faithful to *context*.

        Claims are extracted from *response*, classified concurrently
        against *context*, and aggregated into a final verdict.

        Args:
            response: The LLM-generated text to evaluate.
            context: Source documents or retrieved passages that the
                response should be faithful to.
            prompt: Optional original user prompt forwarded to the
                extractor.  Alias for *query*; *query* takes precedence
                when both are provided.
            query: Optional original user query.  When provided, short
                claims (fewer than 5 tokens) are expanded with the query
                to give the NLI model sufficient context.  Useful for
                QA tasks where the response is a brief phrase or a
                single word.

        Returns:
            A :class:`~cavaquinho.models.ValidationResult` with the
            aggregated score, hallucination flag, per-claim breakdown,
            threshold used, and a human-readable summary.

        Raises:
            ValueError: If *response* or *context* is empty or
                whitespace-only.
        """
        if not response.strip():
            raise ValueError("Empty response is not usable to infer")

        if not context.strip():
            raise ValueError("Empty context is not usable to infer")

        effective_prompt = query or prompt
        claims_text = self.extractor.extract(response, context, effective_prompt)

        claim_results = self.classifier.classify_batch(claims_text, context)

        return self.aggregator.aggregate(claim_results)
