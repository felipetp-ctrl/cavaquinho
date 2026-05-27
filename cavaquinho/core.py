"""Main entry point — the :class:`Validator` / :class:`caco` validator."""

from __future__ import annotations

from .aggregator import Aggregator
from .classifier.deberta import DeBERTaClassifier
from .config import DEFAULT_LANGUAGE, DEFAULT_THRESHOLD
from .extractor.rule_extractor import RuleExtractor
from .models import ValidationResult


class Validator:
    """Faithfulness hallucination detector for LLM responses.

    Orchestrates three sequential stages:

    1. **Extraction** — decomposes the response into atomic claim strings.
    2. **Classification** — runs NLI inference on each (context sentence, claim)
       pair in a single batched call via :meth:`~cavaquinho.classifier.base.ClassifierContract.classify_batch`.
    3. **Aggregation** — combines per-claim results into a single
       :class:`~cavaquinho.models.ValidationResult`.

    Each stage is independently replaceable. See
    :class:`~cavaquinho.extractor.base.ExtractorContract` and
    :class:`~cavaquinho.classifier.base.ClassifierContract` for the
    interfaces required by custom implementations.

    Args:
        extractor: Claim extractor instance. Defaults to
            :class:`~cavaquinho.extractor.rule_extractor.RuleExtractor`.
        classifier: NLI classifier instance. Defaults to
            :class:`~cavaquinho.classifier.deberta.DeBERTaClassifier`.
            Requires ``pip install "cavaquinho[nli]"``.
        aggregator: Aggregator instance. Defaults to
            :class:`~cavaquinho.aggregator.Aggregator`.
        threshold: Hallucination score threshold in ``(0.0, 1.0)``.
            Responses whose aggregated score exceeds this value are flagged
            as hallucinations. Defaults to ``0.5``.
        language: Pipeline language. Passed to all default components.
            Must be one of :data:`~cavaquinho.config.SUPPORTED_LANGUAGES`.

    Note:
        The NLI model (~500 MB) is downloaded and loaded into memory when
        :class:`Validator` is instantiated, not when :meth:`validate` is
        called. Instantiate once and reuse across requests. Use
        :meth:`preload` to make the warm-up intent explicit in server code.

    Example::

        from cavaquinho import Validator

        validator = Validator()  # model loads here
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

    def preload(self) -> None:
        """No-op warm-up hook for explicit model pre-loading.

        The classifier model is already loaded during ``__init__``. This
        method exists to make the warm-up intent self-documenting::

            validator = Validator()
            validator.preload()  # signals that loading is intentional here
        """

    def validate(
        self,
        response: str,
        context: str,
        prompt: str | None = None,
    ) -> ValidationResult:
        """Validate whether *response* is faithful to *context*.

        Claims are extracted from *response*, classified in a single batched
        NLI call against *context*, and aggregated into a final verdict.

        Args:
            response: The LLM-generated text to evaluate.
            context: Source documents or retrieved passages that the
                response should be faithful to.
            prompt: Optional original user prompt forwarded to the extractor.

        Returns:
            A :class:`~cavaquinho.models.ValidationResult`.

        Raises:
            ValueError: If *response* or *context* is empty or whitespace-only.
        """
        if not response.strip():
            raise ValueError("Empty response is not usable to infer")
        if not context.strip():
            raise ValueError("Empty context is not usable to infer")

        claims_text = self.extractor.extract(response, context, prompt)
        claim_results = self.classifier.classify_batch(claims_text, context)
        return self.aggregator.aggregate(claim_results)

    def validate_batch(
        self,
        responses: list[str],
        context: str,
        contexts: list[str] | None = None,
    ) -> list[ValidationResult]:
        """Validate multiple responses, optionally against distinct contexts.

        Args:
            responses: List of LLM-generated texts to evaluate.
            context: Shared context used when *contexts* is not provided.
            contexts: Per-response context strings. When provided, must be
                the same length as *responses*.

        Returns:
            List of :class:`~cavaquinho.models.ValidationResult` in the same
            order as *responses*.

        Raises:
            ValueError: If *contexts* is provided but has a different length
                than *responses*.
        """
        if contexts is not None and len(contexts) != len(responses):
            raise ValueError(
                f"contexts length ({len(contexts)}) must match responses length ({len(responses)})"
            )

        results = []
        for i, response in enumerate(responses):
            ctx = contexts[i] if contexts is not None else context
            results.append(self.validate(response=response, context=ctx))
        return results

    def __repr__(self) -> str:
        return (
            f"Validator(threshold={self.threshold}, language={self.language!r}, "
            f"classifier={type(self.classifier).__name__})"
        )


caco = Validator  # backwards-compatible alias
