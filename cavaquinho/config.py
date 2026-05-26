"""Package-level configuration constants."""

SUPPORTED_LANGUAGES: tuple[str, ...] = ("english", "portuguese")
"""Languages accepted by extractors, classifiers, and the aggregator."""

DEFAULT_THRESHOLD: float = 0.5
"""Hallucination score threshold used when none is explicitly provided."""

DEFAULT_MODEL: str = "cross-encoder/nli-deberta-v3-base"
"""Default NLI model loaded by :class:`~cavaquinho.classifier.deberta.DeBERTaClassifier`."""

DEFAULT_LANGUAGE: str = "english"
"""Default language used throughout the pipeline."""
