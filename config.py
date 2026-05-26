"""Package-level configuration constants."""

SUPPORTED_LANGUAGES: tuple[str, ...] = ("english", "portuguese")
"""Languages accepted by extractors, classifiers, and the aggregator."""

DEFAULT_THRESHOLD: float = 0.5
"""Hallucination score threshold used when none is explicitly provided."""

DEFAULT_MODEL: str = "cross-encoder/nli-deberta-v3-base"
"""Default NLI model loaded by :class:`~cavaquinho.classifier.deberta.DeBERTaClassifier`."""

DEFAULT_LANGUAGE: str = "english"
"""Default language used throughout the pipeline."""

THRESHOLDS: dict[str, float] = {
    "qa": 0.3,
    "summarization": 0.4,
    "dialogue": 0.35,
    "default": 0.5,
}
"""Recommended thresholds per task type, calibrated against HaluEval.

Lower thresholds increase recall (catch more hallucinations) at the cost of
more false positives. Choose based on the cost of missing a hallucination vs.
the cost of a false alarm in your application.

Example::

    from cavaquinho import THRESHOLDS
    validator = caco.caco(threshold=THRESHOLDS["qa"])
"""
