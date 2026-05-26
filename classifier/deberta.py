"""NLI classifier backed by a DeBERTa cross-encoder model."""

from __future__ import annotations

from typing import Any

import torch
from nltk.tokenize import sent_tokenize
from transformers import pipeline

from ..config import DEFAULT_MODEL, SUPPORTED_LANGUAGES
from ..models import ClaimResult, Labels
from .base import ClassifierContract


def _resolve_device(device: str | int | None) -> str | int:
    """Select the best available compute device when none is specified.

    Priority order: CUDA GPU → Apple Silicon MPS → CPU.

    Args:
        device: Explicit device override.  Pass ``0`` for the first CUDA
            GPU, ``"mps"`` for Apple Silicon, or ``-1`` for CPU.  When
            ``None``, the device is chosen automatically.

    Returns:
        A device identifier accepted by the HuggingFace ``pipeline`` API.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


class DeBERTaClassifier(ClassifierContract):
    """NLI classifier using a fine-tuned DeBERTa cross-encoder model.

    For each claim the classifier splits *context* into sentences and runs
    the NLI model over every (sentence, claim) pair.  The sentence that
    produces the highest contradiction score is selected as the evidence
    for a contradiction result; for non-contradiction outcomes the sentence
    with the overall highest model confidence is returned as evidence.

    The default model is ``cross-encoder/nli-deberta-v3-base``, which runs
    fully locally and requires no API key.  An alternative model can be
    substituted by passing *model_name*.

    .. note::

        This class requires ``transformers`` and ``torch`` to be installed.
        Install them with ``pip install "cavaquinho[nli]"``.

    Args:
        model_name: HuggingFace model identifier.  Must be a cross-encoder
            model compatible with the ``text-classification`` pipeline.
        device: Compute device override.  When ``None``, the best available
            device is selected automatically (CUDA → MPS → CPU).
        language: Language used for sentence tokenisation of *context*.
            Must be one of :data:`~cavaquinho.config.SUPPORTED_LANGUAGES`.

    Raises:
        ValueError: If *language* is not supported.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | int | None = None,
        language: str = "english",
        batch_size: int = 32,
    ):
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Choose from: {SUPPORTED_LANGUAGES}"
            )
        self.language = language
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.pipeline = pipeline(
            task="text-classification",
            model=model_name,
            device=self.device,
        )

    _LABEL_MAP = {
        "entailment": Labels.VALUE_ENTAILMENT,
        "neutral": Labels.VALUE_NEUTRAL,
        "contradiction": Labels.VALUE_CONTRADICTION,
    }

    def classify(self, claim: str, context: str) -> ClaimResult:
        """Classify the faithfulness of *claim* against *context*.

        All context sentences are scored against *claim* in a single
        batched pipeline call.  The sentence with the highest contradiction
        score is stored as evidence when a contradiction is detected;
        otherwise the sentence with the highest model confidence is used.

        Args:
            claim: Atomic claim to verify.
            context: Full context string retrieved for the response.

        Returns:
            A :class:`~cavaquinho.models.ClaimResult` with the NLI label,
            model confidence, and the most relevant evidence sentence.
            Returns a neutral result with an empty evidence string if the
            context contains no sentences.
        """
        sentences = sent_tokenize(context, language=self.language)
        if not sentences:
            return ClaimResult(
                text=claim,
                evidence="",
                label=Labels.VALUE_NEUTRAL,
                score=0.0,
                reason=None,
            )

        pairs = [{"text": s, "text_pair": claim} for s in sentences]
        raw = self.pipeline(pairs, batch_size=self.batch_size, truncation=True, max_length=512)
        items: list[dict[str, Any]] = raw if isinstance(raw, list) else [raw]  # type: ignore[assignment]

        return self._select_best(claim, sentences, items)

    def classify_batch(self, claims: list[str], context: str) -> list[ClaimResult]:
        """Classify multiple claims against *context* in a single model call.

        Builds all (sentence, claim) pairs up front and passes them to the
        pipeline in one batched invocation.  This amortises tokenisation
        overhead and maximises GPU/MPS utilisation when many claims are
        present.

        Args:
            claims: Atomic claim strings to verify.
            context: Shared full context string for all claims.

        Returns:
            List of :class:`~cavaquinho.models.ClaimResult` in the same
            order as *claims*.
        """
        if not claims:
            return []

        sentences = sent_tokenize(context, language=self.language)
        if not sentences:
            return [
                ClaimResult(text=c, evidence="", label=Labels.VALUE_NEUTRAL, score=0.0, reason=None)
                for c in claims
            ]

        m = len(sentences)
        all_pairs = [
            {"text": s, "text_pair": claim}
            for claim in claims
            for s in sentences
        ]
        raw = self.pipeline(all_pairs, batch_size=self.batch_size, truncation=True, max_length=512)
        all_items: list[dict[str, Any]] = raw if isinstance(raw, list) else [raw]  # type: ignore[assignment]

        return [
            self._select_best(claim, sentences, all_items[i * m: (i + 1) * m])
            for i, claim in enumerate(claims)
        ]

    def _select_best(
        self,
        claim: str,
        sentences: list[str],
        items: list[dict[str, Any]],
    ) -> ClaimResult:
        best_contradiction_score = -1.0
        best_contradiction_sentence: str | None = None
        best_overall_score = -1.0
        best_overall_sentence: str | None = None
        best_overall_item: dict[str, Any] | None = None

        for sentence, item in zip(sentences, items):
            label = item["label"].lower()
            score = item["score"]

            if score > best_overall_score:
                best_overall_score = score
                best_overall_sentence = sentence
                best_overall_item = item

            if label == "contradiction" and score > best_contradiction_score:
                best_contradiction_score = score
                best_contradiction_sentence = sentence

        if best_overall_item is None:
            return ClaimResult(
                text=claim, evidence="", label=Labels.VALUE_NEUTRAL, score=0.0, reason=None
            )

        top_label = best_overall_item["label"].lower()
        top_score = best_overall_item["score"]
        mapped_label = self._LABEL_MAP[top_label]
        evidence = best_contradiction_sentence or best_overall_sentence or ""
        reason = best_contradiction_sentence if mapped_label == Labels.VALUE_CONTRADICTION else None

        return ClaimResult(
            text=claim,
            evidence=evidence,
            label=mapped_label,
            score=round(top_score, 4),
            reason=reason,
        )
