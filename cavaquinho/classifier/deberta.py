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
    the NLI model over every (sentence, claim) pair in a single batched call.
    The sentence with the highest contradiction score is selected as evidence
    when a contradiction is detected; if any sentence contradicts the claim,
    the result is CONTRADICTION regardless of other sentence scores.

    Args:
        model_name: HuggingFace model identifier.
        device: Compute device override (CUDA → MPS → CPU when ``None``).
        language: Sentence tokenisation language.
        batch_size: Pipeline batch size for GPU/MPS throughput.

    Raises:
        ValueError: If *language* is not supported.
    """

    _LABEL_MAP = {
        "entailment": Labels.VALUE_ENTAILMENT,
        "neutral": Labels.VALUE_NEUTRAL,
        "contradiction": Labels.VALUE_CONTRADICTION,
    }
    _CONTRADICTION_THRESHOLD = 0.5

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | int | None = None,
        language: str = "english",
        batch_size: int = 32,
        calibrator_path: str | None = None,
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
            top_k=None,
        )
        # Optional calibrator (Platt / sklearn estimator) to map raw
        # contradiction probabilities to calibrated probabilities.
        self.calibrator = None
        if calibrator_path is not None:
            try:
                import joblib

                self.calibrator = joblib.load(calibrator_path)
            except Exception:
                self.calibrator = None

    def classify(self, claim: str, context: str) -> ClaimResult:
        """Classify the faithfulness of *claim* against *context*."""
        sentences = sent_tokenize(context, language=self.language)
        if not sentences:
            return ClaimResult(text=claim, evidence="", label=Labels.VALUE_NEUTRAL, score=0.0, reason=None)

        pairs = [{"text": s, "text_pair": claim} for s in sentences]
        raw = self.pipeline(pairs, batch_size=self.batch_size, truncation=True, max_length=512)
        items: list[Any] = raw if isinstance(raw, list) else [raw]  # type: ignore[assignment]
        return self._select_best(claim, sentences, items)

    def classify_batch(self, claims: list[str], context: str) -> list[ClaimResult]:
        """Classify multiple claims against *context* in a single model call.

        All (sentence, claim) pairs are batched into one pipeline invocation,
        maximising GPU/MPS utilisation.

        Args:
            claims: Atomic claim strings to verify.
            context: Shared context string for all claims.

        Returns:
            List of :class:`~cavaquinho.models.ClaimResult` in the same order as *claims*.
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
        all_items: list[Any] = raw if isinstance(raw, list) else [raw]  # type: ignore[assignment]

        return [
            self._select_best(claim, sentences, all_items[i * m: (i + 1) * m])
            for i, claim in enumerate(claims)
        ]

    def _select_best(
        self,
        claim: str,
        sentences: list[str],
        items: list[Any],
    ) -> ClaimResult:
        """Pick the decisive (sentence, label, score) from a list of NLI results.

        Uses direct contradiction probability when deciding contradiction: if any
        sentence reaches ``_CONTRADICTION_THRESHOLD``, the highest contradiction
        score wins regardless of another sentence's top-label confidence.
        """
        best_contradiction_score = 0.0
        best_contradiction_sentence: str | None = None
        best_overall_score = -1.0
        best_overall_sentence: str | None = None
        best_overall_label: str | None = None

        for sentence, item in zip(sentences, items):
            label, score, contradiction_score = self._extract_scores(item)

            if contradiction_score > best_contradiction_score:
                best_contradiction_score = contradiction_score
                best_contradiction_sentence = sentence

            if score > best_overall_score:
                best_overall_score = score
                best_overall_sentence = sentence
                best_overall_label = label

        if best_overall_label is None:
            return ClaimResult(text=claim, evidence="", label=Labels.VALUE_NEUTRAL, score=0.0, reason=None)

        if (
            best_contradiction_sentence is not None
            and best_contradiction_score >= self._CONTRADICTION_THRESHOLD
        ):
            score = best_contradiction_score
            _cal = getattr(self, "calibrator", None)
            if _cal is not None:
                try:
                    import numpy as _np

                    if hasattr(_cal, "predict_proba"):
                        score = float(_cal.predict_proba(_np.array([[score]]))[:, 1])
                    else:
                        score = float(_cal.predict(_np.array([score])))
                except Exception:
                    pass

            return ClaimResult(
                text=claim,
                evidence=best_contradiction_sentence,
                label=Labels.VALUE_CONTRADICTION,
                score=round(score, 4),
                reason=best_contradiction_sentence,
            )

        # If no sentence passed the contradiction threshold, report the
        # maximum contradiction probability seen across sentences as the
        # unified score. This allows the aggregator to weight by p(contradicti
        # on) even when the top label wasn't contradiction.
        score = best_contradiction_score
        _cal = getattr(self, "calibrator", None)
        if _cal is not None:
            try:
                import numpy as _np

                if hasattr(_cal, "predict_proba"):
                    score = float(_cal.predict_proba(_np.array([[score]]))[:, 1])
                else:
                    score = float(_cal.predict(_np.array([score])))
            except Exception:
                pass

        return ClaimResult(
            text=claim,
            evidence=best_overall_sentence or "",
            label=self._LABEL_MAP[best_overall_label],
            score=round(score, 4),
            reason=None,
        )

    @staticmethod
    def _extract_scores(item: Any) -> tuple[str, float, float]:
        """Return (top_label, top_score, contradiction_score) for one sentence output."""
        if isinstance(item, dict):
            label = item["label"].lower()
            score = float(item["score"])
            contradiction_score = score if label == "contradiction" else 0.0
            return label, score, contradiction_score

        if isinstance(item, list):
            top_label = "neutral"
            top_score = -1.0
            contradiction_score = 0.0
            for entry in item:
                label = entry["label"].lower()
                score = float(entry["score"])
                if score > top_score:
                    top_label = label
                    top_score = score
                if label == "contradiction":
                    contradiction_score = score
            if top_score < 0.0:
                return "neutral", 0.0, 0.0
            return top_label, top_score, contradiction_score

        return "neutral", 0.0, 0.0
