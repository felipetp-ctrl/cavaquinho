"""Faithfulness classifier backed by MiniCheck (Flan-T5 or DeBERTa-v3-Large)."""

from __future__ import annotations

from ..models import ClaimResult, Labels
from .base import ClassifierContract

_MINICHECK_MODELS = frozenset({
    "flan-t5-large",
    "deberta-v3-large",
    "roberta-large",
    "Bespoke-MiniCheck-7B",
    "Granite-Guardian-3.3-8B",
})


class MiniCheckClassifier(ClassifierContract):
    """Faithfulness classifier using MiniCheck (Liyang et al., 2024).

    MiniCheck is fine-tuned specifically for claim-level faithfulness
    verification. It scores each (document, claim) pair and returns the
    probability that the claim is *supported* by the document.

    Unlike the DeBERTa NLI cross-encoder, MiniCheck does not operate
    sentence-by-sentence — it consumes the full document per claim.
    The ``evidence`` field in the returned :class:`~cavaquinho.models.ClaimResult`
    is therefore always the full context string.

    Requires the ``minicheck`` package::

        pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git"

    Args:
        model_name: MiniCheck model variant. One of ``"flan-t5-large"``,
            ``"deberta-v3-large"``, ``"roberta-large"``,
            ``"Bespoke-MiniCheck-7B"``. Default: ``"flan-t5-large"``.
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
            When ``None``, MiniCheck selects automatically.
        batch_size: Number of (document, claim) pairs per forward pass.
    """

    def __init__(
        self,
        model_name: str = "flan-t5-large",
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        if model_name not in _MINICHECK_MODELS:
            raise ValueError(
                f"Unknown MiniCheck model '{model_name}'. "
                f"Choose from: {sorted(_MINICHECK_MODELS)}"
            )
        try:
            from minicheck.minicheck import MiniCheck  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "minicheck is required: "
                'pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git"'
            ) from exc

        kwargs: dict = {"model_name": model_name}
        if device is not None:
            kwargs["device"] = device
        self._scorer = MiniCheck(**kwargs)
        self.batch_size = batch_size

    def classify(self, claim: str, context: str) -> ClaimResult:
        pred_labels, raw_probs, _, _ = self._scorer.score(
            docs=[context], claims=[claim]
        )
        return self._make_result(claim, context, pred_labels[0], raw_probs[0])

    def classify_batch(self, claims: list[str], context: str) -> list[ClaimResult]:
        if not claims:
            return []
        docs = [context] * len(claims)
        pred_labels, raw_probs, _, _ = self._scorer.score(
            docs=docs, claims=claims
        )
        return [
            self._make_result(claim, context, pred, prob)
            for claim, pred, prob in zip(claims, pred_labels, raw_probs)
        ]

    @staticmethod
    def _make_result(
        claim: str,
        context: str,
        pred_label: int,
        raw_prob: float,
    ) -> ClaimResult:
        # MiniCheck raw_prob = P(supported). pred_label: 1=supported, 0=hallucinated.
        if pred_label == 0:
            label = Labels.VALUE_CONTRADICTION
            score = round(1.0 - raw_prob, 4)
            reason = context
        else:
            label = Labels.VALUE_ENTAILMENT
            score = round(raw_prob, 4)
            reason = None
        return ClaimResult(
            text=claim,
            evidence=context,
            label=label,
            score=score,
            reason=reason,
        )
