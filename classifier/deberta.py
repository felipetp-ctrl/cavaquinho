import torch
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from ..models import Labels, ClaimResult
from ..config import SUPPORTED_LANGUAGES, DEFAULT_MODEL
from .base import ClassifierContract


class DeBERTaClassifier(ClassifierContract):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: int | None = None,
        language: str = "english"
    ):
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Choose from: {SUPPORTED_LANGUAGES}"
            )
        self.language = language

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.model_name = model_name
        self.device = device
        self.pipeline = pipeline(
            task="text-classification",
            model=model_name,
            device=device
        )

    def classify(self, claim: str, context: str) -> ClaimResult:
        sentences = sent_tokenize(context, language=self.language)

        best_sentence = None
        best_score = -1.0
        best_result = None

        for sentence in sentences:
            result = self.pipeline(
                f"{sentence} [SEP] {claim}",
                truncation=True,
                max_length=512
            )

            label = result[0]["label"].lower()
            score = result[0]["score"]
            contradiction_score = score if label == "contradiction" else 0.0

            if contradiction_score > best_score:
                best_score = contradiction_score
                best_sentence = sentence
                best_result = result

        if best_result is None:
            return ClaimResult(
                text=claim,
                evidence="",
                label=Labels.VALUE_NEUTRAL,
                score=0.0,
                reason=None
            )

        top_label = best_result[0]["label"].lower()
        top_score = best_result[0]["score"]

        label_map = {
            "entailment": Labels.VALUE_ENTAILMENT,
            "neutral": Labels.VALUE_NEUTRAL,
            "contradiction": Labels.VALUE_CONTRADICTION,
        }
        mapped_label = label_map[top_label]

        return ClaimResult(
            text=claim,
            evidence=best_sentence or "",
            label=mapped_label,
            score=round(top_score, 4),
            reason=top_label if mapped_label == Labels.VALUE_CONTRADICTION else None
        )