import torch
from transformers import pipeline
from ..models import Labels, ClaimResult
from .base import ClassifierContract
from nltk.tokenize import sent_tokenize

class DeBERTaClassifier(ClassifierContract):
    SUPPORTED_LANGUAGES = ("english", "portuguese")

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: int | None = None,
        language: str = "english"
    ):
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Choose from: {self.SUPPORTED_LANGUAGES}"
            )
        self.language = language

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.model_name = model_name
        self.device = device
        self.pipeline = pipeline(
            task="zero-shot-classification",
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
                sequences=claim,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="This statement is {}.",
            )

            contradiction_score = result["scores"][result["labels"].index("contradiction")]

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

        labels = best_result["labels"]
        scores = best_result["scores"]

        top_label = labels[0]
        top_score = scores[0]

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