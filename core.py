from .extractor.rule_extractor import RuleExtractor
from .classifier.deberta import DeBERTaClassifier
from .aggregator import Aggregator
from .config import DEFAULT_THRESHOLD, DEFAULT_LANGUAGE
from .models import ValidationResult
from concurrent.futures import ThreadPoolExecutor


class caco:
    def __init__(
        self,
        extractor=None,
        classifier=None,
        aggregator=None,
        threshold: float = DEFAULT_THRESHOLD,
        language: str = DEFAULT_LANGUAGE
    ):
        self.extractor = extractor or RuleExtractor(language=language)
        self.classifier = classifier or DeBERTaClassifier(language=language)
        self.aggregator = aggregator or Aggregator(language=language)
        self.threshold = threshold
        self.language = language

    def validate(self, response: str, context: str, prompt: str | None = None) -> ValidationResult:
        if not response.strip():
            raise ValueError("Empty response is not usable to infer")

        if not context.strip():
            raise ValueError("Empty context is not usable to infer")

        claims_text = self.extractor.extract(response, context, prompt)

        with ThreadPoolExecutor() as executor:
            claim_results = list(executor.map(
                lambda claim: self.classifier.classify(claim, context),
                claims_text
            ))

        return self.aggregator.aggregate(claim_results)