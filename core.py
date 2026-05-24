from .extractor.rule_extractor import RuleExtractor
from .classifier.deberta import DeBERTaClassifier
from .aggregator import Aggregator
from concurrent.futures import ThreadPoolExecutor

class caco:
    def __init__(self, extractor=None, classifier=None, aggregator=None, threshold=0.5, language="english"):
        self.extractor = extractor or RuleExtractor(language=language)
        self.classifier = classifier or DeBERTaClassifier(language=language)
        self.aggregator = aggregator or Aggregator(language=language)
        self.threshold = threshold
        self.language = language

    def validate(self, response: str, context: str, prompt: str | None = None):
        if not response:
            raise ValueError("Empty response is not usable to infer")
        
        if not context:
            raise ValueError("Empty context is not usable to infer")
        
        claims_text = self.extractor.extract(response, context, prompt)

        with ThreadPoolExecutor() as executor:
            claim_results = list(executor.map(
                lambda claim: self.classifier.classify(claim, context),
                claims_text
            ))

        return self.aggregator.aggregate(claim_results)