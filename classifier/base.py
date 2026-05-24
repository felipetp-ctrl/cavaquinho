from ..models import ClaimResult
from abc import ABC, abstractmethod

class ClassifierContract(ABC):
    @abstractmethod
    def classify(self, claim: str, context: str) -> ClaimResult:
        ...