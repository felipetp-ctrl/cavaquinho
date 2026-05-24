from abc import ABC, abstractmethod

class ExtractorContract(ABC):
    @abstractmethod
    def extract(self, response: str, context: str, prompt: str | None = None) -> list[str]:
        ...