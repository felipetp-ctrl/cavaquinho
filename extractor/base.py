from abc import ABC, abstractmethod

class ExtractorContract(ABC):
    @abstractmethod
    def extract(self, response: str, _context: str, _prompt: str | None = None) -> list[str]:
        ...