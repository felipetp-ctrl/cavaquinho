import nltk
from nltk.tokenize import sent_tokenize
from .base import ExtractorContract


class RuleExtractor(ExtractorContract):
    SUPPORTED_LANGUAGES = ("english", "portuguese")

    def __init__(self, language: str = "english"):
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Choose from: {self.SUPPORTED_LANGUAGES}"
            )
        self.language = language
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    def extract(self, response: str, _context: str, _prompt: str | None = None) -> list[str]:
        sentences = sent_tokenize(response, language=self.language)
        return [s.strip() for s in sentences if s.strip()]