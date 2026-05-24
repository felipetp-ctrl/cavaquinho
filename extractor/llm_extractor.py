import json
from .base import ExtractorContract
from .rule_extractor import RuleExtractor
from typing import Callable


class LLMExtractor(ExtractorContract):
    PROMPT_TEMPLATE = """You will receive a text and must extract all atomic factual claims from it.
Each claim must be a single, verifiable statement.
Return ONLY a JSON array of strings, with no explanation, no markdown, no preamble.

Example output:
["Spiders has 8 legs.", "Brazil has 210 million inhabitants."]

Text:
{response}"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        max_claims: int = 10,
        language: str = "english"
    ):
        self.llm_fn = llm_fn
        self.max_claims = max_claims
        self._fallback = RuleExtractor(language=language)

    def extract(self, response: str, _context: str, _prompt: str | None = None) -> list[str]:
        prompt = self.PROMPT_TEMPLATE.format(response=response)

        try:
            raw = self.llm_fn(prompt)
            claims = self._parse(raw)
        except Exception:
            claims = self._fallback.extract(response, _context)

        return claims[:self.max_claims]

    def _parse(self, raw: str) -> list[str]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]).strip()

        parsed = json.loads(cleaned)

        if not isinstance(parsed, list):
            raise ValueError("LLM did not return a JSON array.")

        return [str(claim).strip() for claim in parsed if str(claim).strip()]