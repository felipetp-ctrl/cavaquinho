"""LLM-based atomic claim extractor with rule-based fallback."""

from __future__ import annotations

import json
import logging
from typing import Callable

from .base import ExtractorContract
from .rule_extractor import RuleExtractor

logger = logging.getLogger(__name__)


class LLMExtractor(ExtractorContract):
    """Extracts atomic factual claims from a response using an LLM.

    The extractor sends a structured prompt to *llm_fn* that instructs the
    model to return a JSON array of atomic, independently verifiable
    statements.  If the LLM call fails for any reason, the extractor
    transparently falls back to :class:`~cavaquinho.extractor.rule_extractor.RuleExtractor`
    and logs a warning.

    This extractor is recommended when higher-precision decomposition is
    needed — for example, when responses contain compound sentences that
    bundle multiple verifiable facts.

    Args:
        llm_fn: A callable that accepts a prompt string and returns the
            model response as a string.  Any LLM client can be wrapped::

                from openai import OpenAI
                client = OpenAI()

                extractor = LLMExtractor(
                    llm_fn=lambda p: client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": p}]
                    ).choices[0].message.content
                )

        max_claims: Maximum number of claims returned.  Claims beyond this
            limit are silently truncated.  Defaults to 20.
        language: Language used by the fallback
            :class:`~cavaquinho.extractor.rule_extractor.RuleExtractor`.

    Raises:
        ValueError: If *language* is not supported (raised by the fallback
            extractor on initialisation).
    """

    PROMPT_TEMPLATE = """You will receive a text and must extract all atomic factual claims from it.
Each claim must be a single, verifiable statement.
Return ONLY a JSON array of strings, with no explanation, no markdown, no preamble.

Example output:
["Spiders have 8 legs.", "Brazil has over 210 million inhabitants."]

Text:
{response}"""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        max_claims: int = 20,
        language: str = "english",
    ):
        self.llm_fn = llm_fn
        self.max_claims = max_claims
        self._fallback = RuleExtractor(language=language)

    def extract(
        self, response: str, context: str = "", prompt: str | None = None
    ) -> list[str]:
        """Extract atomic claims via an LLM, falling back to sentence splitting on failure.

        Args:
            response: The LLM-generated text to decompose.
            context: Unused by this extractor; accepted for interface
                compatibility.
            prompt: Unused by this extractor; accepted for interface
                compatibility.

        Returns:
            Up to *max_claims* non-empty claim strings.
        """
        extraction_prompt = self.PROMPT_TEMPLATE.format(response=response)

        try:
            raw = self.llm_fn(extraction_prompt)
            claims = self._parse(raw)
        except Exception as exc:
            logger.warning(
                "LLMExtractor failed (%s: %s); falling back to RuleExtractor.",
                type(exc).__name__,
                exc,
            )
            claims = self._fallback.extract(response, context)

        return claims[: self.max_claims]

    def _parse(self, raw: str) -> list[str]:
        """Parse the raw LLM output into a list of claim strings.

        Args:
            raw: Raw string returned by the LLM.

        Returns:
            List of non-empty claim strings.

        Raises:
            ValueError: If the parsed value is not a JSON array.
            json.JSONDecodeError: If the output cannot be parsed as JSON.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]).strip()

        parsed = json.loads(cleaned)

        if not isinstance(parsed, list):
            raise ValueError("LLM did not return a JSON array.")

        return [str(claim).strip() for claim in parsed if str(claim).strip()]
