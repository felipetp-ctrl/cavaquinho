"""Abstract contract for NLI classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import ClaimResult


class ClassifierContract(ABC):
    """Interface that all NLI classifiers must implement.

    A classifier receives a single atomic claim and the full context string,
    and returns a :class:`~cavaquinho.models.ClaimResult` describing the
    inferred relationship between them.

    Custom classifiers can be passed to :class:`~cavaquinho.core.caco`
    without modifying any other part of the pipeline::

        from cavaquinho.classifier.base import ClassifierContract
        from cavaquinho.models import ClaimResult, Labels

        class MyClassifier(ClassifierContract):
            def classify(self, claim: str, context: str) -> ClaimResult:
                ...
    """

    @abstractmethod
    def classify(self, claim: str, context: str) -> ClaimResult:
        """Classify the relationship between *claim* and *context*.

        The classifier is free to split *context* into sentences internally
        and select the most relevant one as evidence.

        Args:
            claim: A single atomic claim string to verify.
            context: The full context string retrieved for the response.

        Returns:
            A :class:`~cavaquinho.models.ClaimResult` with the assigned
            label, model confidence score, and the context sentence used
            as evidence.
        """
        ...
