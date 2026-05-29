from abc import ABC, abstractmethod
from typing import Dict
from ..schemas import ExtractionResult


class Scorer(ABC):
    """Compute metrics for an extraction result against a reference."""

    @abstractmethod
    def score(self, result: ExtractionResult, reference: Dict[str, any]) -> Dict[str, float]:
        """Return a metric dictionary (e.g., precision, recall, f1)."""
