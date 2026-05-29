from abc import ABC, abstractmethod
from typing import Any
from ..schemas import Document, ExtractionResult


class Extractor(ABC):
    """Extractor runs extraction logic given a document and prompts."""

    @abstractmethod
    def extract(self, document: Document, prompt: str) -> ExtractionResult:
        """Run extraction for a single document using the given prompt."""


class Pipeline:
    """Simple, composable extraction pipeline skeleton."""

    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def run(self, document: Document, prompt: str) -> ExtractionResult:
        return self.extractor.extract(document, prompt)
