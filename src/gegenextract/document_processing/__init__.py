from abc import ABC, abstractmethod
from typing import Iterable
from ..schemas import Document


class DocumentProcessor(ABC):
    """Interface for document ingestion and preprocessing."""

    @abstractmethod
    def ingest(self, path: str) -> Iterable[Document]:
        """Ingest documents from a path and yield `Document` objects."""

    @abstractmethod
    def extract_text(self, document: Document) -> Document:
        """Populate `document.text` from the source file."""
