from abc import ABC, abstractmethod
from typing import Iterable
from ..schemas import PromptVersion


class PromptManager(ABC):
    """Manage prompt templates and versions."""

    @abstractmethod
    def list_versions(self) -> Iterable[PromptVersion]:
        """Yield available prompt versions."""

    @abstractmethod
    def get(self, version_id: str) -> PromptVersion:
        """Return a specific prompt version by id."""

    @abstractmethod
    def save(self, prompt: PromptVersion) -> None:
        """Persist a new prompt version."""
