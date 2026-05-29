from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator


class Optimizer(ABC):
    """Abstract optimizer interface for prompt/hyperparameter search."""

    @abstractmethod
    def suggest(self) -> Dict[str, Any]:
        """Return the next candidate configuration to evaluate."""

    @abstractmethod
    def observe(self, candidate: Dict[str, Any], metric: float) -> None:
        """Record the result for a candidate."""

    @abstractmethod
    def best(self) -> Dict[str, Any]:
        """Return the best seen candidate so far."""


class RandomSearchOptimizer(Optimizer):
    """A tiny example strategy skeleton. Replace with real implementations."""

    def __init__(self, search_space: Dict[str, Any], seed: int | None = None):
        self.search_space = search_space
        self.seed = seed
        self._best = None

    def suggest(self) -> Dict[str, Any]:
        # placeholder: return first element of search space deterministically
        return {k: (v[0] if isinstance(v, list) else v) for k, v in self.search_space.items()}

    def observe(self, candidate: Dict[str, Any], metric: float) -> None:
        if self._best is None or metric > self._best[1]:
            self._best = (candidate, metric)

    def best(self) -> Dict[str, Any]:
        return self._best[0] if self._best else {}
