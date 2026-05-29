from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Callable, Any, Set, Tuple
from .artifact import PromptArtifact
from .mutation import Mutation
import hashlib


def _hash_prompt(serialized: str) -> str:
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


@dataclass
class Candidate:
    artifact: PromptArtifact
    score: float = 0.0
    metadata: Dict[str, Any] = None


class CandidateGenerator:
    def __init__(self, mutation_engine, population_size: int = 8, beam_width: int = 3):
        self.mutation_engine = mutation_engine
        self.population_size = population_size
        self.beam_width = beam_width
        self.seen_hashes: Set[str] = set()

    def is_duplicate(self, artifact: PromptArtifact) -> bool:
        h = _hash_prompt(artifact.serialize())
        return h in self.seen_hashes

    def mark_seen(self, artifact: PromptArtifact):
        self.seen_hashes.add(_hash_prompt(artifact.serialize()))

    def generate(self, seed_artifact: PromptArtifact, mutations: List[Tuple[str, dict]]) -> List[Candidate]:
        # mutations: list of tuples (op_name, params)
        candidates: List[Candidate] = []
        # produce mutated population
        for i, (name, params) in enumerate(mutations):
            mut_obj = Mutation(name=name, params=params, seed=i)
            mut = self.mutation_engine.apply(seed_artifact, mutation=mut_obj)
            if self.is_duplicate(mut):
                continue
            c = Candidate(artifact=mut)
            candidates.append(c)
            self.mark_seen(mut)
            if len(candidates) >= self.population_size:
                break
        # return top beam_width candidates (unscored yet)
        return candidates[: self.beam_width]
