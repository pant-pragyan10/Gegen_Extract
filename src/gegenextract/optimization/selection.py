from __future__ import annotations
from typing import List, Dict, Any
from .candidate import Candidate


class SelectionEngine:
    def __init__(self, acceptance_delta: float = 0.01, stagnation_gens: int = 5, diversity_penalty: float = 0.0):
        self.acceptance_delta = acceptance_delta
        self.stagnation_gens = stagnation_gens
        self.diversity_penalty = diversity_penalty
        self.best_score = None
        self.generations_since_improve = 0

    def select(self, population: List[Candidate], current_best_score: float = None) -> List[Candidate]:
        # population assumed to have scores
        sorted_pop = sorted(population, key=lambda c: c.score, reverse=True)
        if not sorted_pop:
            return []
        top = sorted_pop[0]
        # initialize best
        if self.best_score is None:
            self.best_score = top.score
        # regression protection
        accepted = []
        for c in sorted_pop:
            delta = c.score - self.best_score
            if delta >= -self.acceptance_delta:
                accepted.append(c)
        # update best and stagnation
        if top.score > self.best_score + self.acceptance_delta:
            self.best_score = top.score
            self.generations_since_improve = 0
        else:
            self.generations_since_improve += 1
        # stagnation handling
        if self.generations_since_improve >= self.stagnation_gens:
            # keep top N but force exploration by returning slightly lower-ranked ones
            return sorted_pop[: max(1, len(sorted_pop) // 2)]
        return accepted
