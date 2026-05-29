from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional
import random
from .artifact import PromptArtifact
import hashlib


@dataclass
class Mutation:
    name: str
    params: Dict[str, Any]
    seed: Optional[int] = None

    def id(self) -> str:
        key = f"{self.name}:{sorted(self.params.items())}:{self.seed}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()


class MutationEngine:
    def __init__(self, operators: Dict[str, Callable[[PromptArtifact, Dict[str, Any], random.Random], PromptArtifact]], seed: int = 0):
        self.operators = operators
        self.seed = seed

    def _random(self, seed: Optional[int] = None) -> random.Random:
        return random.Random(self.seed if seed is None else seed)

    def apply(self, artifact: PromptArtifact, mutation: Mutation) -> PromptArtifact:
        rng = self._random(mutation.seed)
        op = self.operators.get(mutation.name)
        if not op:
            raise ValueError(f"Unknown mutation operator: {mutation.name}")
        new = op(artifact, mutation.params, rng)
        # bump version and annotate
        bumped = new.bumped({"mutation_id": mutation.id(), "mutation_name": mutation.name, "mutation_params": mutation.params})
        return bumped

    def propose_from_diagnostics(self, diagnostics: Dict[str, Any]) -> List[Mutation]:
        # failure-aware mapping: simple rules
        muts: List[Mutation] = []
        # malformed JSON -> strengthen formatting
        if diagnostics.get("malformed_json"):
            muts.append(Mutation("strengthen_constraint", {"reason": "malformed_json"}, seed=self.seed))
        # low array recall
        if diagnostics.get("low_array_recall"):
            muts.append(Mutation("schema_guidance", {"focus": "arrays"}, seed=self.seed))
        # hallucinations
        if diagnostics.get("hallucinated_fields"):
            muts.append(Mutation("reduce_hallucination", {"fields": diagnostics.get("hallucinated_fields")}, seed=self.seed))
        return muts


def operator_rewrite_instruction(art: PromptArtifact, params: Dict[str, Any], rng: random.Random) -> PromptArtifact:
    new = art.model_copy(deep=True)
    # naive rewrite: append clarifying sentence
    note = params.get("note", "Be concise and follow the schema strictly.")
    new.instructions = (new.instructions or "") + "\n" + note
    return new


def operator_strengthen_constraint(art: PromptArtifact, params: Dict[str, Any], rng: random.Random) -> PromptArtifact:
    new = art.model_copy(deep=True)
    add = "\nEnsure the output is valid JSON and strictly follows the schema."
    new.constraints = (new.constraints or "") + add
    return new


def operator_reduce_hallucination(art: PromptArtifact, params: Dict[str, Any], rng: random.Random) -> PromptArtifact:
    new = art.model_copy(deep=True)
    add = "\nDo not invent fields; only return fields present in the schema or source text."
    new.instructions = (new.instructions or "") + add
    return new


def operator_formatting_mutation(art: PromptArtifact, params: Dict[str, Any], rng: random.Random) -> PromptArtifact:
    new = art.model_copy(deep=True)
    style = params.get("style", "compact")
    if style == "compact":
        new.constraints = (new.constraints or "") + "\nPrefer compact JSON without extra whitespace."
    else:
        new.constraints = (new.constraints or "") + "\nPretty-print JSON with 2-space indentation."
    return new


def operator_schema_guidance(art: PromptArtifact, params: Dict[str, Any], rng: random.Random) -> PromptArtifact:
    new = art.model_copy(deep=True)
    focus = params.get("focus", "fields")
    new.instructions = (new.instructions or "") + f"\nPay special attention to {focus} and repeated items."
    return new


def operator_repair_instruction(art: PromptArtifact, params: Dict[str, Any], rng: random.Random) -> PromptArtifact:
    new = art.model_copy(deep=True)
    new.repair = (new.repair or "") + "\nIf output is invalid JSON, attempt minimal repairs and reformat."
    return new


DEFAULT_OPERATORS = {
    "rewrite_instruction": operator_rewrite_instruction,
    "strengthen_constraint": operator_strengthen_constraint,
    "reduce_hallucination": operator_reduce_hallucination,
    "formatting_mutation": operator_formatting_mutation,
    "schema_guidance": operator_schema_guidance,
    "repair_instruction": operator_repair_instruction,
}
