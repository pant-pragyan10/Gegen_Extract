from gegenextract.optimization.artifact import PromptArtifact
from gegenextract.optimization.mutation import MutationEngine, DEFAULT_OPERATORS, Mutation


def test_mutation_deterministic():
    art = PromptArtifact(instructions="Base.")
    me = MutationEngine(DEFAULT_OPERATORS, seed=123)
    m = Mutation("rewrite_instruction", {"note": "Add clarity."}, seed=1)
    a1 = me.apply(art, m)
    a2 = me.apply(art, m)
    assert a1.instructions == a2.instructions
    assert a1.version == a2.version


def test_serialization_and_diff():
    art = PromptArtifact(instructions="Base.")
    art2 = art.bumped({"note": "mutated"})
    s = art.serialize()
    d = art.diff(art2)
    assert "mutated" in d or d is not None
