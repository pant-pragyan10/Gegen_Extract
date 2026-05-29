from gegenextract.optimization.artifact import PromptArtifact
from gegenextract.optimization.mutation import MutationEngine, DEFAULT_OPERATORS, Mutation
from gegenextract.optimization.candidate import CandidateGenerator
from gegenextract.optimization.selection import SelectionEngine


def test_duplicate_detection_and_selection():
    seed = PromptArtifact(instructions="Start")
    me = MutationEngine(DEFAULT_OPERATORS, seed=0)
    cg = CandidateGenerator(me, population_size=4, beam_width=2)
    muts = [("rewrite_instruction", {"note": "A"}), ("rewrite_instruction", {"note": "A"}), ("strengthen_constraint", {})]
    cands = cg.generate(seed, muts)
    # duplicates removed
    assert len(cands) <= 2
    # assign scores and test selection
    for i, c in enumerate(cands):
        c.score = i * 0.1
    sel = SelectionEngine(acceptance_delta=0.05, stagnation_gens=2)
    selected = sel.select(cands)
    assert isinstance(selected, list)
