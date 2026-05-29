"""Sample optimization runner. This demonstrates optimizer wiring but relies on an external scorer function.

It is intentionally adapter-based: pass any `score_fn(artifact) -> float` to evaluate candidates.
"""
from gegenextract.optimization.artifact import PromptArtifact
from gegenextract.optimization.mutation import MutationEngine, DEFAULT_OPERATORS, Mutation
from gegenextract.optimization.candidate import CandidateGenerator
from gegenextract.optimization.selection import SelectionEngine
from gegenextract.optimization.tracker import OptimizationTracker
from gegenextract.optimization.analysis import TrajectoryAnalyzer
import yaml


def sample_score_fn(artifact: PromptArtifact) -> float:
    # naive score: prefer longer instructions (placeholder)
    instr = artifact.instructions or ""
    return len(instr)


def run_simple(seed_artifact: PromptArtifact, config_path: str = "configs/optimization.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pop = cfg["population"]["size"]
    beam = cfg["population"]["beam_width"]
    mut_probs = cfg["mutation"]["probabilities"]
    # instantiate engines
    me = MutationEngine(DEFAULT_OPERATORS, seed=cfg["mutation"].get("seed", 0))
    cg = CandidateGenerator(me, population_size=pop, beam_width=beam)
    sel = SelectionEngine(acceptance_delta=cfg["selection"]["acceptance_delta"], stagnation_gens=cfg["selection"]["stagnation_gens"])
    tracker = OptimizationTracker(path="opt_run_sample.json")
    analyzer = TrajectoryAnalyzer(tracker)

    # generate simple mutations list from config keys
    muts = [(k, {}) for k in mut_probs.keys()]

    for gen in range(3):
        candidates = cg.generate(seed_artifact, muts)
        # evaluate
        for c in candidates:
            sc = sample_score_fn(c.artifact)
            c.score = sc
        # record
        recs = []
        for c in candidates:
            recs.append({
                "artifact_id": c.artifact.id,
                "mutation_name": c.artifact.metadata.get("mutation_name"),
                "score": c.score,
            })
        tracker.record_generation(gen, recs)
        accepted = sel.select(candidates)
        if accepted:
            seed_artifact = accepted[0].artifact

    print("Trajectory:", analyzer.score_trajectory())


if __name__ == "__main__":
    seed = PromptArtifact(system="system", instructions="Extract fields as JSON.")
    run_simple(seed)
