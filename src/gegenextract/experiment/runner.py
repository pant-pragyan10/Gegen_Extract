import uuid
import time
from typing import Callable, Dict, Any, Optional
import yaml
from .schemas import ExperimentConfig
from .persistence import PersistenceManager
from .checkpoint import CheckpointManager
from gegenextract.optimization.mutation import MutationEngine, DEFAULT_OPERATORS
from gegenextract.optimization.candidate import CandidateGenerator
from gegenextract.optimization.selection import SelectionEngine
from gegenextract.optimization.tracker import OptimizationTracker
from gegenextract.optimization.analysis import TrajectoryAnalyzer
from gegenextract.optimization.artifact import PromptArtifact


class ExperimentRunner:
    def __init__(self, config_path: str, score_fn: Callable[[PromptArtifact], float], extraction_fn: Optional[Callable] = None):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.config = ExperimentConfig.model_validate(cfg["experiment"]) if hasattr(ExperimentConfig, 'model_validate') else ExperimentConfig(**cfg["experiment"])
        self.experiment_id = self.config.id or str(uuid.uuid4())
        self.score_fn = score_fn
        self.extraction_fn = extraction_fn
        self.persistence = PersistenceManager(self.config.persistence.get("sqlite_path", "experiments.db"))
        self.checkpoint = CheckpointManager()
        self.mutation_engine = MutationEngine(DEFAULT_OPERATORS, seed=self.config.checkpoint.get("seed", 0) if isinstance(self.config.checkpoint, dict) and self.config.checkpoint.get("seed") else 0)
        self.candidate_generator = CandidateGenerator(self.mutation_engine, population_size=8, beam_width=3)
        self.selection_engine = SelectionEngine()
        self.tracker = OptimizationTracker(path=f"opt_{self.experiment_id}.json")
        self.analyzer = TrajectoryAnalyzer(self.tracker)
        self.start_time = time.time()

    def run(self):
        # persist config
        self.persistence.persist_experiment(self.experiment_id, self.config.name, self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.dict())
        # create seed artifact
        seed_cfg = self.config.seed_prompt
        seed_art = PromptArtifact(system=seed_cfg.get("system"), instructions=seed_cfg.get("instructions"))
        # check for checkpoint
        ck = self.checkpoint.load_latest(self.experiment_id)
        start_gen = ck.get("generation", 0) + 1 if ck else 0
        seed_artifact = PromptArtifact(**ck.get("artifact")) if ck and ck.get("artifact") else seed_art

        max_gens = self.config.budget.get("max_generations", 10)
        for gen in range(start_gen, max_gens):
            # produce candidate mutations (simple: use operators keys)
            mut_keys = [(k, {}) for k in self.mutation_engine.operators.keys()]
            cands = self.candidate_generator.generate(seed_artifact, mut_keys)
            # evaluate
            recs = []
            for c in cands:
                score = self.score_fn(c.artifact)
                c.score = score
                # persist prompt and evaluation
                self.persistence.persist_prompt(self.experiment_id, c.artifact.id, c.artifact.version, c.artifact.model_dump() if hasattr(c.artifact, 'model_dump') else c.artifact.dict(), c.artifact.metadata)
                self.persistence.persist_evaluation(self.experiment_id, gen, c.artifact.id, score, {"score": score})
                recs.append({"artifact_id": c.artifact.id, "score": score, "mutation_name": c.artifact.metadata.get("mutation_name")})
            self.persistence.persist_generation(self.experiment_id, gen, {"candidates": recs})
            self.tracker.record_generation(gen, recs)
            # select
            accepted = self.selection_engine.select(cands)
            if accepted:
                seed_artifact = accepted[0].artifact
            # checkpoint
            state = {"generation": gen, "artifact": seed_artifact.model_dump() if hasattr(seed_artifact, 'model_dump') else seed_artifact.dict()}
            self.checkpoint.save(self.experiment_id, gen, state)
            # budget check: runtime
            if time.time() - self.start_time > self.config.budget.get("max_runtime_seconds", 600):
                break
        return self.analyzer.score_trajectory()
