import os
import json
import time
from typing import List, Dict, Any, Optional
from gegenextract.experiment.runner import ExperimentRunner
from gegenextract.experiment.persistence import PersistenceManager
from gegenextract.experiment.checkpoint import CheckpointManager
from gegenextract.optimization.fitness_adapter import FitnessAdapter
from gegenextract.optimization.artifact import PromptArtifact
from gegenextract.dataset.loader import DatasetLoader
from gegenextract.document_processing.pdf_processor import PdfProcessor
from gegenextract.extraction.engine import ExtractionEngine
from gegenextract.scoring.evaluator import Evaluator
from gegenextract.optimization.mutation import MutationEngine, DEFAULT_OPERATORS
from gegenextract.optimization.candidate import CandidateGenerator
from gegenextract.optimization.selection import SelectionEngine
from gegenextract.optimization.tracker import OptimizationTracker
from gegenextract.optimization.analysis import TrajectoryAnalyzer
 


class RealExperimentRunner:
    def __init__(self, config_path: str, dataset_root: str, output_dir: str = "experiments", deterministic_seed: int = 42, extractor=None, evaluator=None, loader=None, pdf_proc=None):
        self.config_path = config_path
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.runner = ExperimentRunner(config_path, score_fn=lambda x: 0.0)
        self.persistence = self.runner.persistence
        self.checkpoint = CheckpointManager()
        self.mutation_engine = MutationEngine(DEFAULT_OPERATORS, seed=deterministic_seed)
        self.candidate_generator = CandidateGenerator(self.mutation_engine, population_size=8, beam_width=3)
        self.selection_engine = SelectionEngine()
        self.tracker = OptimizationTracker(path=os.path.join(self.output_dir, f"opt_{int(time.time())}.json"))
        self.analyzer = TrajectoryAnalyzer(self.tracker)
        # dataset tools
        self.loader = loader or DatasetLoader(self.dataset_root)
        self.pdf_proc = pdf_proc or PdfProcessor()
        self.extractor = extractor or ExtractionEngine(groq_client=None, prompt_builder=None)
        self.evaluator = evaluator or Evaluator()

    def _load_splits(self, split_ratio: float, seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
        # load sample list from DatasetLoader and build deterministic splits by id
        all_samples = {s.id: s for s in self.loader.load()}
        ids = list(all_samples.keys())
        # use a small validation ratio of 0.2 by default
        ratios = {"train": split_ratio, "val": 0.2}
        splits = self.loader.deterministic_split(ids, seed=seed, ratios=ratios) if hasattr(self.loader, 'deterministic_split') else None
        if splits is None:
            # fallback: simple split
            n = len(ids)
            n_train = int(split_ratio * n)
            n_val = max(1, int(0.2 * n))
            train_ids = ids[:n_train]
            val_ids = ids[n_train : n_train + n_val]
            test_ids = ids[n_train + n_val :]
        else:
            train_ids = splits.train
            val_ids = splits.val
            test_ids = splits.test

        def to_samples_from_ids(id_list):
            samples = []
            for sid in id_list:
                base = all_samples.get(sid)
                if base is None:
                    continue
                doc = self.pdf_proc.process(base.document)
                gold = base.metadata.get('gold', {})
                samples.append({"document": doc, "gold": gold})
            return samples

        val = to_samples_from_ids(val_ids)
        test_s = to_samples_from_ids(test_ids)
        return {"validation": val, "test": test_s}

    def _build_fitness_adapter(self):
        def extraction_fn(artifact: PromptArtifact, sample: Dict[str, Any]):
            # pass the whole sample dict to the extractor so it can access processed document and gold
            return self.extractor.extract(sample, artifact)

        def evaluator_fn(pred, gold):
            res = self.evaluator.score(pred, gold)
            out = {}
            for k, v in res.items():
                out[k] = {"precision": v.precision, "recall": v.recall, "f1": v.f1}
            return out

        return FitnessAdapter(self.persistence, extraction_fn, evaluator_fn, cache_enabled=True)

    def run(self, max_generations: int = 10, split_ratio: float = 0.8, seed: int = 42):
        splits = self._load_splits(split_ratio, seed)
        val_samples = splits["validation"]
        test_samples = splits["test"]
        fa = self._build_fitness_adapter()
        # seed prompt
        seed_prompt_cfg = self.runner.config.seed_prompt
        seed_art = PromptArtifact(system=seed_prompt_cfg.get("system"), instructions=seed_prompt_cfg.get("instructions"))
        # baseline
        baseline_score, baseline_report = fa.evaluate_prompt(seed_art, val_samples)
        # persist baseline
        exp_dir = os.path.join(self.output_dir, self.runner.experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "seed_prompt.json"), "w") as f:
            f.write(seed_art.serialize())
        # optimization loop
        current = seed_art
        best = {"artifact": current, "score": baseline_score, "generation": 0}
        generation = 0
        while generation < max_generations:
            # generate mutations list based on operators
            muts = [(k, {}) for k in self.mutation_engine.operators.keys()]
            candidates = self.candidate_generator.generate(current, muts)
            # evaluate candidates on validation split
            recs = []
            for c in candidates:
                score, report = fa.evaluate_prompt(c.artifact, val_samples)
                c.score = score
                recs.append({"artifact_id": c.artifact.id, "score": score, "mutation_name": c.artifact.metadata.get("mutation_name")})
                # persist prompt artifact
                self.persistence.persist_prompt(self.runner.experiment_id, c.artifact.id, c.artifact.version, c.artifact.model_dump() if hasattr(c.artifact, 'model_dump') else c.artifact.dict(), c.artifact.metadata)
                self.persistence.persist_evaluation(self.runner.experiment_id, generation, c.artifact.id, score, report)
            self.tracker.record_generation(generation, recs)
            # selection
            accepted = self.selection_engine.select(candidates)
            # persistence of generation summary
            self.persistence.persist_generation(self.runner.experiment_id, generation, {"candidates": recs})
            # update best
            if accepted and accepted[0].score > best["score"]:
                best = {"artifact": accepted[0].artifact, "score": accepted[0].score, "generation": generation}
            # advance seed
            if accepted:
                current = accepted[0].artifact
            generation += 1
            # checkpoint
            self.checkpoint.save(self.runner.experiment_id, generation, {"generation": generation, "artifact": current.model_dump() if hasattr(current, 'model_dump') else current.dict()})
            # early stopping: stagnation or no improvement over many gens (handled by selection engine state)
        # final evaluation on test set
        final_score, final_report = fa.evaluate_prompt(best["artifact"], test_samples)
        # save artifacts
        with open(os.path.join(exp_dir, "best_prompt.json"), "w") as f:
            f.write(best["artifact"].serialize())
        with open(os.path.join(exp_dir, "trajectory.json"), "w") as f:
            json.dump(self.tracker.lineage(), f, indent=2)
        summary = {
            "seed_score": baseline_score,
            "best_validation_score": best["score"],
            "final_test_score": final_score,
            "best_generation": best["generation"],
            "total_generations": generation,
        }
        with open(os.path.join(exp_dir, "optimization_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary
