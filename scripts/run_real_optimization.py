"""Run a real optimization connecting extraction engine and evaluator via FitnessAdapter."""
from gegenextract.experiment.runner import ExperimentRunner
from gegenextract.optimization.fitness_adapter import FitnessAdapter
from gegenextract.extraction.engine import ExtractionEngine
from gegenextract.scoring.evaluator import Evaluator
from gegenextract.extraction.schema_loader import SchemaLoader
from gegenextract.optimization.artifact import PromptArtifact
from gegenextract.experiment.persistence import PersistenceManager


def build_extraction_fn():
    # instantiate real extraction engine (user should configure it properly)
    engine = ExtractionEngine()

    def extraction_fn(artifact: PromptArtifact, sample: dict):
        # the engine expects prompt artifact and sample; adapt as needed
        return engine.extract(sample["document"], artifact)

    return extraction_fn


def build_evaluator():
    return Evaluator()


def main():
    persistence = PersistenceManager("daten/experiments.db")
    extraction_fn = build_extraction_fn()
    evaluator = build_evaluator()

    def evaluator_fn(pred, gold):
        # Evaluator.score expects prediction and gold dicts
        res = evaluator.score(pred, gold)
        # convert FieldMetric objects to dict-like
        out = {}
        for k, v in res.items():
            out[k] = {"precision": v.precision, "recall": v.recall, "f1": v.f1}
        return out

    fa = FitnessAdapter(persistence, extraction_fn, evaluator_fn, cache_enabled=True)

    def score_fn(artifact: PromptArtifact) -> float:
        # TODO: get validation samples from dataset; here we use placeholder
        samples = []
        score, report = fa.evaluate_prompt(artifact, samples)
        return score

    runner = ExperimentRunner("configs/experiment.yaml", score_fn)
    traj = runner.run()
    print("Trajectory:", traj)


if __name__ == "__main__":
    main()
