from gegenextract.optimization.fitness_adapter import FitnessAdapter
from gegenextract.experiment.persistence import PersistenceManager
from gegenextract.optimization.artifact import PromptArtifact


def main():
    pm = PersistenceManager("daten/experiments.db")
    # example prompt
    p = PromptArtifact(instructions="Extract fields as JSON.")
    # placeholder extraction and evaluator to be provided by user
    def dummy_extraction_fn(artifact, sample):
        return sample.get("gold", {})

    def dummy_evaluator_fn(pred, gold):
        # perfect match
        return {k: {"precision": 1.0, "recall": 1.0, "f1": 1.0} for k in gold.keys()}

    fa = FitnessAdapter(pm, dummy_extraction_fn, dummy_evaluator_fn)
    samples = [{"gold": {"a": 1}}, {"gold": {"a": 1}}]
    score, report = fa.evaluate_prompt(p, samples)
    print(score, report)


if __name__ == "__main__":
    main()
