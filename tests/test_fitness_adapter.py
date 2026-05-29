from gegenextract.optimization.fitness_adapter import FitnessAdapter
from gegenextract.experiment.persistence import PersistenceManager
from gegenextract.optimization.artifact import PromptArtifact


def test_fitness_adapter_caching(tmp_path):
    db = tmp_path / "exp.db"
    pm = PersistenceManager(str(db))
    # dummy extraction returns gold
    def extraction_fn(artifact, sample):
        return sample.get("gold")

    def evaluator_fn(pred, gold):
        # return single-field perfect match
        return {"root": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}

    fa = FitnessAdapter(pm, extraction_fn, evaluator_fn, cache_enabled=True)
    p = PromptArtifact(instructions="Test")
    samples = [{"gold": {"a": 1}}]
    s1, r1 = fa.evaluate_prompt(p, samples)
    s2, r2 = fa.evaluate_prompt(p, samples)
    assert s1 == s2
    assert r1 == r2
