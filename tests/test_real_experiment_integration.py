import os
from gegenextract.experiment.real_runner import RealExperimentRunner
from gegenextract.optimization.artifact import PromptArtifact


def test_real_experiment_with_synthetic_data(tmp_path, monkeypatch):
    # create runner
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("""
experiment:
  id: synexp
  name: synexp
  seed_prompt:
    system: S
    instructions: Start
  dataset:
    split: 0.8
  persistence:
    sqlite_path: """ + str(tmp_path / "exp.db") + """
  budget:
    max_generations: 2
  checkpoint:
    interval_seconds: 1
""")
    runner = RealExperimentRunner(str(cfg_path), dataset_root=str(tmp_path))

    # monkeypatch _load_splits to return synthetic samples
    def fake_load_splits(split_ratio, seed=42):
        # create two samples with gold
        samples = [{"document": {"text": "John Doe, Software Engineer"}, "gold": {"name": "John Doe"}},
                   {"document": {"text": "Jane Smith, Data Scientist"}, "gold": {"name": "Jane Smith"}}]
        return {"validation": samples[:1], "test": samples[1:]}

    monkeypatch.setattr(runner, "_load_splits", fake_load_splits)

    # monkeypatch fitness adapter to simple evaluator returning perfect score when prediction equals gold
    def fake_build_fitness_adapter():
        class FA:
            def evaluate_prompt(self, artifact, samples):
                # if artifact.instructions contains 'Start' return 0.5 else 1.0
                score = 0.5 if 'Start' in (artifact.instructions or '') else 1.0
                return score, {"score": score}
        return FA()

    monkeypatch.setattr(runner, "_build_fitness_adapter", fake_build_fitness_adapter)

    summary = runner.run(max_generations=2, split_ratio=0.8, seed=42)
    assert "seed_score" in summary
    assert "final_test_score" in summary
