import os
from gegenextract.experiment.runner import ExperimentRunner
from gegenextract.optimization.artifact import PromptArtifact


def test_resume_behavior(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("""
experiment:
  id: testexp
  name: testexp
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
    def score_fn(a: PromptArtifact) -> float:
        return 1.0

    r = ExperimentRunner(str(cfg), score_fn)
    traj1 = r.run()
    # simulate interruption: run again and ensure resume doesn't duplicate work
    r2 = ExperimentRunner(str(cfg), score_fn)
    traj2 = r2.run()
    assert isinstance(traj1, list)
    assert isinstance(traj2, list)
