from gegenextract.optimization.tracker import OptimizationTracker
from gegenextract.optimization.analysis import TrajectoryAnalyzer
import os


def test_tracker_and_trajectory(tmp_path):
    p = tmp_path / "opt.json"
    t = OptimizationTracker(path=str(p))
    t.record_generation(0, [{"artifact_id": "a1", "score": 0.1, "mutation_name": "seed"}])
    t.record_generation(1, [{"artifact_id": "a2", "score": 0.5, "mutation_name": "m1"}])
    ta = TrajectoryAnalyzer(t)
    traj = ta.score_trajectory()
    assert traj[0][1] == 0.1
    assert traj[1][1] == 0.5
