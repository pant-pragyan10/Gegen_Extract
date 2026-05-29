from __future__ import annotations
from typing import Dict, Any, List, Tuple
from .tracker import OptimizationTracker


class TrajectoryAnalyzer:
    def __init__(self, tracker: OptimizationTracker):
        self.tracker = tracker

    def score_trajectory(self) -> List[Tuple[int, float]]:
        traj: List[Tuple[int, float]] = []
        for gen in self.tracker.lineage().get("generations", []):
            scores = [c.get("score", 0.0) for c in gen.get("candidates", [])]
            best = max(scores) if scores else 0.0
            traj.append((gen.get("generation", -1), best))
        return traj

    def mutation_effectiveness(self) -> Dict[str, float]:
        counts = {}
        gains = {}
        prev_best = None
        for gen in self.tracker.lineage().get("generations", []):
            best = 0.0
            for c in gen.get("candidates", []):
                mname = c.get("mutation_name") or "seed"
                counts[mname] = counts.get(mname, 0) + 1
                sc = c.get("score", 0.0)
                gains[mname] = gains.get(mname, 0.0) + sc
                best = max(best, sc)
            prev_best = best
        # average gain per mutation
        return {k: (gains.get(k, 0.0) / counts[k] if counts[k] else 0.0) for k in counts}
