from __future__ import annotations
from typing import Dict, Any, List
import json
import os
from datetime import datetime


class OptimizationTracker:
    def __init__(self, path: str = "opt_runs.json"):
        self.path = path
        self.data: Dict[str, Any] = {"generations": []}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"generations": []}

    def record_generation(self, generation_idx: int, candidates: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        entry = {
            "generation": generation_idx,
            "time": datetime.utcnow().isoformat(),
            "candidates": candidates,
            "metadata": metadata or {},
        }
        self.data.setdefault("generations", []).append(entry)
        self._persist()

    def _persist(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def lineage(self) -> Dict[str, Any]:
        return self.data
