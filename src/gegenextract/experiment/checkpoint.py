import os
import json
from typing import Dict, Any, Optional


class CheckpointManager:
    def __init__(self, path: str = "checkpoints"):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def checkpoint_path(self, experiment_id: str, generation: int) -> str:
        return os.path.join(self.path, f"{experiment_id}_gen{generation}.json")

    def save(self, experiment_id: str, generation: int, state: Dict[str, Any]):
        p = self.checkpoint_path(experiment_id, generation)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, p)

    def load_latest(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        # find latest by generation suffix
        candidates = []
        for fn in os.listdir(self.path):
            if fn.startswith(experiment_id + "_") and fn.endswith('.json'):
                parts = fn.split('gen')
                if len(parts) >= 2:
                    gen = int(parts[-1].split('.json')[0])
                    candidates.append((gen, fn))
        if not candidates:
            return None
        latest = sorted(candidates, key=lambda x: x[0])[-1][1]
        with open(os.path.join(self.path, latest)) as f:
            return json.load(f)
