import json
import sys
from pathlib import Path


def summarize(exp_dir: str):
    p = Path(exp_dir)
    summary = {}
    if (p / "optimization_summary.json").exists():
        summary = json.load(open(p / "optimization_summary.json"))
    traj = json.load(open(p / "trajectory.json")) if (p / "trajectory.json").exists() else {}
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("Trajectory generations:", len(traj.get("generations", [])))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments"
    summarize(path)
