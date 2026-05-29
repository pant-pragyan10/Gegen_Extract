import json
from pathlib import Path
import matplotlib.pyplot as plt


def generate_plots(exp_dir: str):
    p = Path(exp_dir)
    traj = json.load(open(p / "trajectory.json")) if (p / "trajectory.json").exists() else {}
    gens = traj.get("generations", [])
    best_scores = []
    for g in gens:
        scores = [c.get("score", 0) for c in g.get("candidates", [])]
        best_scores.append(max(scores) if scores else 0)
    plt.plot(best_scores)
    plt.title("Best score over generations")
    plt.xlabel("generation")
    plt.ylabel("score")
    plt.savefig(p / "best_score_plot.png")


def main():
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else "experiments"
    generate_plots(exp)


if __name__ == "__main__":
    main()
