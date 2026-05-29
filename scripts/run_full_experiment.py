from gegenextract.experiment.runner import ExperimentRunner
from gegenextract.optimization.artifact import PromptArtifact


def score_fn(artifact: PromptArtifact) -> float:
    # placeholder scoring: length of instructions
    return len(artifact.instructions or "")


def main():
    runner = ExperimentRunner("configs/experiment.yaml", score_fn)
    traj = runner.run()
    print("Trajectory:", traj)


if __name__ == "__main__":
    main()
