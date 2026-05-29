from gegenextract.experiment.real_runner import RealExperimentRunner


def main():
    runner = RealExperimentRunner("configs/experiment.yaml", dataset_root="data/hiring")
    summary = runner.run(max_generations=5)
    print(summary)


if __name__ == "__main__":
    main()
