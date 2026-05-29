import json
import os
import random
import time
from pathlib import Path

from gegenextract.experiment.real_runner import RealExperimentRunner
from gegenextract.optimization.artifact import PromptArtifact
from gegenextract.extraction.prompt_builder import PromptBuilder
from gegenextract.extraction.repair import RepairEngine
from gegenextract.extraction.groq_client import GroqClient
from gegenextract.optimization.diff_html import prompt_diff_html


class FakeGroqClient:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
        # if this is a repair prompt, return valid JSON
        if "Please repair" in prompt or "repair" in prompt.lower():
            text = '{"name": "Repaired Name"}'
            return {"text": text, "tokens": 12, "elapsed": 0.05}
        # if prompt contains strengthening phrase, try to extract gold from prompt context
        if "follow the schema strictly" in prompt:
            # find 'Name: <value>' in prompt pages
            import re

            m = re.search(r"Name:\s*(Person_\d+)", prompt)
            if m:
                name = m.group(1)
                return {"text": json.dumps({"name": name}), "tokens": 10, "elapsed": 0.02}
        # deterministic chance of malformed output
        if self.rng.random() < 0.3:
            # malformed
            return {"text": "MALFORMED_JSON: {name: 'Bad'}", "tokens": 8, "elapsed": 0.03}
        # else return valid minimal json
        name = "Person_" + str(self.rng.randint(1, 100))
        text = json.dumps({"name": name})
        return {"text": text, "tokens": 10, "elapsed": 0.02}


class FakeExtractor:
    def __init__(self, persistence, seed=0):
        self.groq = FakeGroqClient(seed=seed)
        self.prompt_builder = PromptBuilder()
        self.repair = RepairEngine(self.groq, self.prompt_builder, max_attempts=2)
        self.persistence = persistence

    def extract(self, document: dict, artifact: PromptArtifact):
        # document expected to have 'text'
        pages = [document.get("text", "")] if isinstance(document, dict) else [str(document)]
        schema = {"description": "synthetic resume", "properties": {"name": {"type": "string"}}}
        prompt = self.prompt_builder.build(schema, pages, extraction_instructions=artifact.instructions or "Extract name")
        resp = self.groq.call(prompt, temperature=0.0)
        text = resp.get("text")
        tokens = resp.get("tokens")
        elapsed = resp.get("elapsed")
        # try parse
        try:
            pred = json.loads(text)
            report_meta = {"tokens": tokens, "elapsed": elapsed, "repair_count": 0}
            return {"_prediction": pred, "_meta": report_meta}
        except Exception:
            # attempt repair via RepairEngine
            repaired = self.repair.repair(text, "parse_error", schema, pages)
            try:
                pred2 = json.loads(repaired)
                report_meta = {"tokens": tokens, "elapsed": elapsed + 0.02, "repair_count": 1}
                return {"_prediction": pred2, "_meta": report_meta}
            except Exception:
                report_meta = {"tokens": tokens, "elapsed": elapsed, "repair_count": 1}
                return {"_prediction": None, "_meta": report_meta}


def run_demo():
    out = Path("experiments/synthetic_demo")
    if out.exists():
        # clear existing
        for f in out.iterdir():
            f.unlink()
    out.mkdir(parents=True, exist_ok=True)

    runner = RealExperimentRunner("configs/experiment_synthetic.yaml", dataset_root="data/synthetic", output_dir=str(out.parent), deterministic_seed=123)
    # monkeypatch splits to create 10 validation + 2 test samples
    samples = []
    for i in range(12):
        samples.append({"document": {"text": f"Name: Person_{i}"}, "gold": {"name": f"Person_{i}"}})

    def fake_load_splits(split_ratio, seed=42):
        return {"validation": samples[:10], "test": samples[10:]}

    runner._load_splits = fake_load_splits

    # inject fake extractor
    fake_ex = FakeExtractor(runner.persistence, seed=123)
    runner.extractor = fake_ex

    # build evaluator wrapper to handle pred/meta and override fitness adapter builder
    from gegenextract.optimization.fitness_adapter import FitnessAdapter

    real_eval = runner.evaluator

    def wrapped_evaluator(pred, gold):
        meta = {}
        if isinstance(pred, dict) and "_prediction" in pred:
            meta = pred.get("_meta", {})
            real_pred = pred.get("_prediction") or {}
        else:
            real_pred = pred
        res = real_eval.score(real_pred or {}, gold or {})
        out = {k: {"precision": v.precision, "recall": v.recall, "f1": v.f1} for k, v in res.items()}
        out["__meta__"] = meta
        return out

    def build_adapter():
        return FitnessAdapter(runner.persistence, lambda art, s: runner.extractor.extract(s["document"] if isinstance(s, dict) and "document" in s else s, art), wrapped_evaluator, cache_enabled=True)

    runner._build_fitness_adapter = build_adapter

    # adjust candidate generator and mutation aggressiveness
    runner.candidate_generator.population_size = 4
    runner.candidate_generator.beam_width = 2
    # run small optimization
    summary = runner.run(max_generations=5, split_ratio=0.8, seed=123)

    exp_dir = Path(runner.output_dir) / runner.runner.experiment_id
    # create mutation_history and evaluation_breakdowns from tracker and persisted evaluations
    traj = runner.tracker.lineage()
    with open(exp_dir / "trajectory.json", "w") as f:
        json.dump(traj, f, indent=2)

    # gather evaluations from persistence DB via experiment persistence
    # experiment persistence stores evaluations in sqlite; but also we persisted via persist_evaluation
    # We'll create evaluation_breakdowns from tracker candidates
    evals = []
    for gen in traj.get("generations", []):
        for c in gen.get("candidates", []):
            evals.append(c)
    with open(exp_dir / "evaluation_breakdowns.json", "w") as f:
        json.dump(evals, f, indent=2)

    # mutation history: pull mutation metadata from persisted prompts (best effort via persistence)
    # For synthetic run, create mutation_history from trajectory
    mutations = []
    for gen in traj.get("generations", []):
        for c in gen.get("candidates", []):
            mutations.append({"generation": gen.get("generation"), **c})
    with open(exp_dir / "mutation_history.json", "w") as f:
        json.dump(mutations, f, indent=2)

    # best prompt already saved by runner; produce prompt diff vs seed
    seed_path = exp_dir / "seed_prompt.json"
    best_path = exp_dir / "best_prompt.json"
    if seed_path.exists() and best_path.exists():
        a = PromptArtifact.model_validate_json(seed_path.read_text()) if hasattr(PromptArtifact, 'model_validate_json') else PromptArtifact(**json.loads(seed_path.read_text()))
        b = PromptArtifact.model_validate_json(best_path.read_text()) if hasattr(PromptArtifact, 'model_validate_json') else PromptArtifact(**json.loads(best_path.read_text()))
        html = prompt_diff_html(a, b)
        (exp_dir / "prompt_diffs").mkdir(exist_ok=True)
        with open(exp_dir / "prompt_diffs" / "seed_vs_best.html", "w") as f:
            f.write(html)

    # generate plot
    try:
        from scripts.export_experiment_report import generate_plots

        generate_plots(str(exp_dir))
    except Exception:
        pass

    print("Synthetic run summary:", summary)


if __name__ == "__main__":
    run_demo()
