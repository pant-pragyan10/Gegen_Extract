import json
import os
from pathlib import Path
import yaml
import traceback
from gegenextract.experiment.real_runner import RealExperimentRunner
from gegenextract.scoring.evaluator import Evaluator


def run_demo(exp_id='ui_run_demo', max_gens=5, seed=42):
    exp_path = Path('experiments') / exp_id
    os.makedirs(exp_path, exist_ok=True)
    cfg = {
        'experiment': {
            'id': exp_id,
            'name': f'UI run {exp_id}',
            'seed_prompt': {'system': '', 'instructions': 'UI-run seed prompt'},
            'dataset': {'root': ''},
            'persistence': {'sqlite_path': 'experiments.db'},
            'budget': {'max_generations': int(max_gens), 'max_runtime_seconds': 600},
            'checkpoint': {'seed': int(seed)}
        }
    }
    cfg_path = exp_path / 'ui_experiment.yaml'
    with open(cfg_path, 'w') as fh:
        yaml.safe_dump(cfg, fh)

    # simple sample with gold drawn from extraction.json if present
    sample_gold = {'Name': 'Demo User', 'Email': 'demo@example.com'}
    try:
        exf = exp_path / 'extraction.json'
        if exf.exists():
            j = json.load(open(exf))
            if isinstance(j, dict) and j:
                sample_gold = j
    except Exception:
        pass

    class _Samp:
        def __init__(self, id, document, metadata):
            self.id = id
            self.document = document
            self.metadata = metadata

    class _Loader:
        def __init__(self, samples):
            self._samples = samples
        def load(self):
            return self._samples

    class _PdfProc:
        def process(self, d):
            return d

    class _Extractor:
        def extract(self, sample, artifact):
            return sample.get('gold') if isinstance(sample, dict) else sample.metadata.get('gold', {})

    samples = [_Samp('ui_demo_1', {'text': 'demo'}, {'gold': sample_gold})]
    loader = _Loader(samples)
    pdf_proc = _PdfProc()
    extractor = _Extractor()
    evaluator = Evaluator()

    logp = exp_path / 'run.log'
    try:
        runner = RealExperimentRunner(str(cfg_path), dataset_root='', output_dir=str(Path('experiments')), deterministic_seed=int(seed), extractor=extractor, evaluator=evaluator, loader=loader, pdf_proc=pdf_proc)
        summary = runner.run(max_generations=int(max_gens), split_ratio=0.8, seed=int(seed))
        with open(logp, 'a') as lf:
            lf.write('\nSUMMARY:\n')
            lf.write(json.dumps(summary, indent=2))
        print('done')
    except Exception:
        with open(logp, 'a') as lf:
            lf.write('\nERROR:\n')
            lf.write(traceback.format_exc())
        raise


if __name__ == '__main__':
    run_demo()
