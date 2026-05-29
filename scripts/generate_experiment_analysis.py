#!/usr/bin/env python3
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import matplotlib.pyplot as plt


def load_trajectory(exp_dir: Path):
    traj_file = exp_dir / "trajectory.json"
    if not traj_file.exists():
        return {}
    return json.loads(traj_file.read_text())


def load_db(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    return conn, cur


def fetch_prompts(cur):
    prompts = {}
    for row in cur.execute('SELECT id,metadata FROM prompts'):
        pid, meta = row
        try:
            m = json.loads(meta) if meta else {}
        except Exception:
            m = {}
        prompts[pid] = m
    return prompts


def fetch_generations(cur, experiment_id):
    gens = []
    for row in cur.execute('SELECT generation, summary FROM generations WHERE experiment_id=? ORDER BY generation', (experiment_id,)):
        gen, summary = row
        try:
            s = json.loads(summary) if summary else {}
        except Exception:
            s = {}
        gens.append({'generation': gen, 'summary': s})
    return gens


def fetch_evaluations(cur, experiment_id):
    rows = []
    for row in cur.execute('SELECT generation,artifact_id,score,report FROM evaluations WHERE experiment_id=? ORDER BY generation', (experiment_id,)):
        gen, aid, score, report = row
        try:
            rep = json.loads(report) if report else {}
        except Exception:
            rep = {}
        rows.append({'generation': gen, 'artifact_id': aid, 'score': score, 'report': rep})
    return rows


def analyze(exp_dir: Path, db_path: Path, experiment_id: str):
    traj = load_trajectory(exp_dir)
    conn, cur = load_db(db_path)
    prompts = fetch_prompts(cur)
    gens = fetch_generations(cur, experiment_id)
    evals = fetch_evaluations(cur, experiment_id)

    # per-generation scores
    gen_scores = {}
    for e in evals:
        gen_scores.setdefault(e['generation'], []).append(e['score'])

    gen_summary = []
    for g, scores in sorted(gen_scores.items()):
        gen_summary.append({'generation': g, 'count': len(scores), 'mean': statistics.mean(scores) if scores else 0.0, 'stdev': statistics.pstdev(scores) if len(scores)>1 else 0.0, 'max': max(scores) if scores else 0.0})

    # mutation effectiveness
    mut_stats = {}
    # map artifact -> mutation
    for aid, m in prompts.items():
        name = m.get('mutation_name', 'seed')
        mut_stats.setdefault(name, {'count': 0, 'scores': []})

    for e in evals:
        meta = prompts.get(e['artifact_id'], {})
        name = meta.get('mutation_name', 'unknown')
        mut = mut_stats.setdefault(name, {'count': 0, 'scores': []})
        mut['count'] += 1
        mut['scores'].append(e['score'])

    for k, v in mut_stats.items():
        v['mean_score'] = statistics.mean(v['scores']) if v['scores'] else 0.0
        v['median_score'] = statistics.median(v['scores']) if v['scores'] else 0.0

    # accepted vs rejected: derive best candidate per generation from generations.summary
    accepted = []
    rejected = []
    for g in gens:
        cands = g['summary'].get('candidates', [])
        if not cands:
            continue
        # best by score
        best = max(cands, key=lambda x: x.get('score', 0.0))
        accepted.append({'generation': g['generation'], 'artifact_id': best.get('artifact_id'), 'score': best.get('score'), 'mutation_name': best.get('mutation_name')})
        for c in cands:
            if c.get('artifact_id') != best.get('artifact_id') or c.get('score') != best.get('score'):
                rejected.append({'generation': g['generation'], 'artifact_id': c.get('artifact_id'), 'score': c.get('score'), 'mutation_name': c.get('mutation_name')})

    # stagnation analysis
    best_so_far = -1.0
    stagnation = {'stagnant_generations': 0, 'improving_generations': 0, 'best_progress': []}
    for gs in sorted(gen_summary, key=lambda x: x['generation']):
        if gs['max'] > best_so_far:
            stagnation['improving_generations'] += 1
            best_so_far = gs['max']
            stagnation['best_progress'].append({'generation': gs['generation'], 'new_best': best_so_far})
        else:
            stagnation['stagnant_generations'] += 1

    # best generation summary
    best_gen = max(gen_summary, key=lambda x: x['max']) if gen_summary else None

    out = {
        'generation_summary': gen_summary,
        'mutation_stats': mut_stats,
        'accepted': accepted,
        'rejected': rejected,
        'stagnation': stagnation,
        'best_generation': best_gen,
    }

    # write outputs
    (exp_dir / 'mutation_analysis.json').write_text(json.dumps(out, indent=2))

    # plots
    gens_x = [g['generation'] for g in gen_summary]
    means = [g['mean'] for g in gen_summary]
    maxs = [g['max'] for g in gen_summary]
    plt.figure()
    plt.plot(gens_x, means, marker='o', label='mean')
    plt.plot(gens_x, maxs, marker='x', label='max')
    plt.xlabel('generation')
    plt.ylabel('score')
    plt.title('Score over generations')
    plt.legend()
    plt.grid(True)
    plt.savefig(exp_dir / 'score_over_generations.png')

    # mutation effectiveness bar
    names = list(mut_stats.keys())
    means = [mut_stats[n]['mean_score'] for n in names]
    plt.figure(figsize=(8,4))
    plt.bar(names, means)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('mean score')
    plt.title('Mutation mean score')
    plt.tight_layout()
    plt.savefig(exp_dir / 'mutation_mean_scores.png')

    # write basic markdown report
    md = []
    md.append('# Experiment Report')
    md.append('\n')
    md.append('## Summary')
    summary_file = exp_dir / 'optimization_summary.json'
    if summary_file.exists():
        summary_text = summary_file.read_text()
        md.append('```\n' + summary_text + '\n```')
    md.append('\n')
    md.append('## Best generation')
    md.append(json.dumps(best_gen, indent=2))
    md.append('\n')
    md.append('## Stagnation')
    md.append(json.dumps(stagnation, indent=2))
    md.append('\n')
    md.append('## Mutation stats (top 10)')
    top_mut = sorted([(k,v['mean_score']) for k,v in mut_stats.items()], key=lambda x: -x[1])
    md.append('\n'.join([f'- {k}: mean_score={v:.4f}, count={mut_stats[k]["count"]}' for k,v in top_mut]))
    md.append('\n')
    (exp_dir / 'experiment_report.md').write_text('\n'.join(md))

    # architecture summary
    arch = []
    arch.append('# Architecture Summary')
    arch.append('Pipeline: ingestion -> document processing -> extraction (Groq) -> repair -> evaluation -> optimizer')
    arch.append('\n')
    (exp_dir / 'architecture_summary.md').write_text('\n'.join(arch))

    conn.close()
    print('Analysis written to', exp_dir)


if __name__ == '__main__':
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('experiments/hiring_run_1/hiring_res_001')
    db_path = Path('experiments/hiring_run_1/experiments.db')
    experiment_id = 'hiring_res_001'
    analyze(exp_dir, db_path, experiment_id)
