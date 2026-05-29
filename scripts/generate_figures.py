#!/usr/bin/env python3
import json
from pathlib import Path
import os

exp = Path('experiments/hiring_run_1/hiring_res_001')
figs = exp / 'figures'
shots = exp / 'screenshots'
figs.mkdir(parents=True, exist_ok=True)
shots.mkdir(parents=True, exist_ok=True)

def ensure_pkg(pkg):
    try:
        __import__(pkg)
        return True
    except Exception:
        return False

if not ensure_pkg('plotly'):
    raise SystemExit('plotly not installed in this interpreter')

import plotly.express as px
import pandas as pd

# Field metrics
fd = exp / 'field_diagnostics.json'
if fd.exists():
    d = json.load(open(fd))
    fs = d.get('field_summary', {})
    rows = []
    for f,v in fs.items():
        rows.append({'field':f,'f1': v.get('f1_mean') or 0,'precision': v.get('precision_mean') or 0,'recall': v.get('recall_mean') or 0})
    rows = sorted(rows, key=lambda x: x['f1'], reverse=True)
    if rows:
        df = pd.DataFrame(rows)
        fig = px.bar(df, x='field', y='f1', title='Field-level F1 (higher is better)')
        fig.update_layout(template='plotly_white')
        fig.write_image(str(figs / 'field_metrics.png'), width=1600, height=800)
        fig.write_image(str(shots / 'field_metrics.png'), width=1600, height=800)

# Mutation effectiveness
ma = exp / 'mutation_analysis.json'
if ma.exists():
    m = json.load(open(ma))
    ms = m.get('mutation_stats', {})
    rows = []
    for name, info in ms.items():
        rows.append({'mutation': name, 'mean_score': info.get('mean_score') or 0, 'count': info.get('count',0)})
    if rows:
        dfm = pd.DataFrame(rows).sort_values('mean_score', ascending=False)
        figm = px.bar(dfm, x='mutation', y='mean_score', title='Mutation mean score')
        figm.update_layout(template='plotly_white')
        figm.write_image(str(figs / 'mutation_mean_scores.png'), width=1600, height=800)
        figm.write_image(str(shots / 'mutation_mean_scores.png'), width=1600, height=800)

print('Generated figures in', figs)
