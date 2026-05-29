# Experiment Report — hiring_res_001

This document summarizes the controlled optimization run performed on the ExtractBench hiring/resume subset.

## High-level results
- Experiment dir: `experiments/hiring_run_1/hiring_res_001`
- Runs: 5 generations, population candidates per generation: 3 (per seed mutation set)
- Persisted artifacts: `trajectory.json`, `mutation_analysis.json`, `score_over_generations.png`, `mutation_mean_scores.png`, `prompt_diffs/seed_vs_best.html`

## Numeric summary (from `optimization_summary.json`)

```
{
  "seed_score": 0.0,
  "best_validation_score": 0.0,
  "final_test_score": 0.0,
  "best_generation": 0,
  "total_generations": 5
}
```

Note: per-sample evaluations recorded in the DB show non-zero scores for individual candidates (max per-generation candidate score reached 1.0). The overall `best_validation_score` field remained 0.0 because the runner's baseline aggregation was configured differently for this run; see `mutation_analysis.json` for per-generation and per-mutation stats.

## Key findings (from `mutation_analysis.json`)

- Best observed candidate score (per-generation max): 1.0 (achieved in generation 0 and persisted across generations).
- Mean score per generation (mean of per-sample evaluations): generation means ≈ [0.228, 0.225, 0.241, 0.25, 0.241].
- Mutation effectiveness: the `reduce_hallucination` family (applied to many artifacts) had mean score ≈ 0.237 (median 0.0), indicating mixed impact: some candidates scored perfectly while many scored 0.
- Accepted vs rejected: the selected-best candidate per generation often corresponded to `rewrite_instruction` and `strengthen_constraint` mutations; rejected top-scoring alternatives included `reduce_hallucination` variants with competitive scores.
- Stagnation: 4 stagnant generations vs 1 improving generation (initial high-scoring candidate established early, limited further improvement).

## Plots & visual artifacts

- `score_over_generations.png` — score trajectory (mean and max per generation).
- `mutation_mean_scores.png` — mean score per mutation category.
- `prompt_diffs/seed_vs_best.html` — HTML diff between seed and best prompt versions.

## Mutation / repair observations

- Many model outputs include markdown/code fences around JSON; the repair loop was effective when invoked: repair prompts were updated to require "ONLY the fixed JSON object" and we strip code fences before parsing; this recovered valid JSON in repair attempts.
- Repair-related improvements are visible where initial parse failed but the repair output parsed successfully and produced correct fields (see `mutation_analysis.json` accepted/rejected and DB `evaluations` diagnostics).

## Engineering observations

- Groq endpoint migration: client updated to use OpenAI-compatible chat endpoint `https://api.groq.com/openai/v1/chat/completions` with `messages` payload; API key handling sanitized.
- Malformed JSON handling: added code-fence stripping and conservative repair prompts; extraction engine now returns repaired parsed JSON when schema validation is not provided.
- OCR fallback: `pdf2image` (Poppler) not installed in this environment; OCR remains optional and was not required for these samples. If OCR is needed, installing Poppler (macOS: `brew install poppler`) will enable page-image conversion.

## Limitations

- Small run (5 generations) on a tiny dataset (7 resumes) — results are preliminary.
- Baseline aggregation semantics caused `optimization_summary.json` to report zeros; use `mutation_analysis.json` for reliable per-generation metrics.

## Next steps (recommended)

1. Re-run with slightly larger validation split (10–15 val samples) and ensure `runner` aggregates per-prompt validation means into the experiment-level summary.
2. Collect more per-field diagnostic metrics (per-field precision/recall) to produce field-level breakdown plots.
3. Optionally enable OCR for any OCR-only PDFs.

---

Files produced in this run live under the experiment folder. Use the HTML diff and PNG files for inclusion in a report or demo.

