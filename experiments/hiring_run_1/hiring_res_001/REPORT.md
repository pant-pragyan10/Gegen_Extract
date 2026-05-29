# GegenExtract — Experiment Report

Date: 2026-05-26

## 1 Executive Summary

Project goal: build a robust, repair-aware, evaluator-guided system for extracting structured information from documents (resumes/hiring bench) and automatically optimize extraction prompts to improve field-level accuracy.

Autonomous prompt optimization: the system performs conservative, deterministic optimization runs that mutate prompts and constraints, evaluate candidate outputs at the field level, and select improvements while relying on a repair loop to recover malformed LLM outputs. The optimizer is configured for low-temperature, small-beam exploration to prioritize safe, reproducible improvements.

Key findings:

- The pipeline produced candidate outputs that reached per-sample maximal scores (1.0) early in the run; mean per-generation scores hovered ~0.22–0.25 across five generations.
- Mutation families show mixed impact: `reduce_hallucination` had a mean score ≈ 0.237 with many zero-scoring candidates and occasional perfect candidates, while `rewrite_instruction` and `strengthen_constraint` often produced the selected best artifacts.
- The repair loop meaningfully recovers code-fenced or malformed JSON, enabling valid downstream evaluations in a number of cases.

## 2 Problem Statement

Structured extraction from noisy LLM output is brittle: naive free-form prompts often produce inconsistent, incomplete or hallucinated structures, and model output formatting (markdown fences, partial JSON) prevents direct parsing. This instability undermines deterministic evaluation and makes prompt engineering manually intensive.

GegenExtract addresses three challenges: (1) stabilize formatting and schema adherence, (2) automatically search prompt variations to improve per-field accuracy, and (3) provide strong observability for auditing, repair, and rollback.

## 3 System Architecture

The system is composed of modular components that separate concerns for reliability and auditability.

- Ingestion: dataset loading and deterministic split logic (validation/test) implemented in `src/gegenextract/dataset/loader.py`.
- Document processing: PDF → text and optional OCR fallback (`src/gegenextract/document_processing/pdf_processor.py`). Primary extractor: PyMuPDF; optional OCR uses `pdf2image` + `pytesseract` when Poppler is installed.
- Extraction engine: schema-aware prompt construction and submission to Groq/OpenAI-compatible chat completions (`src/gegenextract/extraction/engine.py` and `src/gegenextract/extraction/prompt_builder.py`).
- Repair loop: robust JSON repair and code-fence removal implemented in `src/gegenextract/extraction/repair.py` and `src/gegenextract/extraction/parser.py`.
- Evaluator: hierarchical, field-level scorer and semantic checks in `src/gegenextract/scoring/evaluator.py` and `src/gegenextract/scoring/semantic.py`.
- Optimization engine: mutation operators, candidate tracking, selection and stagnation management in `src/gegenextract/optimization/*`.
- Persistence & observability: experiments persisted to `experiments/...` with SQLite DB and artifact folders; detailed LLM call logging to `llm_calls` enables raw-response audits.

## 4 Document Processing Pipeline

Workflow: PDF → text extraction → optional OCR fallback → normalized text representation used by the prompt builder.

- Primary text extraction uses PyMuPDF (`fitz`) which handled the selected resume PDFs in this run successfully.
- OCR fallback is implemented but optional (requires Poppler for `pdf2image`); OCR remains off in this run since PyMuPDF sufficed.

## 5 Extraction + Repair Pipeline

Core techniques:

- Schema-aware prompting: prompts encode a JSON schema and explicit field constraints to reduce hallucination and encourage stable structure.
- Groq integration: LLM calls use the Groq endpoint via an OpenAI-compatible chat completion wrapper in `src/gegenextract/extraction/groq_client.py`. The client sanitizes API keys and records raw calls into `llm_calls` for auditing.
- Malformed JSON handling: many LLM responses include markdown/code fences or trailing text. We first strip code fences, attempt direct JSON parsing, then invoke a constrained repair prompt that requests "ONLY the fixed JSON object" if parsing fails.
- Repair-aware extraction: the extraction engine automatically invokes the repair loop and returns repaired parsed JSON when schema validation is unavailable, improving downstream evaluation throughput.
- Formatting stabilization: standardized prompt templates, explicit constraints, and conservative low-temperature sampling reduce random formatting variations.

## 6 Hierarchical Evaluation Engine

The evaluator computes metrics at multiple granularities:

- Field-level metrics: per-field precision, recall and F1 are computed and stored per-evaluation (see `field_diagnostics.json`).
- Subtree aggregation: nested structures and repeated-array fields are aggregated as subtrees to allow more meaningful alignment (e.g., identity entries, employment-history items).
- Repeated-array alignment: array elements are matched by a lightweight alignment heuristic; current implementation favors ordering-agnostic matching but is intentionally conservative to avoid inflated scores from hallucinated repetitions.
- Semantic comparison abstraction: evaluators can compare normalized strings using semantic similarity heuristics (lightweight) and exact-match checks depending on the field.

## 7 Autonomous Optimization Engine

The optimizer combines mutation strategies with conservative search:

- Mutation strategies: `rewrite_instruction`, `strengthen_constraint`, `reduce_hallucination`, and targeted constraint tweaks. Mutations act on prompt templates and constraint blocks rather than changing model hyperparameters.
- Beam/evolution hybrid: each generation proposes a small beam of mutations; selection uses evaluator scores with duplicate suppression to keep diverse candidates.
- Failure-aware mutations: mutations that previously produced parsing or nested-structure failures are deprioritized; operators include conservative and aggressive variants adjustable per-run.
- Prompt lineage tracking: all prompt versions are tracked and stored to produce diffs (`prompt_diffs/seed_vs_best.html`) and enable rollbacks.
- Duplicate suppression and rollback: exact or near-duplicate candidates are suppressed, and stagnation detection triggers rollback to last known-best when improvement stalls.

## 8 Experiment Methodology

Design choices for defensible, reproducible experiments:

- Validation/test split isolation: the runner enforces a held-out test split and prevents test data from influencing optimization.
- Deterministic runs: random seeds, fixed beam sizes, and low-temperature sampling (temperature=0.1) produce reproducible artifacts.
- Conservative optimization: small population sizes, low mutation aggressiveness, and deterministic selection criteria prioritize reliability over rapid exploration.
- Evaluation methodology: per-field F1 aggregation forms the objective for selection; the system records per-sample and per-field diagnostics for deeper analysis.

## 9 Results & Findings

Optimization trajectory summary:

- Total generations: 5; per-generation mean scores (approx): [0.228, 0.225, 0.241, 0.25, 0.241]; max candidate score observed: 1.0.
- Stagnation was observed (4 stagnant generations vs 1 improving generation); initial good seed candidates were discovered early and persisted.

Per-field observations:

- Strongest fields: see `optimization_insights.md` (fields with highest mean F1 across generations).
- Weakest/most unstable fields: fields with low mean F1 or high variance across generations are enumerated in `optimization_insights.md`.

Mutation effectiveness:

- `reduce_hallucination` produced mixed results: many zero-scoring candidates with occasional high-scoring successes (mean ≈ 0.237).
- `rewrite_instruction` and `strengthen_constraint` were often chosen as the winning mutations for retained best artifacts.

Repair-loop effectiveness:

- The repair loop recovered many code-fenced or malformed JSON responses—these repaired outputs permitted valid evaluation where naive parsing would have failed. Examples are in [repair_examples.md](repair_examples.md) and raw-call extracts in [llm_response_examples.md](llm_response_examples.md).

Illustrative examples:

- Prompt diffs: view `prompt_diffs/seed_vs_best.html` for side-by-side changes that improved evaluation results.
- Malformed → repaired output examples: see `repair_examples.md` for concrete examples of markdown/code-fence cleanup and JSON fixes.

## 10 Engineering Challenges

- Groq endpoint migration: initial client used an incorrect endpoint; we refactored calls to `https://api.groq.com/openai/v1/chat/completions` and adapted to OpenAI-compatible responses.
- Malformed JSON outputs: frequent code-fence wrapping required robust pre-processing and a targeted repair prompt.
- OCR dependency handling: OCR fallback is available but requires Poppler; this environment used PyMuPDF for primary extraction.
- Experiment reproducibility: careful seeding and low-temperature settings are necessary to produce repeatable trajectories.

## 11 Limitations

- OCR dependencies: Poppler is required for the OCR fallback; OCR paths were not exercised heavily in this run.
- Dataset scale: this controlled experiment used a small subset (7 PDFs) — conclusions are preliminary and would benefit from larger-scale trials.
- Semantic backend: current similarity heuristics are lightweight; future work could integrate stronger semantic matchers for robust subtree alignment.

## 12 Future Improvements

- Multimodal extraction: enable richer image and layout-aware extraction models for complex documents.
- Stronger semantic evaluators: integrate embedding-based or supervised matchers for improved alignment and scoring.
- Larger-scale, distributed optimization: scale the optimizer across machines/external workers for broader search while preserving determinism via seeded shards.
- Adaptive mutation policies: dynamically adjust mutation aggressiveness based on observed repairs and failure modes.

## 13 Conclusion

GegenExtract demonstrates a research-oriented system that combines repair-aware extraction, hierarchical evaluation, and conservative autonomous prompt optimization. The design prioritizes reproducibility, observability, and incremental, accountable improvement—qualities required for production-grade structured extraction in high-stakes settings.

## 14 Appendix — Artifacts produced in this run

- Field diagnostics and per-field aggregates: [field_diagnostics.json](field_diagnostics.json)
- Field metrics table: [field_metrics_table.md](field_metrics_table.md)
- LLM response examples: [llm_response_examples.md](llm_response_examples.md)
- Repair examples: [repair_examples.md](repair_examples.md)
- Optimization insights: [optimization_insights.md](optimization_insights.md)
- Prompt diffs: `prompt_diffs/seed_vs_best.html`
- Plots: `score_over_generations.png`, `mutation_mean_scores.png`
- DB: experiments SQLite at `../experiments.db`

---

If you want, I can now (a) convert the field metrics into CSVs for publication tables, (b) expand the report with per-field narrative (examples and concrete per-field diffs), or (c) prepare a short slide deck summarizing the experiment. Which would you prefer next?