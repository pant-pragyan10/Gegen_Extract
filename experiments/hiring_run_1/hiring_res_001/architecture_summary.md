# Architecture Summary

Pipeline: ingestion -> document processing -> extraction (Groq) -> repair -> evaluation -> optimizer

Core components and locations:

- **Ingestion / dataset**: loader and dataset management — [src/gegenextract/dataset/loader.py](src/gegenextract/dataset/loader.py)
- **Document processing**: PDF/text extraction and optional OCR — [src/gegenextract/document_processing/pdf_processor.py](src/gegenextract/document_processing/pdf_processor.py)
- **Prompt management / builder**: constructs extraction prompts and templates — [src/gegenextract/extraction/prompt_builder.py](src/gegenextract/extraction/prompt_builder.py)
- **Groq client (LLM calls)**: OpenAI-compatible chat completions wrapper for Groq — [src/gegenextract/extraction/groq_client.py](src/gegenextract/extraction/groq_client.py)
- **Extraction engine**: orchestrates prompt -> LLM -> parse -> validate -> repair loop — [src/gegenextract/extraction/engine.py](src/gegenextract/extraction/engine.py)
- **Repair & parsing**: robust JSON repair and code-fence stripping — [src/gegenextract/extraction/repair.py](src/gegenextract/extraction/repair.py) and [src/gegenextract/extraction/parser.py](src/gegenextract/extraction/parser.py)
- **Scoring & evaluation**: evaluator and semantic alignment checks — [src/gegenextract/scoring/evaluator.py](src/gegenextract/scoring/evaluator.py) and [src/gegenextract/scoring/semantic.py](src/gegenextract/scoring/semantic.py)
- **Optimization / mutations**: candidate representation, mutation operators, selection and tracking — [src/gegenextract/optimization/mutation.py](src/gegenextract/optimization/mutation.py), [src/gegenextract/optimization/fitness_adapter.py](src/gegenextract/optimization/fitness_adapter.py), [src/gegenextract/optimization/tracker.py](src/gegenextract/optimization/tracker.py)
- **Experiment orchestration & persistence**: deterministic runner, checkpoints, DB persistence, LLM call logging — [src/gegenextract/experiment/real_runner.py](src/gegenextract/experiment/real_runner.py) and [src/gegenextract/experiment/persistence.py](src/gegenextract/experiment/persistence.py)
- **Local file caching & artifacts**: experiment folder, file cache utils — [src/gegenextract/persistence/file_cache.py](src/gegenextract/persistence/file_cache.py)

Observability & artifacts produced in runs:

- LLM call logging table: `llm_calls` persisted by `experiment.persistence.record_llm_call()` — useful for audits and raw-response sampling.
- Analysis scripts: `scripts/generate_experiment_analysis.py` produces `mutation_analysis.json`, plots and `prompt_diffs/` for quick review.

Notes:

- Groq is used via an OpenAI-compatible chat completions endpoint; the client is hardened to sanitize API keys and parse OpenAI-style responses.
- The extraction engine includes a conservative repair loop that strips markdown code fences before attempting to repair JSON outputs; this is effective for recovering responses that include formatting markup.
- For OCR-dependent PDFs, install Poppler (`brew install poppler`) to enable `pdf2image` fallback in `pdf_processor`.


