# Release Checklist

## What Works
- Streamlit dashboard (single-page reviewer flow)
- Extraction engine with Groq/Gemini pluggable client
- Repair engine and prompt-optimizer demo mode
- Persistence via SQLite and LLM call logging
- Experiment artifacts and figures generation scripts

## Known Limitations
- Live Gemini integration depends on correct credentials and endpoint access (may require Google Cloud project and API key or OAuth).
- TLS verification may fail in environments with intercepting proxies; `GEMINI_VERIFY_SSL=false` was used during local debugging but must be true in production.
- Some Groq responses may be rate-limited (HTTP 429) or return malformed JSON; repair heuristics exist but are not perfect.
- No automated history purge for previously committed secrets — consider using `git-filter-repo` or BFG to remove secrets from history.

## Demo Workflow (Reviewer)
1. Copy `.env.example` to `.env` and populate `GEMINI_API_KEY` or `GROQ_API_KEY` as needed.
2. Create a python venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Launch the Streamlit dashboard:

```bash
streamlit run scripts/launch_dashboard.py --server.port 8503
```

4. Upload a sample resume PDF and run "Extract & Optimize".
5. Review the Extraction Comparison, Optimization Summary, and Prompt Evolution panels.
6. Download the optimized JSON from the sidebar.

## Future Improvements
- Add automated secret scanning and replace with placeholders for detection in CI.
- Add `pre-commit` hooks: `detect-secrets`, `black`, `ruff`.
- Add CI pipeline for linting, tests, and basic smoke tests of the dashboard routes.
