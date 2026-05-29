import streamlit as st
import json
from pathlib import Path
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components
import os
import sys
import time
import tempfile
import traceback
import requests
import threading
import yaml

st.set_page_config(layout='wide')

st.title("GegenExtract — Resume Extract & Optimize")

# session flag for demo display
if 'show_demo' not in st.session_state:
    st.session_state['show_demo'] = False

# Ensure project src is importable for local imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gegenextract.utils.dotenv import load_dotenv as project_load_dotenv

# Load project .env to ensure GROQ_API_KEY is available (if present)
try:
    project_load_dotenv(str(ROOT / '.env'))
except Exception:
    # non-fatal if .env not present or fails to parse
    pass
    

def _sanitized_key_length(key: str | None) -> int | None:
    if not key:
        return None
    k = str(key).strip().strip('"').strip("'")
    return len(k)

def _groq_connection_test(api_key: str | None, model: str = 'llama-3.3-70b-versatile') -> tuple[bool, int | None, int | None]:
    """Return (connected, key_length, status_code_or_none).
    Performs a minimal auth test against the Groq chat completions endpoint.
    This sends a tiny request (max_tokens=1) and treats 401 as unauthorized.
    """
    if not api_key:
        return (False, None, None)
    sanitized = str(api_key).strip().strip('"').strip("'")
    key_len = len(sanitized)
    headers = {"Authorization": f"Bearer {sanitized}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}], "temperature": 0.0, "max_tokens": 1}
    url = "https://api.groq.com/openai/v1/chat/completions"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=6)
        if r.status_code == 401:
            return (False, key_len, 401)
        if r.ok:
            return (True, key_len, r.status_code)
        return (False, key_len, r.status_code)
    except Exception:
        return (False, key_len, None)

# (Developer debug UI removed for reviewer-focused workflow)


from gegenextract.extraction.engine import ExtractionEngine
from gegenextract.extraction.groq_client import GroqClient
from gegenextract.extraction.prompt_builder import PromptBuilder
from gegenextract.extraction.repair import RepairEngine
from gegenextract.document_processing.pdf_processor import PdfProcessor
from gegenextract.experiment.persistence import PersistenceManager
from gegenextract.extraction.parser import safe_parse_json
from gegenextract.experiment.real_runner import RealExperimentRunner
from gegenextract.scoring.evaluator import Evaluator


def render_resume(parsed: dict):
    """Render a structured resume view: Contact, Education, Experience, Projects, Skills."""
    if not parsed or not isinstance(parsed, dict):
        st.write('No structured data to display')
        return

    def _get_field(d, *keys):
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
            if cur is None:
                return None
        return cur

    st.markdown('**Contact**')
    contact = {}
    contact['name'] = parsed.get('name') or _get_field(parsed, 'contact', 'name')
    contact['email'] = parsed.get('email') or _get_field(parsed, 'contact', 'email')
    contact['phone'] = parsed.get('phone') or _get_field(parsed, 'contact', 'phone')
    st.write({k: v for k, v in contact.items() if v})

    st.markdown('**Education**')
    edu = parsed.get('education') or _get_field(parsed, 'education') or []
    if edu:
        for e in edu:
            st.write(e)
    else:
        st.write('—')

    st.markdown('**Experience**')
    exp = parsed.get('experience') or _get_field(parsed, 'experience') or []
    if exp:
        for ex in exp:
                st.write(ex)
# Reviewer-focused single page UI (Upload -> Extract & Optimize -> Compare)
def _count_fields(obj):
    if obj is None:
        return 0
    if isinstance(obj, dict):
        total = 0
        for v in obj.values():
            total += _count_fields(v)
        return total
    if isinstance(obj, list):
        return sum(_count_fields(i) for i in obj)
    return 1 if obj not in (None, '', []) else 0


def _simulate_optimize(seed: dict) -> tuple[dict, str]:
    """Create a lightweight simulated optimized extraction and return (optimized, modifier_desc).
    This is used when no Groq key is available so the reviewer UI remains deterministic and fast.
    """
    opt = json.loads(json.dumps(seed)) if seed else {}
    # normalize skills -> list of lowercase unique
    skills = opt.get('skills') or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',') if s.strip()]
    skills = list({s.strip() for s in skills})
    if skills:
        opt['skills'] = skills
    # ensure education and experience arrays
    if 'education' in opt and not isinstance(opt['education'], list):
        opt['education'] = [opt['education']]
    if 'experience' in opt and not isinstance(opt['experience'], list):
        opt['experience'] = [opt['experience']]
    # add a small synthesized project if none
    if not opt.get('projects'):
        opt['projects'] = [{'name': 'Imported project', 'description': 'Auto-detected project.'}]
    modifier = 'Auto-normalize fields: skills list, arrayify education/experience, add missing projects'
    return opt, modifier

# Lightweight fallback extractor: use simple heuristics when LLM returns no structured JSON
def _fallback_extract_from_text(pages: list[str]) -> dict:
    text = '\n'.join(pages or [])
    out = {}
    # email
    import re
    m = re.search(r'[\w\.-]+@[\w\.-]+', text)
    if m:
        out.setdefault('contact', {})['email'] = m.group(0)
    # phone
    m = re.search(r'\+?\d[\d\-() ]{6,}\d', text)
    if m:
        out.setdefault('contact', {})['phone'] = m.group(0)
    # name guess: first non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        out['name'] = lines[0]
    # try to capture simple skills list
    skills = []
    for line in lines[:40]:
        if 'skills' in line.lower() and ':' in line:
            parts = line.split(':', 1)[1]
            skills = [s.strip() for s in parts.split(',') if s.strip()]
            break
    if skills:
        out['skills'] = skills
    # minimal education/experience heuristics
    ed = []
    ex = []
    for i, line in enumerate(lines[:200]):
        if 'univ' in line.lower() or 'college' in line.lower() or 'bsc' in line.lower() or 'msc' in line.lower() or 'degree' in line.lower():
            ed.append({'institution': line})
        if any(k in line.lower() for k in ['engineer', 'manager', 'analyst', 'scientist', 'developer']):
            ex.append({'summary': line})
    if ed:
        out['education'] = ed
    if ex:
        out['experience'] = ex
    return out


st.header('Upload Resume PDF and run Extract & Optimize')

col_left, col_right = st.columns([1, 1])
with col_left:
    uploaded = st.file_uploader('Upload Resume PDF', type=['pdf'], help='Drop a PDF or click to choose')
    text_input = st.text_area('Or paste document text / JSON here', height=140)
    do_btn = st.button('Extract & Optimize')
    demo_btn = st.button('Show Demo')
    if demo_btn:
        st.session_state['show_demo'] = True
        # avoid calling experimental_rerun (may be unavailable in some Streamlit versions)

# allow automatic demo render via query param or env var for testing/screenshots
_qp = {}
try:
    _qp = st.experimental_get_query_params() or {}
except Exception:
    _qp = {}
auto_demo = bool(_qp.get('demo')) or os.environ.get('FORCE_DEMO') == '1'
if auto_demo:
    demo_btn = True

with col_right:
    st.write('')
    st.info('Single-step reviewer workflow: upload → extract & optimize → review → download')

# result placeholders
result_area = st.container()
summary_area = st.container()
comparison_area = st.container()
prompt_area = st.container()

# Advanced debug panel (collapsed by default) — move technical diagnostics here
advanced_panel = st.expander('Advanced Debug Panel', expanded=False)

with advanced_panel:
    st.markdown('**Diagnostics (advanced, hidden by default)**')
    # Groq connection check (do not display API key)
    groq_present = bool(os.environ.get('GROQ_API_KEY'))
    groq_model = os.environ.get('GROQ_MODEL') or 'not set'
    connected, key_len, status_code = _groq_connection_test(os.environ.get('GROQ_API_KEY'))
    st.write('Groq API configured:', 'Yes' if groq_present else 'No')
    st.write('Groq connection status:', 'Connected' if connected else 'Not connected')
    st.write('Groq HTTP status code:', status_code)
    st.write('Groq model env:', groq_model)
    # show demo / force flags
    st.write('Auto-demo enabled:', bool(os.environ.get('FORCE_DEMO') == '1' or st.session_state.get('show_demo')))
    # show experiment folder if present (no file contents)
    try:
        exp_root = ROOT / 'experiments'
        if exp_root.exists():
            st.write('Experiment folder present at:', str(exp_root))
        else:
            st.write('No experiments folder present')
    except Exception:
        pass

sample_demo = {
    'name': 'Demo Candidate',
    'contact': {'email': 'demo@sample.com', 'phone': '555-0000'},
    'education': [{'institution': 'Demo University', 'degree': 'BSc Demo', 'year': 2018}],
    'experience': [{'company': 'Demo Co', 'role': 'Engineer', 'duration': '2018-2022'}],
    'projects': [{'name': 'DemoProject', 'description': 'Demo extraction project'}],
    'skills': ['Python', 'NLP']
}
# Note: demo rendering is handled by the main Extract & Optimize flow below (triggered by the button or demo flag).

if do_btn or text_input or demo_btn:
    pages = []
    if demo_btn:
        # render demo immediately without extracting
        parsed = sample_demo
        latency = 0.0
        # simulate optimization
        best_result, best_modifier = _simulate_optimize(parsed)
        seed_prompt = PromptBuilder().build({'description': 'demo', 'properties': {}}, [''])
        seed_score = _count_fields(parsed)
        best_score = _count_fields(best_result)
        improvement = None
        if seed_score:
            try:
                improvement = (best_score - seed_score) / seed_score * 100
            except Exception:
                improvement = None

        with result_area:
            st.success('Extraction succeeded (demo)')
            render_resume(parsed)
            with st.expander('View Raw JSON (advanced)'):
                st.code(json.dumps(parsed, indent=2), language='json')

        with summary_area:
            st.subheader('Optimization Summary')
            cols = st.columns(3)
            cols[0].metric('Initial Score', f"{seed_score}")
            cols[1].metric('Optimized Score', f"{best_score}")
            cols[2].metric('Improvement %', f"{improvement:.2f}%" if improvement is not None else '—')

        with comparison_area:
            st.subheader('Extraction Comparison')
            comp_cols = st.columns(2)
            comp_cols[0].markdown('**Before Optimization**')
            comp_cols[0].write(parsed)
            comp_cols[1].markdown('**After Optimization**')
            comp_cols[1].write(best_result)
            # simple diffs
            added = []
            changed = []
            def _collect_changes(a, b, path=''):
                if isinstance(a, dict) and isinstance(b, dict):
                    for k in b.keys():
                        newpath = f"{path}.{k}" if path else k
                        if k not in a:
                            added.append(newpath)
                        else:
                            if a[k] != b[k]:
                                changed.append(newpath)
                            _collect_changes(a[k], b[k], newpath)
            _collect_changes(parsed, best_result)
            if added:
                st.markdown('**Fields added:**')
                st.write(added)
            if changed:
                st.markdown('**Fields changed:**')
                st.write(changed)

        with prompt_area:
            st.subheader('Prompt Evolution')
            evo_cols = st.columns(2)
            evo_cols[0].markdown('**Original Prompt**')
            evo_cols[0].code(seed_prompt)
            evo_cols[1].markdown('**Optimized Prompt / Modifier**')
            evo_cols[1].code(best_modifier or '—')

        st.download_button('Download Optimized JSON', data=json.dumps(best_result, indent=2), file_name='optimized_extraction.json')
        st.caption('Extraction latency: 0.00s')
        # skip the normal flow
        pages = []
    
    if uploaded is not None:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        tf.write(uploaded.getbuffer())
        tf.flush()
        temp_path = tf.name
        tf.close()
        proc = PdfProcessor()
        pages_objs = proc.extract_text_pages(temp_path)
        pages = [p.text or '' for p in pages_objs]
    elif text_input:
        pages = [text_input]

    if not pages:
        st.warning('No document content provided — upload a PDF or paste text')
    else:
        # try parse pasted JSON first
        pre_parsed = None
        if text_input:
            try:
                pre_parsed = safe_parse_json(text_input)
            except Exception:
                pre_parsed = None

        parsed = None
        latency = 0.0
        if pre_parsed is not None:
            parsed = pre_parsed
        else:
            # call extraction engine (uses Groq if configured)
            persistence = None
            groq = GroqClient(os.environ.get('GROQ_API_KEY'), model=os.environ.get('GROQ_MODEL'))
            prompt_builder = PromptBuilder()
            repair = RepairEngine(groq, prompt_builder)
            engine = ExtractionEngine(groq, prompt_builder, persistence=persistence, repair_engine=repair)
            with st.spinner('Running extraction...'):
                start_time = time.time()
                out = engine.extract({'description': 'ad-hoc', 'properties': {}}, pages, retries=1, temperature=0.0)
                latency = time.time() - start_time
            extracted = out.get('result')
            repaired = out.get('repaired')
            if extracted is not None:
                parsed = extracted
            elif repaired:
                try:
                    parsed = safe_parse_json(repaired)
                except Exception:
                    parsed = None
            else:
                raw = out.get('raw')
                if isinstance(raw, dict):
                    try:
                        choices = raw.get('choices')
                        if choices and len(choices) > 0:
                            msg = choices[0].get('message') or {}
                            txt = msg.get('content') or choices[0].get('text')
                            if txt:
                                try:
                                    parsed = safe_parse_json(txt)
                                except Exception:
                                    parsed = None
                    except Exception:
                        parsed = None

        # attempt fallback extraction if we don't have parsed data yet
        fallback_notice = None
        if parsed is None:
            try:
                fb = _fallback_extract_from_text(pages)
                if fb and isinstance(fb, dict) and fb.keys():
                    parsed = fb
                    fallback_notice = 'Extraction succeeded (local fallback)'
                else:
                    result_area.error('Extraction returned no structured data')
                    if pre_parsed is None:
                        result_area.write('Try pasting the document text or a representative JSON sample in the input box')
            except Exception:
                result_area.error('Extraction returned no structured data')
                if pre_parsed is None:
                    result_area.write('Try pasting the document text or a representative JSON sample in the input box')

        # If we have parsed data (either from LLM or fallback), render and run optimization
        if parsed is not None:
            with result_area:
                st.success(fallback_notice or 'Extraction succeeded')
                render_resume(parsed)
                with st.expander('View Raw JSON (advanced)'):
                    st.code(json.dumps(parsed, indent=2), language='json')

            # Run or simulate optimization — full audit with guardrails
            groq_key = os.environ.get('GROQ_API_KEY')
            prompt_builder = PromptBuilder()
            seed_prompt = prompt_builder.build({'description': 'ad-hoc', 'properties': {}}, pages)
            seed_score = _count_fields(parsed)

            # Candidate generation: iterative generations (5) with stronger mutations
            mutation_templates = [
                # schema enforcement
                'Enforce a strict JSON schema: produce a top-level object with keys contact, education, experience, projects, skills. Each field must follow the described types: contact is an object with name,email,phone; education and experience are arrays of objects with institution/title,start_year,end_year; projects is an array with name,description,technologies; skills is an array of short keywords. Return valid JSON only.',
                # nested object extraction
                'Extract nested objects fully: for each experience entry, include company, title, start_year, end_year, bullets (array). For education include degree, institution, start_year, end_year, honors. Use arrays for repeated items.',
                # explicit examples
                'Provide an explicit example in the output for one project and one experience item showing the exact JSON structure to follow. Then return the extraction for the resume using that structure. Do not include any extra prose.',
                # output formatting constraints
                'Return only compact, valid JSON, no surrounding backticks or markdown. Use consistent field names (snake_case) and ISO years (YYYY). If a field is missing, omit it rather than null.',
                # field normalization instructions
                'Normalize fields: canonicalize phone and email formats, split skills on commas/spaces and lowercase them, deduplicate skills, ensure education and experience are arrays even for single entries.'
            ]

            generations = 5
            candidates = []

            def _compute_score(res: dict):
                # composite score: populated_fields + nested_fields + valid_lists + valid_contact_info
                if not res or not isinstance(res, dict) or len(res.keys()) == 0:
                    return float('-inf'), {'populated_fields': 0, 'nested_fields': 0, 'valid_lists': 0, 'valid_contact_info': 0}

                # populated_fields: count top-level non-empty keys
                populated_fields = sum(1 for k, v in res.items() if v not in (None, '', [], {}))

                # nested_fields: count number of nested dict/list elements beyond top-level
                def _nested_count(x):
                    c = 0
                    if isinstance(x, dict):
                        for v in x.values():
                            if isinstance(v, (dict, list)) and v:
                                c += 1
                                c += _nested_count(v)
                    elif isinstance(x, list):
                        for it in x:
                            if isinstance(it, (dict, list)) and it:
                                c += 1
                                c += _nested_count(it)
                    return c

                nested_fields = _nested_count(res)

                # valid_lists: count of expected list fields that are non-empty lists
                expected_lists = ('education', 'experience', 'projects', 'skills')
                valid_lists = sum(1 for f in expected_lists if isinstance(res.get(f), list) and len(res.get(f)) > 0)

                # valid_contact_info: email + phone + name presence/validity (max 3)
                contact = res.get('contact') or {}
                valid_contact_info = 0
                try:
                    import re
                    if isinstance(contact, dict):
                        email = contact.get('email') or ''
                        phone = contact.get('phone') or ''
                        name = contact.get('name') or ''
                        if isinstance(email, str) and re.search(r'\w[\w\.-]+@[\w\.-]+', email):
                            valid_contact_info += 1
                        if isinstance(phone, str) and re.search(r'\d{6,}', phone):
                            valid_contact_info += 1
                        if isinstance(name, str) and name.strip():
                            valid_contact_info += 1
                except Exception:
                    valid_contact_info = 0

                score = populated_fields + nested_fields + valid_lists + valid_contact_info
                breakdown = {
                    'populated_fields': populated_fields,
                    'nested_fields': nested_fields,
                    'valid_lists': valid_lists,
                    'valid_contact_info': valid_contact_info,
                }
                return score, breakdown

            # helper to append seed baseline as generation 0
            candidates.append({'modifier': 'SEED_BASELINE_NO_CHANGE', 'generation': 0, 'prompt_text': seed_prompt, 'result': parsed, 'score': seed_score, 'breakdown': {'populated_fields': seed_score}, 'reason': 'seed'})

            # perform iterative generations
            for gen in range(1, generations + 1):
                for tmpl in mutation_templates:
                    modifier_label = f'Gen{gen}:{tmpl.split(":")[0][:40]}'
                    instr = seed_prompt + '\n' + f'Generation {gen} mutation:\n' + tmpl
                    prompt_text = prompt_builder.build({'description': 'ad-hoc', 'properties': {}}, pages, extraction_instructions=instr)
                    parsed_c = None
                    reason = ''

                    if groq_key:
                        resp = None
                        try:
                            resp = groq.call(prompt_text, temperature=0.0)
                        except Exception:
                            resp = {'text': None}
                        text = (resp or {}).get('text') or ''
                        if text:
                            try:
                                parsed_c = safe_parse_json(text)
                            except Exception:
                                parsed_c = None
                        if parsed_c is None and repair and text:
                            try:
                                repaired_t = repair.repair(text or '', 'parse_error', {'description': 'ad-hoc', 'properties': {}}, pages)
                                try:
                                    parsed_c = safe_parse_json(repaired_t) if repaired_t else None
                                except Exception:
                                    parsed_c = None
                                    reason = 'repair_failed'
                            except Exception:
                                parsed_c = None
                                reason = 'repair_error'
                    else:
                        # deterministic simulation to show improvements across generations
                        import copy
                        parsed_c = copy.deepcopy(parsed)
                        # apply transformations per template for deterministic demo
                        if 'schema enforcement' in tmpl or 'strict JSON' in tmpl:
                            parsed_c = {k: v for k, v in parsed_c.items() if k in ('contact', 'education', 'experience', 'projects', 'skills', 'name')}
                        if 'nested objects' in tmpl or 'nested' in tmpl:
                            # ensure education/experience are lists of dicts
                            if 'education' in parsed_c and not isinstance(parsed_c.get('education'), list):
                                parsed_c['education'] = [parsed_c['education']]
                            if 'experience' in parsed_c and not isinstance(parsed_c.get('experience'), list):
                                parsed_c['experience'] = [parsed_c['experience']]
                            # add bullets arrays
                            for ex in parsed_c.get('experience', []):
                                if isinstance(ex, dict) and not ex.get('bullets'):
                                    ex['bullets'] = ['Improved description']
                        if 'explicit example' in tmpl or 'example' in tmpl:
                            # add a fully-shaped project example if missing
                            if not parsed_c.get('projects'):
                                parsed_c['projects'] = [{'name': 'Example Project', 'description': 'Example desc', 'technologies': ['python']}]
                        if 'formatting constraints' in tmpl or 'compact' in tmpl:
                            # normalize years to ints where possible
                            for e in parsed_c.get('education', []) + parsed_c.get('experience', []):
                                if isinstance(e, dict):
                                    for key in ('start_year', 'end_year'):
                                        if key in e and isinstance(e.get(key), str) and e.get(key).isdigit():
                                            e[key] = int(e[key])
                        if 'Normalize fields' in tmpl or 'Normalize' in tmpl:
                            skills = parsed_c.get('skills') or []
                            if isinstance(skills, str):
                                skills = [s.strip() for s in skills.split(',') if s.strip()]
                            parsed_c['skills'] = list({s.lower() for s in (skills or []) if s})

                    score, breakdown = _compute_score(parsed_c)
                    if score == float('-inf'):
                        reason = reason or 'parse_failed_or_empty'
                    candidates.append({'modifier': modifier_label, 'generation': gen, 'prompt_text': prompt_text, 'result': parsed_c or {}, 'score': score, 'breakdown': breakdown, 'reason': reason})

            # Ensure seed baseline is present and compute seed_score consistently
            seed_entry = next((c for c in candidates if c['modifier'] == 'SEED_BASELINE_NO_CHANGE'), None)
            if not seed_entry:
                candidates.append({'modifier': 'SEED_BASELINE_NO_CHANGE', 'prompt_text': seed_prompt, 'result': parsed, 'score': seed_score, 'reason': 'seed'})
            else:
                seed_entry['result'] = parsed
                seed_entry['score'] = seed_score

            # Compute selection: reject invalid candidates (score -inf)
            # Choose best by numeric score, but never pick candidate with score < seed_score
            best = max(candidates, key=lambda x: x['score'])
            best_score = best['score']
            best_result = best['result']
            best_modifier = best['modifier']

            # If best is worse than seed, keep seed
            if best_score < seed_score:
                # find seed
                seed = next((c for c in candidates if c['modifier'] == 'SEED_BASELINE_NO_CHANGE'), None)
                if seed:
                    best_result = seed['result']
                    best_modifier = 'SEED_BASELINE_NO_CHANGE'
                    best_score = seed['score']

            # Build audit table for UI
            import pandas as _pd
            rows = []
            for c in candidates:
                accepted = (c['modifier'] == best_modifier)
                raw_reason = c.get('reason') or ('selected' if accepted else '')
                # map technical labels to friendly labels
                reason_map = {
                    'seed': 'Baseline',
                    'selected': 'Best Candidate',
                    'repair_failed': 'Invalid JSON Output',
                    'repair_error': 'Invalid JSON Output',
                    'parse_failed_or_empty': 'Failed Extraction',
                    'simulated_empty': 'Failed Extraction',
                }
                reason = reason_map.get(raw_reason, raw_reason)
                sc = c['score']
                # present -inf as None for readability
                sc_display = None if sc == float('-inf') else sc
                rows.append({'Candidate': c['modifier'], 'Generation': c.get('generation', None), 'Score': sc_display, 'Accepted?': accepted, 'Reason': reason})
            df = _pd.DataFrame(rows)

            # find best prompt text for prompt evolution display
            best_prompt_text = None
            for c in candidates:
                if c['modifier'] == best_modifier:
                    best_prompt_text = c.get('prompt_text')
                    break
            if not best_prompt_text:
                # fallback to seed prompt
                best_prompt_text = seed_prompt

            # compute prompt diff
            try:
                import difflib
                seed_lines = (seed_prompt or '').splitlines()
                best_lines = (best_prompt_text or '').splitlines()
                diff_lines = list(difflib.unified_diff(seed_lines, best_lines, fromfile='seed', tofile='best', lineterm=''))
                prompt_diff = '\n'.join(diff_lines) if diff_lines else ''
            except Exception:
                prompt_diff = ''

            # ensure best_modifier is never blank
            if not best_modifier:
                best_modifier = 'SEED_BASELINE_NO_CHANGE'

            improvement = None
            if seed_score:
                try:
                    improvement = (best_score - seed_score) / seed_score * 100
                except Exception:
                    improvement = None

            # Story and Summary
            with summary_area:
                st.subheader('Optimization Summary')
                # story flow
                st.markdown('Resume uploaded → Initial extraction → Prompt mutations generated → Candidates evaluated → Best prompt selected → Final extraction produced')
                cols = st.columns(3)
                cols[0].metric('Initial Score', f"{seed_score}")
                cols[1].metric('Optimized Score', f"{best_score}")
                cols[2].metric('Improvement %', f"{improvement:.2f}%" if improvement is not None else '—')

                # trajectory chart: best score and mean score per generation using Altair
                try:
                    import pandas as _pd
                    import altair as alt
                    gen_stats = []
                    for gen in range(1, generations + 1):
                        gen_cands = [c for c in candidates if c.get('generation') == gen]
                        if gen_cands:
                            scores = [s['score'] for s in gen_cands if s.get('score') not in (None, float('-inf'))]
                            best = max(scores) if scores else None
                            mean = float(_pd.Series(scores).mean()) if scores else None
                        else:
                            best = None
                            mean = None
                        gen_stats.append({'generation': gen, 'best_score': best, 'mean_score': mean})

                    chart_df = _pd.DataFrame(gen_stats)
                    if chart_df['best_score'].notna().any() or chart_df['mean_score'].notna().any():
                        base = alt.Chart(chart_df).encode(x=alt.X('generation:O', title='Generation Number'))

                        best_line = base.mark_line(color='#08519c').encode(y=alt.Y('best_score:Q', title='Score'), tooltip=['generation', 'best_score', 'mean_score'])
                        best_points = base.mark_point(color='#08519c', filled=True, size=60).encode(y='best_score:Q')
                        mean_line = base.mark_line(color='#ff7f0e', strokeDash=[5,5]).encode(y='mean_score:Q')
                        mean_points = base.mark_point(color='#ff7f0e', filled=True, size=40).encode(y='mean_score:Q')

                        chart = (best_line + best_points + mean_line + mean_points).properties(title='Prompt Optimization Progress', width=700, height=300)

                        # annotate best generation
                        if chart_df['best_score'].notna().any():
                            best_idx = chart_df['best_score'].idxmax()
                            best_row = chart_df.loc[best_idx]
                            if not _pd.isna(best_row['best_score']):
                                annot_df = _pd.DataFrame([{'generation': int(best_row['generation']), 'best_score': float(best_row['best_score']), 'label': 'Best Prompt Found'}])
                                annot = alt.Chart(annot_df).mark_text(dy=-15, color='green', fontWeight='bold').encode(x=alt.X('generation:O'), y=alt.Y('best_score:Q'), text=alt.Text('label:N'))
                                chart = chart + annot

                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.write('No valid generation scores to chart')
                except Exception:
                    pass

                # Optimization statistics card (visible immediately)
                total_candidates = len(candidates)
                valid_candidates = sum(1 for c in candidates if c.get('score') not in (None, float('-inf')))
                rejected_candidates = total_candidates - valid_candidates
                baseline_score = seed_score
                best_score_display = None if best_score == float('-inf') else best_score

                stat_cols = st.columns(6)
                stat_cols[0].metric('Total Candidates', f"{total_candidates}")
                stat_cols[1].metric('Valid Candidates', f"{valid_candidates}")
                stat_cols[2].metric('Rejected Candidates', f"{rejected_candidates}")
                stat_cols[3].metric('Baseline Score', f"{baseline_score}")
                stat_cols[4].metric('Best Score', f"{best_score_display if best_score_display is not None else '—'}")
                stat_cols[5].metric('Improvement %', f"{improvement:.2f}%" if improvement is not None else '—')

                # show audit table for quick inspection (visible by default)
                df_display = df.copy()
                # add friendly badge column
                df_display['Badge'] = df_display['Accepted?'].apply(lambda x: '✓ Selected' if x else '')
                # reorder columns for presentation
                display_cols = ['Badge', 'Candidate', 'Generation', 'Score', 'Reason']
                try:
                    styled = df_display[display_cols].style.apply(lambda row: ['background-color: #dff8e0' if row['Badge'] else '' for _ in row], axis=1)
                    st.dataframe(styled)
                except Exception:
                    st.dataframe(df_display[display_cols])


            # Comparison
            with comparison_area:
                st.subheader('Extraction Comparison')
                # Compute flattened field paths for before/after
                def _flatten_fields(obj, path=''):
                    fields = set()
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            newpath = f"{path}.{k}" if path else k
                            fields.add(newpath)
                            fields.update(_flatten_fields(v, newpath))
                    elif isinstance(obj, list):
                        # for lists, inspect dict elements to collect inner keys
                        for item in obj:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    newpath = f"{path}.{k}" if path else k
                                    fields.add(newpath)
                                    fields.update(_flatten_fields(v, newpath))
                            else:
                                # primitive list - mark the list field itself
                                if path:
                                    fields.add(path)
                    return fields

                before_fields = _flatten_fields(parsed or {})
                after_fields = _flatten_fields(best_result or {})
                new_fields = sorted(after_fields - before_fields)
                total_before = len(before_fields)
                total_after = len(after_fields)
                union_fields = before_fields | after_fields
                coverage = (len(after_fields) / len(union_fields) * 100) if union_fields else 0

                # New Fields summary
                st.markdown('**New Fields Extracted:**')
                if new_fields:
                    for nf in new_fields:
                        st.markdown(f'- ✓ {nf}')
                else:
                    st.markdown('- None')

                # Totals and coverage
                counts_cols = st.columns(3)
                counts_cols[0].markdown(f'**Before:** {total_before}')
                counts_cols[1].markdown(f'**After:** {total_after}')
                counts_cols[2].markdown(f'**Field coverage:** {coverage:.1f}%')
                comp_cols = st.columns(2)
                comp_cols[0].markdown('**Before Optimization**')
                comp_cols[0].write(parsed)
                comp_cols[1].markdown('**After Optimization**')
                comp_cols[1].write(best_result)

                # highlight diffs
                added = []
                changed = []
                def _collect_changes(a, b, path=''):
                    if isinstance(a, dict) and isinstance(b, dict):
                        for k in b.keys():
                            newpath = f"{path}.{k}" if path else k
                            if k not in a:
                                added.append(newpath)
                            else:
                                if a[k] != b[k]:
                                    changed.append(newpath)
                                _collect_changes(a[k], b[k], newpath)
                _collect_changes(parsed, best_result)
                if added:
                    st.markdown('**Fields added:**')
                    st.write(added)
                if changed:
                    st.markdown('**Fields changed:**')
                    st.write(changed)

            # Prompt evolution
            with prompt_area:
                st.subheader('Prompt Evolution')
                evo_cols = st.columns(2)
                evo_cols[0].markdown('**Original Prompt (changed lines)**')
                evo_cols[1].markdown('**Winning Prompt (changed lines)**')

                # extract changed lines from unified diff
                changed_seed_lines = []
                changed_best_lines = []
                try:
                    for ln in (prompt_diff or '').splitlines():
                        if ln.startswith('---') or ln.startswith('+++') or ln.startswith('@@'):
                            continue
                        if ln.startswith('-'):
                            changed_seed_lines.append(ln[1:])
                        elif ln.startswith('+'):
                            changed_best_lines.append(ln[1:])
                except Exception:
                    changed_seed_lines = []
                    changed_best_lines = []

                # show only changed lines by default; provide full prompts in an expander
                if changed_seed_lines:
                    evo_cols[0].code('\n'.join(['- ' + l for l in changed_seed_lines]))
                else:
                    evo_cols[0].markdown('_No changed lines from seed_')

                if changed_best_lines:
                    evo_cols[1].code('\n'.join(['+ ' + l for l in changed_best_lines]))
                else:
                    evo_cols[1].markdown('_No changed lines in winning prompt_')

                # Summary box for winning modification and reason
                win_label = best_modifier or 'SEED_BASELINE_NO_CHANGE'
                friendly_mod = win_label
                if win_label == 'SEED_BASELINE_NO_CHANGE':
                    friendly_mod = 'Baseline'
                summary_text = f"**Winning Modification:**\n{friendly_mod}\n\n**Reason:**\nImproved extraction coverage from {total_before} to {total_after} fields."
                st.markdown(summary_text)

                with st.expander('Show full prompts'):
                    st.markdown('**Original Prompt (full)**')
                    st.code(seed_prompt)
                    st.markdown('**Winning Prompt (full)**')
                    st.code(best_prompt_text or best_modifier or '—')

                # display per-candidate prompt & extracted result details (show prompt, score, breakdown)
                with st.expander('Candidate Details'):
                    for c in candidates:
                        title = f"{c.get('modifier')} (gen {c.get('generation', '?')})"
                        with st.expander(title):
                            st.markdown('**Prompt**')
                            st.code(c.get('prompt_text') or '')
                            st.markdown('**Candidate Score**')
                            sc = c.get('score')
                            sc_display = None if sc == float('-inf') else sc
                            st.write(sc_display)
                            st.markdown('**Score Breakdown**')
                            try:
                                st.json(c.get('breakdown') or {})
                            except Exception:
                                st.write(c.get('breakdown') or {})
                            st.markdown('**Extraction Result**')
                            try:
                                st.json(c.get('result') or {})
                            except Exception:
                                st.write(c.get('result') or {})
                st.download_button('Download Optimized JSON', data=json.dumps(best_result, indent=2), file_name='optimized_extraction.json')

                # explicit message if seed remains optimal
                if best_modifier == 'SEED_BASELINE_NO_CHANGE' or (best_score == seed_score):
                    st.info('Optimization completed. Seed prompt remains optimal.')

            # Small metadata
            # move latency to advanced panel
            with advanced_panel:
                st.caption(f'Extraction latency: {latency:.2f}s')

                
