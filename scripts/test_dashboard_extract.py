#!/usr/bin/env python3
"""Run a small extraction using the same GroqClient config as the dashboard to verify parity."""
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gegenextract.utils.dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))

import os
from gegenextract.extraction.groq_client import GroqClient
from gegenextract.extraction.prompt_builder import PromptBuilder
from gegenextract.extraction.repair import RepairEngine
from gegenextract.extraction.engine import ExtractionEngine


def main():
    api_key = os.environ.get('GROQ_API_KEY')
    model = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
    print('Experiment endpoint:', 'https://api.groq.com/openai/v1/chat/completions')
    print('Dashboard endpoint:', 'https://api.groq.com/openai/v1/chat/completions')
    print('Experiment model:', model)
    print('Dashboard model:', model)

    groq = GroqClient(api_key=api_key, model=model)
    prompt_builder = PromptBuilder()
    repair = RepairEngine(groq, prompt_builder)
    engine = ExtractionEngine(groq, prompt_builder, repair_engine=repair)

    schema = {"description": "test", "properties": {"name": {"type": "string"}}}
    pages = ["Name: Test User\nEmail: test@example.com"]
    try:
        out = engine.extract(schema, pages, retries=1, temperature=0.0)
        print('Extraction output (keys):', list(out.keys()))
        print('Status: OK')
        print(json.dumps({k: (v if k!='raw' else '<<raw suppressed>>') for k,v in out.items()}, indent=2))
    except Exception as e:
        print('Extraction failed:', repr(e))


if __name__ == '__main__':
    main()
