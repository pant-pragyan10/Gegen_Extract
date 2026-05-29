#!/usr/bin/env python3
"""Run the existing extract_runner with project .env loaded so it uses the same key as dashboard."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gegenextract.utils.dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))

from scripts.extract_runner import main


if __name__ == '__main__':
    main()
