import sqlite3
from typing import Optional
from pathlib import Path
from ..schemas import LLMCallRecord


class PersistenceManager:
    """Simple SQLite-backed persistence manager for LLM calls and experiment state."""

    def __init__(self, db_url: str):
        # support sqlite:///./path
        if db_url.startswith("sqlite:///"):
            path = db_url.replace("sqlite:///", "")
        else:
            path = db_url
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY,
                model TEXT,
                prompt TEXT,
                response TEXT,
                tokens INTEGER,
                elapsed_seconds REAL,
                created_at TEXT
            )
            """
        )
        self.conn.commit()

        # additional tables for extraction logs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS extractions (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                raw_output TEXT,
                repaired_output TEXT,
                tokens INTEGER,
                elapsed_seconds REAL,
                created_at TEXT
            )
            """
        )
        self.conn.commit()

    def record_llm_call(self, record: LLMCallRecord) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO llm_calls (id, model, prompt, response, tokens, elapsed_seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.model,
                record.prompt,
                record.response,
                record.tokens,
                record.elapsed_seconds,
                record.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    def record_extraction(self, id: str, prompt: str, raw_output: str, repaired_output: str | None, tokens: int | None, elapsed_seconds: float | None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO extractions (id, prompt, raw_output, repaired_output, tokens, elapsed_seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (id, prompt, raw_output, repaired_output, tokens, elapsed_seconds),
        )
        self.conn.commit()
