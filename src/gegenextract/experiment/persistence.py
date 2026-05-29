import sqlite3
import json
import threading
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import os


class PersistenceManager:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT,
                config TEXT,
                created_at TEXT
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY,
                experiment_id TEXT,
                version INTEGER,
                artifact TEXT,
                metadata TEXT
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                generation INTEGER,
                summary TEXT
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                generation INTEGER,
                artifact_id TEXT,
                score REAL,
                report TEXT
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_cache (
                prompt_hash TEXT PRIMARY KEY,
                score REAL,
                report TEXT
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY,
                model TEXT,
                prompt TEXT,
                response TEXT,
                tokens INTEGER,
                elapsed_seconds REAL,
                created_at TEXT
            )""")
            conn.commit()

    @contextmanager
    def _conn(self):
        # thread-safe connection per operation
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            try:
                yield conn
            finally:
                conn.close()

    def persist_experiment(self, experiment_id: str, name: str, config: Dict[str, Any]):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO experiments (id,name,config,created_at) VALUES (?,?,?,datetime('now'))",
                        (experiment_id, name, json.dumps(config)))
            conn.commit()

    def persist_prompt(self, experiment_id: str, artifact_id: str, version: int, artifact: Dict[str, Any], metadata: Dict[str, Any]):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO prompts (id,experiment_id,version,artifact,metadata) VALUES (?,?,?,?,?)",
                        (artifact_id, experiment_id, version, json.dumps(artifact), json.dumps(metadata)))
            conn.commit()

    def persist_generation(self, experiment_id: str, generation: int, summary: Dict[str, Any]):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO generations (experiment_id,generation,summary) VALUES (?,?,?)",
                        (experiment_id, generation, json.dumps(summary)))
            conn.commit()

    def persist_evaluation(self, experiment_id: str, generation: int, artifact_id: str, score: float, report: Dict[str, Any]):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO evaluations (experiment_id,generation,artifact_id,score,report) VALUES (?,?,?,?,?)",
                        (experiment_id, generation, artifact_id, score, json.dumps(report)))
            conn.commit()

    def get_cached_evaluation(self, prompt_hash: str):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT score, report FROM evaluation_cache WHERE prompt_hash=?", (prompt_hash,))
            row = cur.fetchone()
            if not row:
                return None
            return {"score": row[0], "report": json.loads(row[1])}

    def persist_cached_evaluation(self, prompt_hash: str, score: float, report: Dict[str, Any]):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO evaluation_cache (prompt_hash,score,report) VALUES (?,?,?)",
                        (prompt_hash, score, json.dumps(report)))
            conn.commit()

    def load_latest_generation(self, experiment_id: str) -> Optional[int]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT generation FROM generations WHERE experiment_id=? ORDER BY generation DESC LIMIT 1", (experiment_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def record_llm_call(self, record) -> None:
        """Persist an LLM call record. Accepts an object with attributes: id, model, prompt, response, tokens, elapsed_seconds, created_at"""
        with self._conn() as conn:
            cur = conn.cursor()
            try:
                created = getattr(record, 'created_at', None)
                created_str = created.isoformat() if hasattr(created, 'isoformat') else str(created)
            except Exception:
                created_str = ''
            cur.execute("INSERT OR REPLACE INTO llm_calls (id,model,prompt,response,tokens,elapsed_seconds,created_at) VALUES (?,?,?,?,?,?,?)",
                        (getattr(record, 'id', ''), getattr(record, 'model', ''), getattr(record, 'prompt', ''), getattr(record, 'response', ''), getattr(record, 'tokens', None), getattr(record, 'elapsed_seconds', None), created_str))
            conn.commit()
