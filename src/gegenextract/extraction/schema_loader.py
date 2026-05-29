import json
from pathlib import Path
from typing import Dict, Any


class SchemaLoader:
    """Load JSON schemas dynamically for ExtractBench datasets."""

    def __init__(self, schemas_dir: str | Path):
        self.schemas_dir = Path(schemas_dir)

    def load(self, schema_name: str) -> Dict[str, Any]:
        p = self.schemas_dir / f"{schema_name}.json"
        if not p.exists():
            raise FileNotFoundError(f"Schema not found: {p}")
        return json.loads(p.read_text(encoding="utf-8"))
