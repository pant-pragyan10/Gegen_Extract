from pathlib import Path
from typing import Optional
import json
import tempfile
import os
from ..schemas import DatasetSample


CACHE_SCHEMA_VERSION = 1


class FileCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, doc_id: str) -> Path:
        return self.root / f"{doc_id}.json"

    def exists(self, doc_id: str) -> bool:
        return self.path_for(doc_id).exists()

    def load(self, doc_id: str) -> Optional[DatasetSample]:
        p = self.path_for(doc_id)
        if not p.exists():
            return None
        try:
            raw = p.read_text(encoding="utf-8")
            obj = json.loads(raw)
            # Backwards-compatible: if file directly contains model dump, accept it
            if isinstance(obj, dict) and "schema_version" in obj:
                version = obj.get("schema_version")
                payload = obj.get("sample")
                # future: handle different versions
                if payload is None:
                    return None
            else:
                payload = obj
            return DatasetSample.model_validate(payload)
        except Exception:
            return None

    def save(self, sample: DatasetSample) -> None:
        p = self.path_for(sample.id)
        wrapper = {"schema_version": CACHE_SCHEMA_VERSION, "sample": sample.model_dump()}
        # atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(wrapper, indent=2, ensure_ascii=False))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, str(p))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
