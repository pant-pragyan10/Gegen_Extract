import random
from pathlib import Path
from typing import List, Dict, Iterable
import logging
from ..schemas import DatasetSample, DatasetSplit, Document, Page
from ..config import load_config

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load ExtractBench-style datasets from a local directory.

    Directory layout (expected):
      dataset_root/
        samples/
          <id>.pdf
        metadata/
          <id>.json (optional)

    This loader discovers PDF files and returns deterministic splits.
    """

    SUPPORTED_SCHEMAS = {
        "academic/research",
        "finance/10kq",
        "finance/credit_agreement",
        "hiring/resume",
        "sport/sport",
    }

    def __init__(self, root: str, schema: str = "academic/research", config_path: str | None = None):
        self.root = Path(root)
        if schema not in self.SUPPORTED_SCHEMAS:
            raise ValueError(f"Unsupported schema: {schema}")
        self.schema = schema
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = None

    def discover_pdfs(self) -> List[Path]:
        samples_dir = self.root / "samples"
        if not samples_dir.exists():
            logger.warning("samples directory not found: %s", samples_dir)
            return []
        pdfs = sorted([p for p in samples_dir.iterdir() if p.suffix.lower() in {".pdf"}])
        return pdfs

    def deterministic_split(self, ids: List[str], seed: int, ratios: Dict[str, float]) -> DatasetSplit:
        rng = random.Random(seed)
        ids_sorted = sorted(ids)
        rng.shuffle(ids_sorted)
        n = len(ids_sorted)
        n_train = int(ratios.get("train", 0.8) * n)
        n_val = int(ratios.get("val", 0.1) * n)
        train = ids_sorted[:n_train]
        val = ids_sorted[n_train : n_train + n_val]
        test = ids_sorted[n_train + n_val :]
        return DatasetSplit(train=train, val=val, test=test)

    def load(self) -> Iterable[DatasetSample]:
        pdfs = self.discover_pdfs()
        for p in pdfs:
            sample_id = p.stem
            doc = Document(id=sample_id, path=str(p))
            sample = DatasetSample(id=sample_id, document=doc, pages=[])
            # attempt to load gold/metadata if present in dataset_root/metadata
            try:
                meta_dir = self.root / "metadata"
                if meta_dir.exists():
                    # support both <id>.gold.json and <id>.json
                    candidate_files = [meta_dir / f"{sample_id}.gold.json", meta_dir / f"{sample_id}.json"]
                    for mf in candidate_files:
                        if mf.exists():
                            import json

                            with open(mf, "r", encoding="utf-8") as fh:
                                payload = json.load(fh)
                            # attach to document metadata for downstream access
                            sample.metadata["gold"] = payload
                            break
            except Exception:
                # non-fatal: continue without gold
                pass
            yield sample
