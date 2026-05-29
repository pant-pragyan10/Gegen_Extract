from typing import Any, Dict, List, Tuple
import logging
from pydantic import BaseModel
from .alignment import align_arrays
from .semantic import SemanticComparator, InMemorySemanticCache

logger = logging.getLogger(__name__)


class FieldMetric(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


def _f1_from_pr(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


class Evaluator:
    def __init__(self, semantic_backend: SemanticComparator = None):
        self.semantic = semantic_backend or SemanticComparator(InMemorySemanticCache())

    def normalize(self, value: Any) -> Any:
        # simple normalization: strip strings, lower-case
        if isinstance(value, str):
            return value.strip()
        return value

    def compare_leaf(self, pred, gold) -> Tuple[float, float]:
        # return (match, support) where match is 1/0 or semantic similarity
        if pred is None and gold is None:
            return 1.0, 1
        if pred is None or gold is None:
            return 0.0, 1
        if isinstance(gold, str):
            sim = self.semantic.similarity(self.normalize(pred), self.normalize(gold))
            return float(sim), 1
        if isinstance(gold, (int, float)):
            return (1.0 if pred == gold else 0.0), 1
        # fallback exact
        return (1.0 if pred == gold else 0.0), 1

    def score(self, prediction: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, FieldMetric]:
        results: Dict[str, FieldMetric] = {}

        def recurse(pred, gld, path=""):
            if isinstance(gld, dict):
                for k, v in gld.items():
                    p_val = pred.get(k) if isinstance(pred, dict) else None
                    recurse(p_val, v, f"{path}.{k}" if path else k)
            elif isinstance(gld, list):
                # align arrays
                p_list = pred if isinstance(pred, list) else []
                matches = align_arrays(p_list, gld, comparator=self)
                # aggregate per-item metrics
                for idx, (p_item, g_item, score_val) in enumerate(matches):
                    key = f"{path}[{idx}]"
                    pmatch, support = self.compare_leaf(p_item, g_item) if not isinstance(g_item, (dict, list)) else (score_val, 1)
                    fm = results.setdefault(key, FieldMetric())
                    fm.precision += pmatch
                    fm.recall += pmatch
                    fm.support += support
            else:
                # leaf
                pmatch, support = self.compare_leaf(pred, gld)
                key = path
                fm = results.setdefault(key, FieldMetric())
                fm.precision += pmatch
                fm.recall += pmatch
                fm.support += support

        recurse(prediction, gold)

        # finalize metrics
        for k, fm in results.items():
            if fm.support > 0:
                p = fm.precision / fm.support
                r = fm.recall / fm.support
                fm.precision = p
                fm.recall = r
                fm.f1 = _f1_from_pr(p, r)
        return results
