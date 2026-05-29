from typing import Any, List, Tuple
import math


def _object_similarity(a: Any, b: Any) -> float:
    # simplistic similarity: for dicts count key overlap; for strings exact match -> 1 else 0
    if isinstance(a, dict) and isinstance(b, dict):
        if not a and not b:
            return 1.0
        keys = set(a.keys()) | set(b.keys())
        matches = sum(1 for k in keys if a.get(k) == b.get(k))
        return matches / len(keys) if keys else 0.0
    if isinstance(a, str) and isinstance(b, str):
        return 1.0 if a.strip() == b.strip() else 0.0
    return 1.0 if a == b else 0.0


def align_arrays(pred: List[Any], gold: List[Any], comparator=None, strategy: str = "greedy") -> List[Tuple[Any, Any, float]]:
    """Align two arrays and return list of tuples (pred_item, gold_item, score).
    Currently implements greedy matching: for each gold, find best pred and remove it.
    """
    if not gold:
        return []
    preds = list(pred)
    matches = []
    for g in gold:
        best_idx = None
        best_score = -math.inf
        for i, p in enumerate(preds):
            score = _object_similarity(p, g)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None:
            matches.append((preds.pop(best_idx), g, best_score))
        else:
            matches.append((None, g, 0.0))
    # any leftover preds are unmatched (penalize separately if desired)
    return matches
