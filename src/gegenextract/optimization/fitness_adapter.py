from __future__ import annotations
from typing import Callable, Dict, Any, List, Tuple, Optional
import hashlib
from gegenextract.optimization.artifact import PromptArtifact
from gegenextract.experiment.persistence import PersistenceManager
import time


def _prompt_hash(artifact: PromptArtifact) -> str:
    return hashlib.sha1(artifact.serialize().encode("utf-8")).hexdigest()


class FitnessAdapter:
    def __init__(self, persistence: PersistenceManager, extraction_fn: Callable[[PromptArtifact, Dict[str, Any]], Any], evaluator_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]], cache_enabled: bool = True):
        """extraction_fn(artifact, sample) -> prediction
           evaluator_fn(prediction, gold) -> dict of field metrics / diagnostics
        """
        self.persistence = persistence
        self.extraction_fn = extraction_fn
        self.evaluator_fn = evaluator_fn
        self.cache_enabled = cache_enabled

    def evaluate_prompt_on_sample(self, artifact: PromptArtifact, sample: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        # run extraction
        try:
            pred = self.extraction_fn(artifact, sample)
        except Exception as e:
            # malformed output or extraction failure
            diagnostics = {"malformed_json": True, "error": str(e)}
            return 0.0, diagnostics
        # evaluate
        diag = self.evaluator_fn(pred, sample.get("gold", {}))
        # compute simple aggregate fitness: mean f1 across fields
        f1s = []
        for k, v in diag.items():
            f1 = v.get("f1") if isinstance(v, dict) else getattr(v, 'f1', None)
            if f1 is not None:
                f1s.append(float(f1))
        agg = sum(f1s) / len(f1s) if f1s else 0.0
        # synthesize diagnostics summary for mutation engine
        diagnostics = {
            "malformed_json": diag.get("__malformed__", False) or False,
            "low_array_recall": any((v.get("recall", 1.0) < 0.5) for v in diag.values() if isinstance(v, dict)),
            "hallucinated_fields": [k for k, v in diag.items() if isinstance(v, dict) and v.get("precision", 1.0) < 0.5],
        }
        return agg, diagnostics

    def evaluate_prompt(self, artifact: PromptArtifact, samples: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        ph = _prompt_hash(artifact)
        if self.cache_enabled:
            cached = self.persistence.get_cached_evaluation(ph)
            if cached:
                return cached["score"], cached.get("report", {})
        # batch evaluate
        scores = []
        merged_diag = {"malformed_json": False, "low_array_recall": False, "hallucinated_fields": []}
        for s in samples:
            sc, diag = self.evaluate_prompt_on_sample(artifact, s)
            scores.append(sc)
            merged_diag["malformed_json"] = merged_diag["malformed_json"] or diag.get("malformed_json", False)
            merged_diag["low_array_recall"] = merged_diag["low_array_recall"] or diag.get("low_array_recall", False)
            merged_diag["hallucinated_fields"].extend(diag.get("hallucinated_fields", []))
        agg_score = sum(scores) / len(scores) if scores else 0.0
        report = {"score": agg_score, "diagnostics": merged_diag}
        if self.cache_enabled:
            self.persistence.persist_cached_evaluation(ph, agg_score, report)
        return agg_score, report
