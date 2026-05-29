from pydantic import BaseModel
from typing import Dict, Any, List


class FieldReport(BaseModel):
    field: str
    precision: float
    recall: float
    f1: float
    support: int


class EvaluationReport(BaseModel):
    aggregate_f1: float
    fields: Dict[str, FieldReport]
    metadata: Dict[str, Any] = {}
