from pydantic import BaseModel
from typing import Optional, Dict, Any


class ExperimentConfig(BaseModel):
    id: Optional[str]
    name: str
    seed_prompt: Dict[str, Any]
    dataset: Dict[str, Any]
    persistence: Dict[str, Any]
    budget: Dict[str, Any]
    checkpoint: Dict[str, Any]
