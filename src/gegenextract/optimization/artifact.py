from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import json
from difflib import unified_diff


class PromptArtifact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    system: Optional[str] = None
    instructions: Optional[str] = None
    constraints: Optional[str] = None
    repair: Optional[str] = None
    examples: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    def serialize(self) -> str:
        # use model_dump for pydantic v2 compatibility when available
        try:
            data = self.model_dump()
        except Exception:
            data = json.loads(self.json())
        return json.dumps(data, sort_keys=True, indent=2)

    def diff(self, other: "PromptArtifact") -> str:
        a = self.serialize().splitlines(keepends=True)
        b = other.serialize().splitlines(keepends=True)
        return "".join(unified_diff(a, b, fromfile=f"v{self.version}", tofile=f"v{other.version}"))

    def bumped(self, change_metadata: Dict[str, Any] = None) -> "PromptArtifact":
        new = self.model_copy(deep=True)
        new.version = self.version + 1
        if change_metadata:
            new.metadata = {**self.metadata, **change_metadata}
        return new
