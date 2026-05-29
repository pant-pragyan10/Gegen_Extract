from typing import Tuple
from .artifact import PromptArtifact
from difflib import unified_diff


def prompt_diff(a: PromptArtifact, b: PromptArtifact) -> str:
    return a.diff(b)


def prompt_hash(a: PromptArtifact) -> str:
    return a.serialize()
