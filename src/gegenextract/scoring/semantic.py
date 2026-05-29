from typing import Dict, Tuple
import hashlib


class InMemorySemanticCache:
    def __init__(self):
        self.cache: Dict[str, float] = {}

    def _key(self, a: str, b: str) -> str:
        h = hashlib.sha256()
        h.update(a.encode("utf-8"))
        h.update(b.encode("utf-8"))
        return h.hexdigest()

    def get(self, a: str, b: str):
        return self.cache.get(self._key(a, b))

    def set(self, a: str, b: str, val: float):
        self.cache[self._key(a, b)] = val


class SemanticComparator:
    def __init__(self, cache: InMemorySemanticCache = None):
        self.cache = cache or InMemorySemanticCache()

    def similarity(self, a: str, b: str) -> float:
        if a is None or b is None:
            return 0.0
        key = self.cache._key(a, b)
        if key in self.cache.cache:
            return self.cache.cache[key]
        # lightweight deterministic similarity: token Jaccard
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens and not b_tokens:
            s = 1.0
        else:
            s = len(a_tokens & b_tokens) / len(a_tokens | b_tokens) if (a_tokens | b_tokens) else 0.0
        self.cache.set(a, b, s)
        return s
