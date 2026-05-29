from typing import Any, Optional
from time import time


class SimpleCache:
    """A tiny in-memory cache with TTL. Replace with Redis/file-backed cache as needed."""

    def __init__(self):
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at and expires_at < time():
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = time() + ttl if ttl else 0
        self._store[key] = (expires_at, value)

    def clear(self) -> None:
        self._store.clear()
