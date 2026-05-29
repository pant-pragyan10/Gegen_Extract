import time
import logging
from typing import Any, Dict, Optional
import os
import requests
from requests.adapters import HTTPAdapter, Retry
from ..schemas import LLMCallRecord
from ..persistence import PersistenceManager

logger = logging.getLogger(__name__)

# Import GeminiClient lazily to avoid circular imports when not used
from .gemini_client import GeminiClient


class GroqClient:
    """
    Backwards-compatible GroqClient facade. If the environment variable PROVIDER is
    set to 'gemini', this class will delegate to `GeminiClient` while preserving
    the same constructor and `call` signature. This allows the rest of the codebase
    to keep importing `GroqClient` without changes.
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", timeout: int = 30, max_retries: int = 2, persistence: Optional[PersistenceManager] = None):
        provider = os.environ.get('PROVIDER', 'groq').strip().lower()
        self._provider = provider
        if provider == 'gemini':
            # Use GeminiClient but keep the same attribute names where sensible
            gemini_key = os.environ.get('GEMINI_API_KEY') or api_key
            self._delegate = GeminiClient(gemini_key, model=(os.environ.get('GEMINI_MODEL') or 'gemini-2.5-flash'), timeout=timeout, max_retries=max_retries, persistence=persistence)
            self.api_key = self._delegate.api_key
            self.model = self._delegate.model
            self.timeout = self._delegate.timeout
            self.session = self._delegate.session
            self.persistence = self._delegate.persistence
        else:
            # Original Groq behavior
            # sanitize api_key: strip surrounding quotes and whitespace
            if api_key is None:
                self.api_key = None
            else:
                self.api_key = str(api_key).strip().strip('"').strip("'")
            self.model = model
            self.timeout = timeout
            self.session = requests.Session()
            retries = Retry(total=max_retries, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
            self.session.mount("https://", HTTPAdapter(max_retries=retries))
            self.persistence = persistence

    def _record(self, prompt: str, response: Optional[str], tokens: Optional[int], elapsed: float):
        if not self.persistence:
            return
        rec = LLMCallRecord(id=str(time.time_ns()), model=self.model, prompt=prompt, response=response, tokens=tokens, elapsed_seconds=elapsed)
        try:
            self.persistence.record_llm_call(rec)
        except Exception:
            logger.exception("Failed to persist LLM call")

    def call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._provider == 'gemini':
            # Delegate to Gemini client
            return self._delegate.call(prompt, temperature=temperature, max_tokens=max_tokens, metadata=metadata)

        # Original Groq call
        start = time.time()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = "https://api.groq.com/openai/v1/chat/completions"
        try:
            resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
            elapsed = time.time() - start
            resp.raise_for_status()
            data = resp.json()
            # Extract token usage and the chat content (OpenAI-compatible shape)
            tokens = None
            try:
                tokens = data.get("usage", {}).get("total_tokens")
            except Exception:
                tokens = None

            text = None
            try:
                choices = data.get("choices") if isinstance(data, dict) else None
                if choices and len(choices) > 0:
                    # OpenAI-compatible: choices[0].message.content
                    msg = choices[0].get("message") or {}
                    text = msg.get("content") or choices[0].get("text")
            except Exception:
                text = None

            self._record(prompt, text, tokens, elapsed)
            return {"raw": data, "text": text, "tokens": tokens, "elapsed": elapsed}
        except Exception:
            elapsed = time.time() - start
            self._record(prompt, None, None, elapsed)
            logger.exception("Groq API call failed")
            raise
