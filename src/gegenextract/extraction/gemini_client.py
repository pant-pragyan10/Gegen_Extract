import time
import logging
from typing import Any, Dict, Optional
import os
import requests
from requests.adapters import HTTPAdapter, Retry
from ..schemas import LLMCallRecord
from ..persistence import PersistenceManager

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", timeout: int = 30, max_retries: int = 2, persistence: Optional[PersistenceManager] = None):
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
        # SSL verification toggle for local testing (default: True). Set GEMINI_VERIFY_SSL=false to disable.
        try:
            env_val = os.environ.get('GEMINI_VERIFY_SSL', 'true').strip().lower()
            self.verify = not (env_val in ('0', 'false', 'no'))
        except Exception:
            self.verify = True
        # Startup diagnostic: print endpoint template, model, provider
        try:
            endpoint_template = "https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
            provider = os.environ.get('PROVIDER', 'gemini')
            logger.info("GeminiClient startup - endpoint template: %s", endpoint_template)
            logger.info("GeminiClient startup - model: %s", model)
            logger.info("GeminiClient startup - provider: %s", provider)
            # Also print to stdout for visibility in Streamlit logs
            print(f"[GeminiClient] endpoint_template={endpoint_template} model={model} provider={provider} GEMINI_VERIFY_SSL={self.verify}")
        except Exception:
            pass

    def _record(self, prompt: str, response: Optional[str], tokens: Optional[int], elapsed: float):
        if not self.persistence:
            return
        rec = LLMCallRecord(id=str(time.time_ns()), model=self.model, prompt=prompt, response=response, tokens=tokens, elapsed_seconds=elapsed)
        try:
            self.persistence.record_llm_call(rec)
        except Exception:
            logger.exception("Failed to persist LLM call")

    def _extract_text(self, data: Dict[str, Any]) -> Optional[str]:
        # Try several known shapes from Google Generative APIs
        try:
            if not isinstance(data, dict):
                return None
            # new Generative API: 'candidates' or 'output' or 'content'
            if 'candidates' in data and isinstance(data['candidates'], list) and len(data['candidates']) > 0:
                c = data['candidates'][0]
                if isinstance(c, dict):
                    return c.get('content') or c.get('text') or None
                if isinstance(c, str):
                    return c
            if 'output' in data:
                out = data['output']
                if isinstance(out, dict):
                    # sometimes output.text or output[0].content
                    if 'text' in out:
                        return out.get('text')
                    if 'content' in out:
                        return out.get('content')
                if isinstance(out, list) and len(out) > 0:
                    first = out[0]
                    if isinstance(first, dict):
                        return first.get('content') or first.get('text')
            # legacy: top-level 'response' or 'text'
            if 'response' in data and isinstance(data['response'], dict):
                return data['response'].get('output') or data['response'].get('text')
            if 'text' in data:
                return data.get('text')
        except Exception:
            pass
        return None

    def call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start = time.time()
        # We'll try the canonical generateContent endpoint and payload shape per spec.
        # Also support trying a fallback model (gemini-1.5-flash) before the requested model for testing.
        models_to_try = []
        # if user requested 2.5, try 1.5 first for compatibility
        if isinstance(self.model, str) and '2.5' in self.model:
            models_to_try.append('gemini-1.5-flash')
        models_to_try.append(self.model)

        endpoint_template = "https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

        last_exc = None
        for model in models_to_try:
            url = endpoint_template.replace('{MODEL}', model)
            # prepare standard generateContent payload per spec
            body = {
                "contents": [
                    {"parts": [{"text": prompt}]}
                ]
            }

            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"} if self.api_key else {"Content-Type": "application/json"}
            try:
                # Try with Bearer header first
                resp = self.session.post(url, json=body, headers=headers, timeout=self.timeout, verify=self.verify)
                elapsed = time.time() - start
                resp.raise_for_status()
                data = resp.json()
                # parse generateContent response
                text = None
                try:
                    # typical shape: {"candidates":[{"content":[{"text":"..."}]}]}
                    candidates = data.get('candidates') or []
                    if isinstance(candidates, list) and candidates:
                        first = candidates[0]
                        content = first.get('content') or first.get('output') or []
                        if isinstance(content, list):
                            parts = []
                            for it in content:
                                if isinstance(it, dict):
                                    t = it.get('text') or it.get('content') or it.get('payload')
                                    if t:
                                        parts.append(str(t))
                                elif isinstance(it, str):
                                    parts.append(it)
                            text = '\n'.join(parts) if parts else None
                    # fallback shapes
                    if not text:
                        # sometimes: data['output'][0]['content'][0]['text']
                        out = data.get('output')
                        if isinstance(out, list) and out:
                            first = out[0]
                            if isinstance(first, dict):
                                cont = first.get('content')
                                if isinstance(cont, list) and cont:
                                    segs = [c.get('text') for c in cont if isinstance(c, dict) and c.get('text')]
                                    if segs:
                                        text = '\n'.join(segs)
                    # last resort: top-level text
                    if not text:
                        text = data.get('text') or data.get('response') or None

                except Exception:
                    text = None

                tokens = None
                try:
                    tokens = data.get('usage', {}).get('total_tokens')
                except Exception:
                    tokens = None

                self._record(prompt, text, tokens, elapsed)
                return {"raw": data, "text": text, "tokens": tokens, "elapsed": elapsed, "endpoint": url, "model": model}
            except requests.exceptions.SSLError as ssl_e:
                # Provide clearer guidance about SSL issues (corporate proxies, MITM, or wrong host)
                logger.exception("Gemini SSL error when calling %s: %s", url, ssl_e)
                last_exc = ssl_e
                # SSL problems are unlikely to be resolved by trying other endpoints; break to report helpful error
                break
            except requests.exceptions.HTTPError as http_e:
                # If auth via Bearer fails (401/403), try API key as query param
                status = None
                try:
                    status = http_e.response.status_code
                except Exception:
                    status = None
                if status in (401, 403) and self.api_key:
                    try:
                        url_with_key = f"{url}?key={self.api_key}"
                        resp = self.session.post(url_with_key, json=body, headers={"Content-Type": "application/json"}, timeout=self.timeout, verify=self.verify)
                        elapsed = time.time() - start
                        resp.raise_for_status()
                        data = resp.json()
                        # parse as above
                        text = None
                        try:
                            candidates = data.get('candidates') or []
                            if isinstance(candidates, list) and candidates:
                                first = candidates[0]
                                content = first.get('content') or []
                                if isinstance(content, list):
                                    parts = []
                                    for it in content:
                                        if isinstance(it, dict):
                                            t = it.get('text') or it.get('content')
                                            if t:
                                                parts.append(str(t))
                                        elif isinstance(it, str):
                                            parts.append(it)
                                    text = '\n'.join(parts) if parts else None
                        except Exception:
                            text = None
                        tokens = None
                        try:
                            tokens = data.get('usage', {}).get('total_tokens')
                        except Exception:
                            tokens = None
                        self._record(prompt, text, tokens, elapsed)
                        return {"raw": data, "text": text, "tokens": tokens, "elapsed": elapsed, "endpoint": url_with_key, "model": model}
                    except Exception as e2:
                        last_exc = e2
                        logger.debug("Gemini endpoint with key %s failed: %s", url, e2)
                        continue
                last_exc = http_e
                logger.debug("Gemini endpoint %s failed: %s", url, http_e)
                continue
            except Exception as e:
                last_exc = e
                logger.debug("Gemini endpoint %s failed: %s", url, e)
                continue

        elapsed = time.time() - start
        self._record(prompt, None, None, elapsed)
        # If the last exception was an SSL error, raise a clearer runtime error with guidance
        if last_exc and isinstance(last_exc, requests.exceptions.SSLError):
            raise RuntimeError(
                "SSL error when contacting Gemini endpoints. This may be caused by a corporate proxy, local MITM, or incorrect hostname. "
                "Verify network, proxy settings, and that the GEMINI endpoint is reachable from this host."
            ) from last_exc

        logger.exception("Gemini API call failed; tried endpoints")
        raise last_exc or RuntimeError("Gemini client failed to call any endpoint")
