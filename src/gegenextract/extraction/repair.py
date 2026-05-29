import logging
from typing import Any, Dict
from .groq_client import GroqClient
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class RepairEngine:
    def __init__(self, groq: GroqClient, prompt_builder: PromptBuilder, max_attempts: int = 2):
        self.groq = groq
        self.prompt_builder = prompt_builder
        self.max_attempts = max_attempts

    def repair(self, original_text: str, errors: str, schema: Dict[str, Any], document_pages: list) -> str:
        attempts = 0
        last = original_text
        while attempts < self.max_attempts:
            attempts += 1
            repair_prompt = self._build_repair_prompt(last, errors, schema, document_pages)
            resp = self.groq.call(repair_prompt)
            text = resp.get("text")
            if not text:
                continue
            # strip common code fences/markdown wrappers that block JSON parsing
            stripped = self._strip_codefence(text)
            # caller will perform parsing/validation
            last = stripped
            logger.info("Repair attempt %d produced output", attempts)
            return last
        return original_text

    def _build_repair_prompt(self, original_text: str, errors: str, schema: Dict[str, Any], document_pages: list) -> str:
        parts = [
            "Please repair the following JSON to conform to the schema.",
            "Return ONLY the fixed JSON object. Do NOT include markdown, backticks, code fences, or any explanatory text.",
        ]
        parts.append("Original output:")
        parts.append(original_text)
        parts.append("Validation errors:")
        parts.append(str(errors))
        parts.append("Schema:")
        parts.append(schema.get("description", ""))
        parts.append("Document context (pages):")
        parts.extend(document_pages)
        return "\n".join(parts)

    def _strip_codefence(self, text: str) -> str:
        t = text.strip()
        # remove triple backtick fences with optional language
        if t.startswith('```') and t.endswith('```'):
            # strip first line if it's like ```json
            inner = t[3:-3].strip()
            # if inner begins with a language token on its first line, drop it
            lines = inner.splitlines()
            if lines and lines[0].strip().isalpha():
                return "\n".join(lines[1:]).strip()
            return inner
        # also remove single-line code fences
        if t.startswith('`') and t.endswith('`'):
            return t.strip('`').strip()
        return t
