import logging
import time
from typing import Any, Dict, Optional, List
from ..persistence import PersistenceManager
from .groq_client import GroqClient
from .prompt_builder import PromptBuilder
from .parser import safe_parse_json, validate_with_model
from .repair import RepairEngine
from ..schemas import LLMCallRecord

logger = logging.getLogger(__name__)


class ExtractionEngine:
    def __init__(self, groq_client: GroqClient, prompt_builder: PromptBuilder, persistence: Optional[PersistenceManager] = None, repair_engine: Optional[RepairEngine] = None):
        self.groq = groq_client
        self.prompt_builder = prompt_builder
        self.persistence = persistence
        self.repair_engine = repair_engine

    def extract(self, schema: Dict[str, Any], document_pages: List[str], model_cls=None, retries: int = 1, temperature: float = 0.0) -> Dict[str, Any]:
        prompt = self.prompt_builder.build(schema, document_pages)
        attempt = 0
        last_raw = None
        while attempt < retries:
            attempt += 1
            start = time.time()
            resp = self.groq.call(prompt, temperature=temperature)
            elapsed = resp.get("elapsed")
            raw = resp.get("raw")
            text = resp.get("text")
            tokens = resp.get("tokens")
            last_raw = {"raw": raw, "text": text}
            # persist raw output
            if self.persistence:
                try:
                    # store prompt and raw output via persistence.record_llm_call already done in GroqClient
                    pass
                except Exception:
                    logger.exception("Failed to persist raw output")

            # parse
            parsed = None
            try:
                parsed = safe_parse_json(text) if text else None
            except Exception:
                parsed = None

            if parsed is None:
                logger.warning("Parsing failed, will retry or attempt repair")
                # attempt repair even when parsing fails
                if self.repair_engine and text:
                    repaired_text = self.repair_engine.repair(text, "parse_error", schema, document_pages)
                    try:
                        repaired_parsed = safe_parse_json(repaired_text)
                        if model_cls:
                            validated2, err2 = validate_with_model(model_cls, repaired_parsed)
                            if validated2 is not None:
                                # persist extraction if possible
                                if self.persistence:
                                    try:
                                        self.persistence.record_extraction(str(time.time_ns()), prompt, str(raw), repaired_text, resp.get("tokens"), elapsed)
                                    except Exception:
                                        logger.exception("Failed to persist extraction record")
                                return {"result": validated2, "raw": last_raw, "repaired": repaired_text, "elapsed": elapsed, "tokens": tokens}
                        else:
                            # when no model class is provided, return the repaired parsed JSON
                            if repaired_parsed is not None:
                                if self.persistence:
                                    try:
                                        self.persistence.record_extraction(str(time.time_ns()), prompt, str(raw), repaired_text, resp.get("tokens"), elapsed)
                                    except Exception:
                                        logger.exception("Failed to persist extraction record")
                                return {"result": repaired_parsed, "raw": last_raw, "repaired": repaired_text, "elapsed": elapsed, "tokens": tokens}
                    except Exception:
                        logger.exception("Repair parsing failed")
                # else will loop to retry or exit
            else:
                if model_cls:
                    validated, err = validate_with_model(model_cls, parsed)
                    if validated is not None:
                        return {"result": validated, "raw": last_raw, "elapsed": elapsed, "tokens": tokens}
                    else:
                        # try repair
                        if self.repair_engine:
                            repaired_text = self.repair_engine.repair(text, err, schema, document_pages)
                            try:
                                repaired_parsed = safe_parse_json(repaired_text)
                                if model_cls:
                                    validated2, err2 = validate_with_model(model_cls, repaired_parsed)
                                    if validated2 is not None:
                                        return {"result": validated2, "raw": last_raw, "repaired": repaired_text, "elapsed": elapsed, "tokens": tokens}
                            except Exception:
                                logger.exception("Repair parsing failed")
                else:
                    return {"result": parsed, "raw": last_raw, "elapsed": elapsed, "tokens": tokens}

        # if we exit loop, return failure object
        return {"result": None, "raw": last_raw, "error": "extraction_failed"}
