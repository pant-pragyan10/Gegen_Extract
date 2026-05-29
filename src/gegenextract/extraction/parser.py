import json
import logging
from typing import Any, Dict
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        # Try to recover common issues: replace single quotes, trailing commas
        t = text.replace("\'", '"')
        t = t.replace(",}\n", "}\n")
        try:
            return json.loads(t)
        except Exception:
            logger.exception("Failed to parse JSON")
            raise


def validate_with_model(model_cls, data: Dict[str, Any]):
    try:
        return model_cls.model_validate(data), None
    except ValidationError as e:
        return None, e
