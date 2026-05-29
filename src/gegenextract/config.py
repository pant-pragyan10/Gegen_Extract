import yaml
from pathlib import Path
from .schemas import AppConfig


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config and parse into Pydantic AppConfig."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.parse_obj(raw)

