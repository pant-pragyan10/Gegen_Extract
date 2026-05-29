import logging
import logging.config
from typing import Optional


DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}


def configure_logging(level: Optional[str] = None, logfile: Optional[str] = None) -> None:
    cfg = DEFAULT_LOGGING_CONFIG.copy()
    if level:
        cfg["root"]["level"] = level
        cfg["handlers"]["console"]["level"] = level
    if logfile:
        cfg["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "filename": logfile,
            "formatter": "default",
            "level": level or "INFO",
        }
        cfg["root"]["handlers"].append("file")
    logging.config.dictConfig(cfg)
