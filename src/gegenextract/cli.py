"""Minimal CLI entrypoints for GegenExtract."""
import argparse
from .config import load_config
from .logging_config import configure_logging


def main():
    parser = argparse.ArgumentParser("GegenExtract CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.logging.level, getattr(cfg.logging, "file", None))
    print(f"Loaded config for {cfg.app.get('name')}")


if __name__ == "__main__":
    main()
