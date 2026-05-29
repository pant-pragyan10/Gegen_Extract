"""Sample runner to ingest a dataset using GegenExtract PdfProcessor."""
import argparse
from gegenextract.dataset.loader import DatasetLoader
from gegenextract.document_processing.pdf_processor import PdfProcessor
from gegenextract.config import load_config
from gegenextract.logging_config import configure_logging


def main():
    parser = argparse.ArgumentParser("ingest-runner")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.logging.level, getattr(cfg.logging, "file", None))

    loader = DatasetLoader(args.dataset_root, schema="academic/research", config_path=args.config)
    ingestion_cfg = cfg.ingestion or {}
    processor = PdfProcessor(
        cache_dir=ingestion_cfg.get("cache_dir", "./data/cache"),
        ocr_threshold_chars_per_page=ingestion_cfg.get("ocr_threshold_chars_per_page", 50),
        ocr_trigger_page_ratio=ingestion_cfg.get("ocr_trigger_page_ratio", 0.5),
    )

    for sample in loader.load():
        processed = processor.process(sample.document)
        print(f"Processed: {processed.id} pages={len(processed.pages)} ocr={processed.metadata.get('ocr_used')}")


if __name__ == "__main__":
    main()
