import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List
from time import time
from ..schemas import Document, Page, DatasetSample
from ..persistence.file_cache import FileCache
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


class PdfProcessor:
    """PDF -> text extraction pipeline with OCR fallback and caching.

    Steps:
      - read PDF with PyMuPDF
      - extract page-wise text
      - detect low-text pages and trigger OCR fallback
      - store per-page text and metadata

    TODO: add chunking support hooks for downstream processing
    """

    def __init__(self, cache_dir: str | Path = "./data/cache", ocr_threshold_chars_per_page: int = 50, ocr_trigger_page_ratio: float = 0.5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = FileCache(self.cache_dir)
        self.ocr_threshold = ocr_threshold_chars_per_page
        self.ocr_trigger_ratio = ocr_trigger_page_ratio

    def is_cached(self, doc_id: str) -> bool:
        return self.cache.exists(doc_id)

    def load_from_cache(self, doc_id: str) -> DatasetSample | None:
        return self.cache.load(doc_id)

    def persist_cache(self, sample: DatasetSample) -> None:
        self.cache.save(sample)

    def extract_text_pages(self, path: str) -> List[Page]:
        doc = fitz.open(path)
        pages: List[Page] = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            rect = page.rect
            pages.append(Page(index=i, text=text, width=int(rect.width), height=int(rect.height)))
        return pages

    def needs_ocr(self, pages: List[Page]) -> bool:
        low_text_pages = sum(1 for p in pages if not p.text or len(p.text.strip()) < self.ocr_threshold)
        ratio = low_text_pages / max(1, len(pages))
        logger.debug("Low text pages: %d/%d (ratio=%.2f) threshold=%d", low_text_pages, len(pages), ratio, self.ocr_threshold)
        return ratio >= self.ocr_trigger_ratio

    def run_ocr_on_pages(self, path: str, pages: List[Page]) -> List[Page]:
        # Convert PDF pages to images and run pytesseract on each
        images = convert_from_path(path)
        for i, img in enumerate(images):
            try:
                ocr_text = pytesseract.image_to_string(img)
            except Exception:
                logger.exception("OCR failed on page %d", i)
                ocr_text = ""
            if i < len(pages):
                pages[i].ocr = ocr_text
                if not pages[i].text:
                    pages[i].text = ocr_text
            else:
                pages.append(Page(index=i, text=ocr_text, ocr=ocr_text))
        return pages

    def process(self, document: Document, use_cache: bool = True) -> DatasetSample:
        start = time()
        if use_cache and self.is_cached(document.id):
            cached = self.load_from_cache(document.id)
            if cached:
                logger.info("Loaded document %s from cache", document.id)
                return cached

        sample = DatasetSample(id=document.id, document=document, pages=[])
        try:
            pages = self.extract_text_pages(document.path)
            ocr_used = False
            if self.needs_ocr(pages):
                logger.info("OCR triggered for document %s", document.id)
                pages = self.run_ocr_on_pages(document.path, pages)
                ocr_used = True
            sample.pages = pages
            sample.metadata = {"page_count": len(pages), "ocr_used": ocr_used}
            self.persist_cache(sample)
            elapsed = time() - start
            logger.info("Processed %s: pages=%d ocr=%s time=%.2fs", document.id, len(pages), ocr_used, elapsed)
        except Exception:
            logger.exception("Failed to process document %s", document.id)
            sample.metadata = {"error": "processing_failed"}
        return sample
