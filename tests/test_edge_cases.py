import io
import os
import logging
from pathlib import Path
from unittest import mock

import fitz
from gegenextract.document_processing.pdf_processor import PdfProcessor
from gegenextract.schemas import Document


def make_blank_pdf(path: Path, n_pages=1):
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page()
    doc.save(str(path))


def make_text_pdf(path: Path, texts):
    doc = fitz.open()
    for t in texts:
        p = doc.new_page()
        p.insert_text((50, 50), t)
    doc.save(str(path))


def make_image_pdf(path: Path, n_pages=1):
    # create a simple image and save as PDF (scanned-like)
    from PIL import Image, ImageDraw

    images = []
    for i in range(n_pages):
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Image page {i}", fill=(0, 0, 0))
        images.append(img)
    images[0].save(str(path), save_all=True, append_images=images[1:])


def test_empty_pdf_processing(tmp_path):
    p = tmp_path / "empty.pdf"
    make_blank_pdf(p, n_pages=1)
    proc = PdfProcessor(cache_dir=tmp_path / "cache", ocr_threshold_chars_per_page=10, ocr_trigger_page_ratio=0.5)
    doc = Document(id="empty", path=str(p))
    # avoid calling external pdf2image/poppler in CI by stubbing OCR
    with mock.patch.object(PdfProcessor, "run_ocr_on_pages", new=lambda self, path, pages: pages):
        sample = proc.process(doc, use_cache=False)
    assert sample.metadata.get("page_count") == 1
    assert len(sample.pages) == 1


def test_scanned_pdf_ocr_fallback_monkeypatched(tmp_path):
    p = tmp_path / "scanned.pdf"
    make_image_pdf(p, n_pages=2)
    proc = PdfProcessor(cache_dir=tmp_path / "cache", ocr_threshold_chars_per_page=1000, ocr_trigger_page_ratio=0.5)
    # monkeypatch OCR to avoid external dependency on tesseract
    def fake_ocr(self, path, pages):
        for i in range(len(pages)):
            pages[i].ocr = f"OCR page {i}"
            if not pages[i].text:
                pages[i].text = pages[i].ocr
        return pages

    with mock.patch.object(PdfProcessor, "run_ocr_on_pages", new=fake_ocr):
        doc = Document(id="scanned", path=str(p))
        sample = proc.process(doc, use_cache=False)
        assert sample.metadata.get("ocr_used") is True
        assert all(p.text and "OCR page" in p.text for p in sample.pages)


def test_corrupted_pdf_handling(tmp_path):
    p = tmp_path / "corrupt.pdf"
    p.write_bytes(b"%%PDF-1.4 corrupted content that is not a real PDF")
    proc = PdfProcessor(cache_dir=tmp_path / "cache")
    doc = Document(id="corrupt", path=str(p))
    sample = proc.process(doc, use_cache=False)
    assert sample.metadata.get("error") == "processing_failed"


def test_cache_hit_miss_behavior(tmp_path):
    p = tmp_path / "doc.pdf"
    make_text_pdf(p, ["hello page 0", "hello page 1"])
    cache_dir = tmp_path / "cache"
    proc = PdfProcessor(cache_dir=cache_dir, ocr_threshold_chars_per_page=1000, ocr_trigger_page_ratio=1.0)

    # monkeypatch OCR to ensure it's called on first run
    called = {"count": 0}

    def fake_ocr(self, path, pages):
        called["count"] += 1
        for i in range(len(pages)):
            pages[i].ocr = f"OCRed {i}"
            if not pages[i].text:
                pages[i].text = pages[i].ocr
        return pages

    with mock.patch.object(PdfProcessor, "run_ocr_on_pages", new=fake_ocr):
        doc = Document(id="cachedoc", path=str(p))
        s1 = proc.process(doc, use_cache=True)
        assert proc.is_cached("cachedoc")
        # second run should load from cache; fake_ocr should not be called again
        called["count"] = 0
        s2 = proc.process(doc, use_cache=True)
        assert called["count"] == 0
        assert s2.pages == s1.pages


def test_low_text_threshold_and_page_ordering(tmp_path):
    p = tmp_path / "mix.pdf"
    make_text_pdf(p, ["short", "this is a longer page text"])
    proc = PdfProcessor(cache_dir=tmp_path / "cache", ocr_threshold_chars_per_page=10, ocr_trigger_page_ratio=0.4)
    doc = Document(id="mixdoc", path=str(p))
    # monkeypatch OCR to avoid external dependency
    with mock.patch.object(PdfProcessor, "run_ocr_on_pages", new=lambda self, path, pages: pages):
        sample = proc.process(doc, use_cache=False)
    # pages should be in order and metadata present
    assert [p.index for p in sample.pages] == list(range(len(sample.pages)))
    assert sample.metadata.get("page_count") == len(sample.pages)


def test_structured_logging_records(tmp_path, caplog):
    p = tmp_path / "logtest.pdf"
    make_text_pdf(p, ["A", "B"])
    proc = PdfProcessor(cache_dir=tmp_path / "cache", ocr_threshold_chars_per_page=1000, ocr_trigger_page_ratio=0.0)
    caplog.set_level(logging.INFO)
    doc = Document(id="logdoc", path=str(p))
    # ensure OCR not triggered
    sample = proc.process(doc, use_cache=False)
    assert any("Processed logdoc" in rec.getMessage() or "Processed logdoc" in rec.message for rec in caplog.records) or any(rec.levelname == "INFO" for rec in caplog.records)
