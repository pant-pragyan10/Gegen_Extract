from pathlib import Path
from gegenextract.document_processing.pdf_processor import PdfProcessor
import fitz


def make_pdf(path: Path, texts):
    doc = fitz.open()
    for t in texts:
        page = doc.new_page()
        page.insert_text((72, 72), t)
    doc.save(str(path))


def test_extract_text_pages(tmp_path):
    p = tmp_path / "text.pdf"
    make_pdf(p, ["Page one text", "Page two text"])
    proc = PdfProcessor(cache_dir=tmp_path / "cache", ocr_threshold_chars_per_page=10, ocr_trigger_page_ratio=0.5)
    pages = proc.extract_text_pages(str(p))
    assert len(pages) == 2
    assert "Page one" in pages[0].text


def test_needs_ocr_detection(tmp_path):
    p = tmp_path / "lowtext.pdf"
    # very small text on pages
    make_pdf(p, [".", ".", "."]) 
    proc = PdfProcessor(cache_dir=tmp_path / "cache", ocr_threshold_chars_per_page=5, ocr_trigger_page_ratio=0.5)
    pages = proc.extract_text_pages(str(p))
    assert proc.needs_ocr(pages) is True
