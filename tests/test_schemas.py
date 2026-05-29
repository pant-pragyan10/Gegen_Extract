from gegenextract.schemas import Document, ExtractionResult, PromptVersion


def test_document_schema():
    d = Document(id="doc1", path="/tmp/doc.pdf")
    assert d.id == "doc1"


def test_prompt_version():
    p = PromptVersion(id="v1", template="Hello {{name}}")
    assert p.template.startswith("Hello")


def test_extraction_result():
    r = ExtractionResult(document_id="doc1", extracted={"field": "value"})
    assert r.extracted["field"] == "value"
