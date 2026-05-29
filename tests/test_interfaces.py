from gegenextract.document_processing import DocumentProcessor
from gegenextract.extraction_pipeline import Extractor
from gegenextract.prompt_management import PromptManager
from gegenextract.optimization import Optimizer
from gegenextract.scoring import Scorer


def test_interfaces_exist():
    assert DocumentProcessor
    assert Extractor
    assert PromptManager
    assert Optimizer
    assert Scorer
