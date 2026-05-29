import tempfile
from pathlib import Path
from gegenextract.dataset.loader import DatasetLoader
import fitz


def make_pdf(path: Path, text: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))


def test_deterministic_split(tmp_path):
    root = tmp_path / "dataset"
    samples = root / "samples"
    samples.mkdir(parents=True)
    # create 5 pdfs
    ids = [f"doc{i}" for i in range(5)]
    for i in ids:
        make_pdf(samples / f"{i}.pdf", f"Content {i}")

    loader = DatasetLoader(str(root), schema="academic/research")
    pdfs = list(loader.discover_pdfs())
    assert len(pdfs) == 5

    split1 = loader.deterministic_split([p.stem for p in pdfs], seed=123, ratios={"train": 0.6, "val": 0.2, "test": 0.2})
    split2 = loader.deterministic_split([p.stem for p in pdfs], seed=123, ratios={"train": 0.6, "val": 0.2, "test": 0.2})
    assert split1.train == split2.train
    assert len(split1.train) + len(split1.val) + len(split1.test) == 5
