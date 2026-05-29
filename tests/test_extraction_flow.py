from unittest import mock
from gegenextract.extraction.groq_client import GroqClient
from gegenextract.extraction.prompt_builder import PromptBuilder
from gegenextract.extraction.repair import RepairEngine
from gegenextract.extraction.engine import ExtractionEngine
from gegenextract.persistence import PersistenceManager
from pydantic import BaseModel


class SimpleModel(BaseModel):
    name: str


def test_malformed_json_and_repair(tmp_path):
    # Mock Groq client to first return malformed JSON, then repaired JSON
    class FakeGroq:
        def __init__(self):
            self.calls = 0

        def call(self, prompt, temperature=0.0):
            self.calls += 1
            if self.calls == 1:
                return {"text": "{name: 'Alice',}" , "raw": {}, "elapsed": 0.1}
            else:
                return {"text": '{"name": "Alice"}', "raw": {}, "elapsed": 0.2}

    groq = FakeGroq()
    prompt_builder = PromptBuilder()
    repair = RepairEngine(groq, prompt_builder, max_attempts=2)
    engine = ExtractionEngine(groq, prompt_builder, persistence=None, repair_engine=repair)
    schema = {"description": "Test schema", "properties": {"name": {"type": "string"}}}
    pages = ["Name: Alice"]
    res = engine.extract(schema, pages, model_cls=SimpleModel, retries=1)
    assert res.get("result") is not None
    assert res["result"].name == "Alice"
