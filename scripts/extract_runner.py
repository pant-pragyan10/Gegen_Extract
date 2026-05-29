"""Sample runner for extraction pipeline."""
from gegenextract.extraction.groq_client import GroqClient
from gegenextract.extraction.prompt_builder import PromptBuilder
from gegenextract.extraction.engine import ExtractionEngine
from gegenextract.extraction.repair import RepairEngine
from gegenextract.persistence import PersistenceManager
import os


def main():
    api_key = os.getenv("GROQ_API_KEY", "test-key")
    db = PersistenceManager(os.getenv("DATABASE_URL", "./data/gegenextract.db"))
    groq = GroqClient(api_key=api_key, persistence=db)
    prompt_builder = PromptBuilder()
    repair = RepairEngine(groq, prompt_builder, max_attempts=2)
    engine = ExtractionEngine(groq, prompt_builder, persistence=db, repair_engine=repair)
    # Example usage
    schema = {"description": "Example schema", "properties": {"name": {"type": "string"}}}
    pages = ["This is a document mentioning name: Alice."]
    res = engine.extract(schema, pages, model_cls=None, retries=1)
    print(res)


if __name__ == "__main__":
    main()
