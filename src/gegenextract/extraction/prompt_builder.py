from typing import Dict, Any, List


class PromptBuilder:
    """Build modular prompts including instructions, schema, and document context."""

    def __init__(self, system_instructions: str = "Extract structured data as JSON."):
        self.system_instructions = system_instructions

    def build(self, schema: Dict[str, Any], document_pages: List[str], extraction_instructions: str = "Extract fields matching schema") -> str:
        parts = [self.system_instructions]
        parts.append("\n---\nJSON Schema:\n")
        # include a compact representation of schema to guide the model
        parts.append(schema.get("description", ""))
        parts.append("\nSchema properties:\n")
        properties = schema.get("properties", {})
        for name, prop in properties.items():
            parts.append(f"- {name}: {prop.get('type')} - {prop.get('description', '')}")
        parts.append("\n---\nExtraction instructions:\n")
        parts.append(extraction_instructions)
        parts.append("\nDocument pages (page separated):\n")
        parts.extend([f"[Page {i}] {p}" for i, p in enumerate(document_pages)])
        parts.append("\nReturn valid JSON conforming to the schema. Avoid hallucination.")
        return "\n".join(parts)
