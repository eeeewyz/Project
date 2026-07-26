"""Parse product requests into strict, catalogue-backed filters."""

from fashion_rag.llm import StructuredGenerator
from fashion_rag.prompts import build_product_query_prompt
from fashion_rag.router import bound_history
from fashion_rag.schemas import ChatTurn, ProductQuery


class ProductQueryParser:
    """Extract a schema-validated product request from a user query."""

    def __init__(self, structured: StructuredGenerator, max_history_turns: int = 4):
        if max_history_turns < 0:
            raise ValueError("max_history_turns must be non-negative")
        self.structured = structured
        self.max_history_turns = max_history_turns

    def parse(self, query: str, history: list[ChatTurn]) -> ProductQuery:
        """Return filters constrained to the public catalogue schema."""

        prompt = build_product_query_prompt(
            query, bound_history(history, self.max_history_turns)
        )
        return self.structured.generate(
            prompt, ProductQuery, temperature=0, max_tokens=300
        )
