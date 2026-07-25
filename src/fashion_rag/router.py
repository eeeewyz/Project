"""Route user requests to FAQ, product, or unsupported workflows."""

from fashion_rag.llm import StructuredGenerator
from fashion_rag.prompts import build_router_prompt
from fashion_rag.schemas import ChatTurn, Route, RouteDecision


def bound_history(history: list[ChatTurn], max_turns: int) -> list[ChatTurn]:
    """Return only the latest turns used as routing context."""

    if max_turns < 0:
        raise ValueError("max_turns must be non-negative")
    if max_turns == 0:
        return []
    return history[-max_turns:]


class QueryRouter:
    """Classify a query using a schema-validated structured generator."""

    def __init__(self, structured: StructuredGenerator, max_history_turns: int = 4):
        if max_history_turns < 0:
            raise ValueError("max_history_turns must be non-negative")
        self.structured = structured
        self.max_history_turns = max_history_turns

    def route(self, query: str, history: list[ChatTurn]) -> Route:
        prompt = build_router_prompt(
            query, bound_history(history, self.max_history_turns)
        )
        return self.structured.generate(
            prompt, RouteDecision, temperature=0, max_tokens=40
        ).route
