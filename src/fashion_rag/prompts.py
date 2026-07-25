"""Prompt builders for the fashion RAG assistant."""

from fashion_rag.schemas import ChatTurn


def format_history(history: list[ChatTurn]) -> str:
    """Format prior chat turns for inclusion in an LLM prompt."""

    if not history:
        return "(no previous conversation)"
    return "\n".join(f"{turn.role}: {turn.content}" for turn in history)


def build_router_prompt(query: str, history: list[ChatTurn]) -> str:
    """Build the structured classification prompt for a user query."""

    return f'''Classify a fashion-store assistant query.
Return JSON with exactly one route: "faq", "product", or "unsupported".
faq: store policy, delivery, returns, payment, sizing, order, or support.
product: catalog search, product comparison, or outfit recommendation.
unsupported: everything outside those scopes.

Recent conversation:
{format_history(history)}

Current query:
{query}

Return JSON only, for example {{"route":"faq"}}.'''
