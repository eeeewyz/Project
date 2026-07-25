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


def build_product_query_prompt(query: str, history: list[ChatTurn]) -> str:
    """Build a constrained product-filter extraction prompt."""

    return f'''Extract a product-catalog request as JSON.
Allowed nature values: "technical" or "creative".
technical: explicit catalog constraints such as product type, colour, price,
gender, season, usage, or a requested number of products.
creative: open-ended outfit or style recommendation where the assistant should
infer suitable catalog filters from a desired occasion, aesthetic, or look.
requested_count must be an integer from 1 to 5; use 3 when unspecified.
Allowed filter keys: gender, master_category, article_type, base_colour,
usage, season, min_price, max_price.
Categorical filters are JSON lists or null. Never output "Any".
Price bounds are numbers or null.

Catalog values:
gender: Men, Women, Unisex
master_category: Apparel, Footwear
article_type: T-shirt, Dress, Shirt, Sneakers
base_colour: Blue, Black, White, Red
usage: Casual, Formal, Sports
season: Spring, Summer, Fall

Recent conversation:
{format_history(history)}

Current query:
{query}

Return JSON only.'''
