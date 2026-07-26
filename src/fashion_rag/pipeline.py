"""End-to-end orchestration with citation grounding for fashion RAG."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from fashion_rag.llm import LLMClient
from fashion_rag.prompts import (
    build_faq_answer_prompt,
    build_product_answer_prompt,
)
from fashion_rag.query_parser import ProductQueryParser
from fashion_rag.retriever import (
    FAQRetriever,
    ProductRetriever,
    SOFT_FILTER_ORDER,
)
from fashion_rag.router import QueryRouter, bound_history
from fashion_rag.schemas import (
    ChatTurn,
    PipelineResponse,
    ProductFilters,
    ProductQuery,
    RetrievalHit,
    Route,
    TaskNature,
)


UNSUPPORTED_MESSAGE = (
    "I can help with fashion products and store policies. "
    "Please ask about the catalog, outfits, shipping, returns, sizing, or support."
)
NO_PRODUCT_MESSAGE = (
    "I could not find a matching product without violating your core "
    "product-type or price constraints. Try changing a color, season, or usage."
)
NO_FAQ_MESSAGE = (
    "I could not find a relevant store policy for that question. "
    "Please ask about shipping, returns, sizing, orders, payment, or support."
)
GROUNDING_MESSAGE = "The model cited items outside the retrieved context."

PRODUCT_ID_PATTERN = re.compile(r"P\d{4}")
FAQ_ID_PATTERN = re.compile(r"FAQ-\d{2}")
_CITATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:"
    r"(?P<product>[Pp])(?P<product_separator>[-_]?)(?P<product_digits>\d+)"
    r"|"
    r"(?P<faq>[Ff][Aa][Qq])(?P<faq_separator>[-_]?)(?P<faq_digits>\d+)"
    r")"
    r"(?![A-Za-z0-9_-])"
)


class ApplicationError(RuntimeError):
    """Public application-layer failure without provider implementation detail."""


class GroundingError(ApplicationError):
    """Raised when generation cites evidence outside the retrieved context."""


def _canonical_citation(match: re.Match[str]) -> str | None:
    """Return the canonical form of a citation-like token when well formed."""

    product_digits = match.group("product_digits")
    if product_digits is not None:
        return f"P{product_digits}" if len(product_digits) == 4 else None

    faq_digits = match.group("faq_digits")
    if faq_digits is not None:
        return f"FAQ-{faq_digits}" if len(faq_digits) == 2 else None
    return None


def _canonicalize_grounded_answer(
    answer: str,
    allowed_ids: set[str],
) -> str | None:
    """Validate all citation-like tokens and normalize allowed aliases.

    Bare numbers and tokens embedded in a larger identifier are intentionally
    ignored. Citation-like product and FAQ tokens are both recognized so a
    cross-route citation cannot evade grounding validation.
    """

    matches = list(_CITATION_PATTERN.finditer(answer))
    if not matches:
        return None

    replacements: list[tuple[int, int, str]] = []
    for match in matches:
        canonical = _canonical_citation(match)
        if canonical is None or canonical not in allowed_ids:
            return None
        replacements.append((match.start(), match.end(), canonical))

    normalized = answer
    for start, end, canonical in reversed(replacements):
        normalized = f"{normalized[:start]}{canonical}{normalized[end:]}"
    return normalized


def _is_grounded(
    answer: str,
    allowed_ids: set[str],
    pattern: re.Pattern[str],
) -> bool:
    """Compatibility predicate backed by the stricter citation validator."""

    del pattern
    return _canonicalize_grounded_answer(answer, allowed_ids) is not None


def _validate_allowed_ids(
    hits: Sequence[RetrievalHit],
    *,
    route: Route,
) -> set[str]:
    pattern = PRODUCT_ID_PATTERN if route is Route.PRODUCT else FAQ_ID_PATTERN
    label = "product" if route is Route.PRODUCT else "FAQ"
    allowed_ids = {hit.record_id for hit in hits}
    if len(allowed_ids) != len(hits) or any(
        pattern.fullmatch(record_id) is None for record_id in allowed_ids
    ):
        raise ApplicationError(
            f"Retrieved evidence must use unique canonical {label} IDs."
        )
    return allowed_ids


def _effective_filters(
    filters: ProductFilters,
    hits: Sequence[RetrievalHit],
) -> ProductFilters:
    """Remove relaxed soft filters while retaining every hard constraint."""

    if hits:
        relaxed = set(hits[0].relaxed_filters)
        if any(set(hit.relaxed_filters) != relaxed for hit in hits[1:]):
            raise ApplicationError(
                "Retrieved products reported inconsistent filter relaxation."
            )
    else:
        # The product retriever exhausts all supplied soft constraints before
        # returning no hits. Reconstruct that deterministic terminal state.
        relaxed = {
            field for field in SOFT_FILTER_ORDER if getattr(filters, field)
        }

    unknown = relaxed.difference(SOFT_FILTER_ORDER)
    if unknown:
        raise ApplicationError("Retrieved products reported unknown filters.")
    return filters.model_copy(
        update={field: None for field in SOFT_FILTER_ORDER if field in relaxed}
    )


class FashionRAGPipeline:
    """Route, retrieve, generate, and verify grounded fashion answers."""

    def __init__(
        self,
        *,
        router: QueryRouter,
        product_parser: ProductQueryParser,
        product_retriever: ProductRetriever,
        faq_retriever: FAQRetriever,
        llm: LLMClient,
        top_k: int = 5,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.router = router
        self.product_parser = product_parser
        self.product_retriever = product_retriever
        self.faq_retriever = faq_retriever
        self.llm = llm
        self.top_k = top_k

    @staticmethod
    def _response(
        *,
        started_at: float,
        answer: str,
        route: Route,
        nature: TaskNature | None = None,
        applied_filters: ProductFilters | None = None,
        hits: list[RetrievalHit] | None = None,
    ) -> PipelineResponse:
        return PipelineResponse(
            answer=answer,
            route=route,
            nature=nature,
            applied_filters=applied_filters,
            hits=hits or [],
            latency_ms=float((time.perf_counter() - started_at) * 1000),
        )

    def _generate_grounded(
        self,
        prompt: str,
        *,
        hits: list[RetrievalHit],
        route: Route,
        temperature: float,
    ) -> str:
        allowed_ids = _validate_allowed_ids(hits, route=route)
        current_prompt = prompt
        for attempt in range(2):
            answer = self.llm.complete(
                current_prompt,
                temperature=temperature,
                max_tokens=400,
            )
            normalized = _canonicalize_grounded_answer(answer, allowed_ids)
            if normalized is not None:
                return normalized
            if attempt == 0:
                ids = ", ".join(sorted(allowed_ids))
                current_prompt = f"{prompt}\n\nUse only these IDs: {ids}."
        raise GroundingError(GROUNDING_MESSAGE)

    def _answer_faq(
        self,
        query: str,
        *,
        started_at: float,
    ) -> PipelineResponse:
        hits = self.faq_retriever.search(query, top_k=3)
        if not hits:
            return self._response(
                started_at=started_at,
                answer=NO_FAQ_MESSAGE,
                route=Route.FAQ,
            )
        answer = self._generate_grounded(
            build_faq_answer_prompt(query, hits),
            hits=hits,
            route=Route.FAQ,
            temperature=0.1,
        )
        return self._response(
            started_at=started_at,
            answer=answer,
            route=Route.FAQ,
            hits=hits,
        )

    def _answer_product(
        self,
        query: str,
        history: list[ChatTurn],
        *,
        started_at: float,
    ) -> PipelineResponse:
        product_query: ProductQuery = self.product_parser.parse(query, history)
        hits = self.product_retriever.search(
            query,
            product_query.filters,
            self.top_k,
        )
        applied_filters = _effective_filters(product_query.filters, hits)
        if not hits:
            return self._response(
                started_at=started_at,
                answer=NO_PRODUCT_MESSAGE,
                route=Route.PRODUCT,
                nature=product_query.nature,
                applied_filters=applied_filters,
            )
        temperature = (
            0.2
            if product_query.nature is TaskNature.TECHNICAL
            else 0.8
        )
        answer = self._generate_grounded(
            build_product_answer_prompt(query, product_query, hits),
            hits=hits,
            route=Route.PRODUCT,
            temperature=temperature,
        )
        return self._response(
            started_at=started_at,
            answer=answer,
            route=Route.PRODUCT,
            nature=product_query.nature,
            applied_filters=applied_filters,
            hits=hits,
        )

    def answer(
        self,
        query: str,
        history: list[ChatTurn],
    ) -> PipelineResponse:
        """Return a bounded-history, grounded answer for one user query."""

        started_at = time.perf_counter()
        recent_history = bound_history(history, 4)
        try:
            route = self.router.route(query, recent_history)
            if route is Route.UNSUPPORTED:
                return self._response(
                    started_at=started_at,
                    answer=UNSUPPORTED_MESSAGE,
                    route=route,
                )
            if route is Route.FAQ:
                return self._answer_faq(query, started_at=started_at)
            if route is Route.PRODUCT:
                return self._answer_product(
                    query,
                    recent_history,
                    started_at=started_at,
                )
            raise ApplicationError("The router returned an unsupported route.")
        except ApplicationError:
            raise
        except Exception as exc:
            raise ApplicationError(
                "The assistant could not process the request."
            ) from exc
