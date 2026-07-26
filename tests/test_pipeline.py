from __future__ import annotations

import pytest

from fashion_rag.pipeline import (
    ApplicationError,
    FashionRAGPipeline,
    GroundingError,
)
from fashion_rag.schemas import (
    ChatTurn,
    ProductFilters,
    ProductQuery,
    RetrievalHit,
    Route,
    TaskNature,
)
from tests.fakes import FakeLLM


class StubRouter:
    def __init__(self, route: Route):
        self.value = route
        self.calls: list[tuple[str, list[ChatTurn]]] = []

    def route(self, query: str, history: list[ChatTurn]) -> Route:
        self.calls.append((query, history))
        return self.value


class StubParser:
    def __init__(self, product_query: ProductQuery | None = None):
        self.value = product_query or ProductQuery(
            nature=TaskNature.TECHNICAL,
            requested_count=3,
            filters=ProductFilters(),
        )
        self.calls: list[tuple[str, list[ChatTurn]]] = []

    def parse(self, query: str, history: list[ChatTurn]) -> ProductQuery:
        self.calls.append((query, history))
        return self.value


class StubRetriever:
    def __init__(self, hits: list[RetrievalHit]):
        self.hits = hits
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def search(self, *args: object, **kwargs: object) -> list[RetrievalHit]:
        self.calls.append((args, kwargs))
        return self.hits


def product_hit(
    record_id: str,
    *,
    relaxed_filters: list[str] | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        record_id=record_id,
        text=f"{record_id} Blue T-shirt",
        score=0.9,
        metadata={
            "product_id": record_id,
            "article_type": "T-shirt",
            "base_colour": "Blue",
            "price": 80.0,
        },
        relaxed_filters=relaxed_filters or [],
    )


def faq_hit(record_id: str) -> RetrievalHit:
    return RetrievalHit(
        record_id=record_id,
        text="How do I return an item? Return within 30 days.",
        score=0.9,
        metadata={
            "faq_id": record_id,
            "question": "How do I return an item?",
            "answer": "Return within 30 days.",
            "topic": "returns",
        },
    )


def make_pipeline(
    *,
    route: Route,
    product_hits: list[RetrievalHit] | None = None,
    faq_hits: list[RetrievalHit] | None = None,
    llm: FakeLLM | None = None,
    product_query: ProductQuery | None = None,
) -> tuple[
    FashionRAGPipeline,
    FakeLLM,
    StubRouter,
    StubParser,
    StubRetriever,
    StubRetriever,
]:
    active_llm = llm or FakeLLM([])
    router = StubRouter(route)
    parser = StubParser(product_query)
    product_retriever = StubRetriever(product_hits or [])
    faq_retriever = StubRetriever(faq_hits or [])
    return (
        FashionRAGPipeline(
            router=router,
            product_parser=parser,
            product_retriever=product_retriever,
            faq_retriever=faq_retriever,
            llm=active_llm,
            top_k=5,
        ),
        active_llm,
        router,
        parser,
        product_retriever,
        faq_retriever,
    )


def test_unsupported_route_does_not_call_generator():
    pipeline, llm, *_ = make_pipeline(route=Route.UNSUPPORTED)

    response = pipeline.answer("Write Python code", [])

    assert response.route is Route.UNSUPPORTED
    assert response.hits == []
    assert "fashion products and store policies" in response.answer
    assert response.latency_ms >= 0
    assert llm.calls == []


def test_empty_product_result_is_deterministic_and_does_not_generate():
    filters = ProductFilters(
        gender=["Men"],
        article_type=["T-shirt"],
        base_colour=["Red"],
        usage=["Formal"],
        max_price=50.0,
    )
    product_query = ProductQuery(
        nature=TaskNature.TECHNICAL,
        requested_count=3,
        filters=filters,
    )
    pipeline, llm, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[],
        product_query=product_query,
    )

    response = pipeline.answer("Find a formal red men's T-shirt under $50", [])

    assert "could not find a matching product" in response.answer.lower()
    assert response.applied_filters == ProductFilters(
        article_type=["T-shirt"],
        max_price=50.0,
    )
    assert llm.calls == []


def test_empty_faq_result_is_deterministic_and_does_not_generate():
    pipeline, llm, *_ = make_pipeline(route=Route.FAQ, faq_hits=[])

    response = pipeline.answer("Do you ship to the moon?", [])

    assert "could not find a relevant store policy" in response.answer.lower()
    assert response.route is Route.FAQ
    assert response.hits == []
    assert llm.calls == []


def test_product_answer_retries_when_it_cites_unknown_id():
    llm = FakeLLM(["Try product P9999.", "Try product P0001."])
    pipeline, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("P0001")],
        llm=llm,
    )

    response = pipeline.answer("Show a blue T-shirt", [])

    assert response.answer == "Try product P0001."
    assert len(llm.calls) == 2
    assert "Use only these IDs: P0001." in llm.calls[1]


def test_faq_answer_may_cite_only_retrieved_faq_ids():
    llm = FakeLLM(["Returns are accepted within 30 days [FAQ-01]."])
    pipeline, *_ = make_pipeline(
        route=Route.FAQ,
        faq_hits=[faq_hit("FAQ-01")],
        llm=llm,
    )

    response = pipeline.answer("How do returns work?", [])

    assert response.route is Route.FAQ
    assert response.hits[0].record_id == "FAQ-01"
    assert len(llm.calls) == 1


@pytest.mark.parametrize("alias", ["p0001", "P-0001", "p_0001"])
def test_product_id_aliases_are_canonicalized(alias: str):
    llm = FakeLLM([f"Try [{alias}]."])
    pipeline, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("P0001")],
        llm=llm,
    )

    response = pipeline.answer("Show a blue T-shirt", [])

    assert response.answer == "Try [P0001]."
    assert len(llm.calls) == 1


@pytest.mark.parametrize("alias", ["faq01", "FAQ_01", "FaQ-01"])
def test_faq_id_aliases_are_canonicalized(alias: str):
    llm = FakeLLM([f"See [{alias}]."])
    pipeline, *_ = make_pipeline(
        route=Route.FAQ,
        faq_hits=[faq_hit("FAQ-01")],
        llm=llm,
    )

    response = pipeline.answer("How do returns work?", [])

    assert response.answer == "See [FAQ-01]."
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    ("route", "bad_answer", "good_answer"),
    [
        (Route.PRODUCT, "Try P00001.", "Try P0001."),
        (Route.PRODUCT, "See FAQ-01.", "Try P0001."),
        (Route.FAQ, "See FAQ-001.", "See FAQ-01."),
        (Route.FAQ, "Try P0001.", "See FAQ-01."),
    ],
)
def test_longer_and_cross_route_ids_are_rejected(
    route: Route,
    bad_answer: str,
    good_answer: str,
):
    llm = FakeLLM([bad_answer, good_answer])
    pipeline, *_ = make_pipeline(
        route=route,
        product_hits=[product_hit("P0001")],
        faq_hits=[faq_hit("FAQ-01")],
        llm=llm,
    )

    response = pipeline.answer("Help me", [])

    assert response.answer == good_answer
    assert len(llm.calls) == 2


def test_ordinary_numbers_and_embedded_tokens_are_not_false_citations():
    llm = FakeLLM(
        ["XP9999Y is internal; p = 9999; the retrieved item is [P0001]."]
    )
    pipeline, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("P0001")],
        llm=llm,
    )

    response = pipeline.answer("Show a blue T-shirt", [])

    assert "XP9999Y" in response.answer
    assert "p = 9999" in response.answer
    assert len(llm.calls) == 1


def test_citation_token_embedded_in_hyphenated_identifier_is_ignored():
    llm = FakeLLM(["SKU-P9999-X is internal; use [P0001]."])
    pipeline, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("P0001")],
        llm=llm,
    )

    response = pipeline.answer("Show a blue T-shirt", [])

    assert response.answer == "SKU-P9999-X is internal; use [P0001]."
    assert len(llm.calls) == 1


@pytest.mark.parametrize("unknown_id", ["p-9999", "P_9999", "fAq_99"])
def test_noncanonical_unknown_aliases_are_still_detected(unknown_id: str):
    llm = FakeLLM([f"Use {unknown_id}.", "Use P0001."])
    pipeline, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("P0001")],
        llm=llm,
    )

    response = pipeline.answer("Show a blue T-shirt", [])

    assert response.answer == "Use P0001."
    assert len(llm.calls) == 2


def test_second_invalid_answer_raises_typed_grounding_error():
    llm = FakeLLM(["Try P9999.", "Try p-0002."])
    pipeline, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("P0001")],
        llm=llm,
    )

    with pytest.raises(
        GroundingError,
        match="The model cited items outside the retrieved context",
    ):
        pipeline.answer("Show a blue T-shirt", [])

    assert len(llm.calls) == 2


def test_noncanonical_retrieved_id_raises_application_error():
    pipeline, llm, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("p0001")],
        llm=FakeLLM(["Try p0001."]),
    )

    with pytest.raises(ApplicationError, match="canonical product IDs"):
        pipeline.answer("Show a blue T-shirt", [])

    assert llm.calls == []


def test_applied_filters_reflect_relaxation_and_preserve_hard_filters():
    filters = ProductFilters(
        gender=["Men"],
        article_type=["T-shirt"],
        base_colour=["Red"],
        usage=["Formal"],
        season=["Fall"],
        min_price=20.0,
        max_price=100.0,
    )
    product_query = ProductQuery(
        nature=TaskNature.TECHNICAL,
        requested_count=2,
        filters=filters,
    )
    llm = FakeLLM(["Try P0001."])
    pipeline, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[
            product_hit(
                "P0001",
                relaxed_filters=["usage", "base_colour", "gender"],
            )
        ],
        llm=llm,
        product_query=product_query,
    )

    response = pipeline.answer("Find products", [])

    assert response.applied_filters == ProductFilters(
        article_type=["T-shirt"],
        season=["Fall"],
        min_price=20.0,
        max_price=100.0,
    )


def test_pipeline_passes_at_most_four_history_turns_to_collaborators():
    history = [
        ChatTurn(role="user", content=f"turn {index}") for index in range(6)
    ]
    llm = FakeLLM(["Try P0001."])
    pipeline, _, router, parser, *_ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("P0001")],
        llm=llm,
    )

    pipeline.answer("Show a blue T-shirt", history)

    assert router.calls[0][1] == history[-4:]
    assert parser.calls[0][1] == history[-4:]


def test_product_and_faq_use_expected_retrieval_and_generation_parameters():
    product_llm = FakeLLM(["Try P0001."])
    product_pipeline, _, _, _, product_retriever, _ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("P0001")],
        llm=product_llm,
    )
    faq_llm = FakeLLM(["See FAQ-01."])
    faq_pipeline, _, _, _, _, faq_retriever = make_pipeline(
        route=Route.FAQ,
        faq_hits=[faq_hit("FAQ-01")],
        llm=faq_llm,
    )

    product_pipeline.answer("Show a blue T-shirt", [])
    faq_pipeline.answer("How do returns work?", [])

    assert product_retriever.calls[0][0][2] == 5
    assert faq_retriever.calls[0][0] == ("How do returns work?",)
    assert faq_retriever.calls[0][1] == {"top_k": 3}


def test_collaborator_failure_is_exposed_as_application_error():
    class BrokenRouter:
        def route(self, query: str, history: list[ChatTurn]) -> Route:
            raise RuntimeError("provider internals")

    pipeline, *_ = make_pipeline(route=Route.UNSUPPORTED)
    pipeline.router = BrokenRouter()

    with pytest.raises(ApplicationError, match="could not process"):
        pipeline.answer("Help me", [])
