from __future__ import annotations

import numpy as np
import pytest

from fashion_rag.retriever import ProductRetriever, apply_product_filters
from fashion_rag.schemas import Product, ProductFilters
from tests.fakes import FakeEmbedder


PRODUCTS = [
    Product(
        product_id="P1",
        name="Blue Tee",
        gender="Men",
        master_category="Apparel",
        article_type="T-shirt",
        base_colour="Blue",
        season="Summer",
        usage="Casual",
        price=80.0,
    ),
    Product(
        product_id="P2",
        name="Blue Premium Tee",
        gender="Women",
        master_category="Apparel",
        article_type="T-shirt",
        base_colour="Blue",
        season="Fall",
        usage="Casual",
        price=130.0,
    ),
    Product(
        product_id="P3",
        name="Red Tee",
        gender="Men",
        master_category="Apparel",
        article_type="T-shirt",
        base_colour="Red",
        season="Summer",
        usage="Sports",
        price=70.0,
    ),
    Product(
        product_id="P4",
        name="Blue Dress",
        gender="Women",
        master_category="Apparel",
        article_type="Dress",
        base_colour="Blue",
        season="Spring",
        usage="Formal",
        price=90.0,
    ),
]


def test_upper_price_bound_works_without_lower_bound() -> None:
    result = apply_product_filters(
        PRODUCTS,
        ProductFilters(article_type=["T-shirt"], max_price=100.0),
        relaxed=set(),
    )

    assert {product.product_id for product in result} == {"P1", "P3"}


def test_article_type_and_price_are_never_relaxed() -> None:
    retriever = ProductRetriever(PRODUCTS, FakeEmbedder(), min_results=2)

    hits = retriever.search(
        "blue formal product",
        ProductFilters(
            article_type=["T-shirt"],
            base_colour=["Blue"],
            usage=["Formal"],
            max_price=100.0,
        ),
        top_k=5,
    )

    assert {hit.record_id for hit in hits} == {"P1", "P3"}
    assert all(hit.metadata["article_type"] == "T-shirt" for hit in hits)
    assert all(hit.metadata["price"] <= 100.0 for hit in hits)


def test_filter_helper_cannot_relax_article_type_directly() -> None:
    result = apply_product_filters(
        PRODUCTS,
        ProductFilters(article_type=["Dress"], usage=["Casual"]),
        relaxed={"article_type", "usage"},
    )

    assert [product.product_id for product in result] == ["P4"]


def test_soft_filters_relax_in_declared_order() -> None:
    retriever = ProductRetriever(PRODUCTS, FakeEmbedder(), min_results=2)

    hits = retriever.search(
        "blue T-shirt",
        ProductFilters(
            article_type=["T-shirt"],
            base_colour=["Blue"],
            usage=["Formal"],
            season=["Summer"],
            gender=["Men"],
            max_price=100.0,
        ),
        top_k=5,
    )

    assert hits[0].relaxed_filters == ["usage", "season", "base_colour"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_non_positive_top_k(top_k: int) -> None:
    retriever = ProductRetriever(PRODUCTS, FakeEmbedder())

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.search("blue T-shirt", ProductFilters(), top_k=top_k)


def test_retriever_rejects_duplicate_product_ids() -> None:
    duplicate = PRODUCTS[0].model_copy(update={"name": "Duplicate"})

    with pytest.raises(ValueError, match="duplicate product_id: P1"):
        ProductRetriever([*PRODUCTS, duplicate], FakeEmbedder())


def test_retriever_snapshots_the_supplied_product_list() -> None:
    supplied_products = list(PRODUCTS)
    retriever = ProductRetriever(supplied_products, FakeEmbedder())
    supplied_products.reverse()

    hits = retriever.search(
        "blue T-shirt", ProductFilters(max_price=100.0), top_k=1
    )

    assert hits[0].record_id == "P1"
    assert hits[0].metadata["name"] == "Blue Tee"


class UnnormalizedEmbedder:
    """Makes vector magnitude conflict with cosine similarity."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            if text == "query":
                vectors.append([1.0, 0.0])
            elif text.startswith("Blue Tee."):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([10.0, 1.0])
        return np.asarray(vectors, dtype="float32")


def test_retriever_normalizes_injected_embeddings_for_cosine_ranking() -> None:
    products = [PRODUCTS[0], PRODUCTS[2]]
    retriever = ProductRetriever(products, UnnormalizedEmbedder())

    hits = retriever.search("query", ProductFilters(), top_k=2)

    assert [hit.record_id for hit in hits] == ["P1", "P3"]
    assert hits[0].score == pytest.approx(1.0)
