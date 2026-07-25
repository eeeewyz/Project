from __future__ import annotations

from typing import Protocol

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from fashion_rag.schemas import Product, ProductFilters, RetrievalHit

SOFT_FILTER_ORDER = ("usage", "season", "base_colour", "gender")


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")


def apply_product_filters(
    products: list[Product],
    filters: ProductFilters,
    relaxed: set[str],
) -> list[Product]:
    """Apply hard constraints and all soft constraints not explicitly relaxed."""
    relaxed_soft_filters = relaxed.intersection(SOFT_FILTER_ORDER)

    def allowed(product: Product) -> bool:
        categorical = {
            "gender": product.gender,
            "master_category": product.master_category,
            "article_type": product.article_type,
            "base_colour": product.base_colour,
            "usage": product.usage,
            "season": product.season,
        }
        for field, actual in categorical.items():
            expected = getattr(filters, field)
            if field not in relaxed_soft_filters and expected:
                accepted = {value.casefold() for value in expected}
                if actual.casefold() not in accepted:
                    return False
        if filters.min_price is not None and product.price < filters.min_price:
            return False
        if filters.max_price is not None and product.price > filters.max_price:
            return False
        return True

    return [product for product in products if allowed(product)]


def _normalized_embeddings(
    values: np.ndarray,
    *,
    expected_rows: int,
) -> np.ndarray:
    array = np.asarray(values, dtype="float32")
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] == 0:
        raise ValueError(
            "embedder output must be a two-dimensional matrix with one row "
            "per input text"
        )
    if not np.isfinite(array).all():
        raise ValueError("embedder output must contain only finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.ascontiguousarray(array / np.maximum(norms, 1e-12))


class ProductRetriever:
    def __init__(
        self,
        products: list[Product],
        embedder: Embedder,
        min_results: int = 5,
    ):
        self.products = [product.model_copy(deep=True) for product in products]
        product_ids = [product.product_id for product in self.products]
        duplicates = sorted(
            product_id
            for product_id in set(product_ids)
            if product_ids.count(product_id) > 1
        )
        if duplicates:
            raise ValueError(f"duplicate product_id: {duplicates[0]}")

        self.embedder = embedder
        self.embeddings = _normalized_embeddings(
            embedder.encode([product.search_text for product in self.products]),
            expected_rows=len(self.products),
        )
        self.min_results = min_results
        self._positions = {
            product.product_id: position
            for position, product in enumerate(self.products)
        }

    def search(
        self,
        query: str,
        filters: ProductFilters,
        top_k: int,
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        relaxed: list[str] = []
        candidates = apply_product_filters(self.products, filters, set())
        for field in SOFT_FILTER_ORDER:
            if len(candidates) >= self.min_results:
                break
            if getattr(filters, field):
                relaxed.append(field)
                candidates = apply_product_filters(
                    self.products,
                    filters,
                    set(relaxed),
                )

        if not candidates:
            return []

        positions = [
            self._positions[product.product_id] for product in candidates
        ]
        matrix = np.ascontiguousarray(self.embeddings[positions])
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        query_vector = _normalized_embeddings(
            self.embedder.encode([query]),
            expected_rows=1,
        )
        scores, local_positions = index.search(
            query_vector,
            min(top_k, len(candidates)),
        )

        return [
            RetrievalHit(
                record_id=candidates[position].product_id,
                text=candidates[position].search_text,
                score=float(score),
                metadata=candidates[position].model_dump(),
                relaxed_filters=relaxed.copy(),
            )
            for score, position in zip(
                scores[0],
                local_positions[0],
                strict=True,
            )
            if position >= 0
        ]
