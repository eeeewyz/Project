from __future__ import annotations

import numpy as np
import pytest

from fashion_rag.retriever import FAQRetriever
from fashion_rag.schemas import FAQEntry
from tests.fakes import FakeEmbedder


FAQS = [
    FAQEntry(
        faq_id="FAQ-01",
        question="How do I return an item?",
        answer="Return within 30 days.",
        topic="returns",
    ),
    FAQEntry(
        faq_id="FAQ-02",
        question="How fast is shipping?",
        answer="Three to five days.",
        topic="shipping",
    ),
    FAQEntry(
        faq_id="FAQ-03",
        question="How do I choose my size?",
        answer="Use the size guide before ordering.",
        topic="sizing",
    ),
    FAQEntry(
        faq_id="FAQ-04",
        question="Can I exchange an item?",
        answer="Exchanges follow the return policy.",
        topic="returns",
    ),
]


def test_return_question_retrieves_return_faq_first() -> None:
    hits = FAQRetriever(FAQS[:2], FakeEmbedder()).search(
        "I need to return it", top_k=2
    )

    assert hits[0].record_id == "FAQ-01"
    assert hits[0].metadata["topic"] == "returns"


def test_search_returns_exact_top_three_with_faq_id_evidence() -> None:
    hits = FAQRetriever(FAQS, PreciseFAQEmbedder()).search("return policy")

    assert len(hits) == 3
    assert [hit.record_id for hit in hits] == ["FAQ-01", "FAQ-04", "FAQ-02"]
    assert all(hit.record_id == hit.metadata["faq_id"] for hit in hits)
    assert all(hit.relaxed_filters == [] for hit in hits)


class PreciseFAQEmbedder:
    """Offline vectors with distinct relevance scores for FAQ ranking tests."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if lowered == "return policy":
                vectors.append([1.0, 0.0])
            elif "how do i return" in lowered:
                vectors.append([1.0, 0.0])
            elif "exchange" in lowered:
                vectors.append([0.8, 0.2])
            elif "shipping" in lowered:
                vectors.append([0.2, 0.8])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype="float32")


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_search_rejects_invalid_top_k(top_k: int | bool) -> None:
    retriever = FAQRetriever(FAQS, FakeEmbedder())

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.search("return", top_k=top_k)


def test_retriever_rejects_duplicate_faq_ids() -> None:
    duplicate = FAQS[0].model_copy(update={"question": "Another return question"})

    with pytest.raises(ValueError, match="duplicate faq_id: FAQ-01"):
        FAQRetriever([*FAQS, duplicate], FakeEmbedder())


def test_retriever_snapshots_the_supplied_faq_list() -> None:
    supplied_faqs = [faq.model_copy(deep=True) for faq in FAQS]
    retriever = FAQRetriever(supplied_faqs, FakeEmbedder())
    supplied_faqs[0].question = "Mutated question"
    supplied_faqs.reverse()

    hits = retriever.search("return", top_k=1)

    assert hits[0].record_id == "FAQ-01"
    assert hits[0].metadata["question"] == "How do I return an item?"


class NonFiniteFAQEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.full((len(texts), 2), np.nan, dtype="float32")


def test_retriever_rejects_non_finite_injected_embeddings() -> None:
    with pytest.raises(ValueError, match="finite values"):
        FAQRetriever(FAQS, NonFiniteFAQEmbedder())


class UnnormalizedFAQEmbedder:
    """Makes vector magnitude conflict with cosine similarity."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            if text == "query":
                vectors.append([1.0, 0.0])
            elif text.startswith("How do I return"):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([10.0, 1.0])
        return np.asarray(vectors, dtype="float32")


def test_retriever_normalizes_injected_embeddings_for_cosine_ranking() -> None:
    retriever = FAQRetriever(FAQS[:2], UnnormalizedFAQEmbedder())

    hits = retriever.search("query", top_k=2)

    assert [hit.record_id for hit in hits] == ["FAQ-01", "FAQ-02"]
    assert hits[0].score == pytest.approx(1.0)
