from fashion_rag.llm import StructuredGenerator
from fashion_rag.router import QueryRouter, bound_history
from fashion_rag.schemas import ChatTurn, Route
from tests.fakes import FakeLLM
import pytest


def test_history_is_bounded_to_four_latest_turns():
    history = [ChatTurn(role="user", content=str(i)) for i in range(7)]

    assert [turn.content for turn in bound_history(history, 4)] == ["3", "4", "5", "6"]


def test_history_bound_of_zero_returns_no_turns():
    history = [ChatTurn(role="user", content=str(i)) for i in range(3)]

    assert bound_history(history, 0) == []


def test_negative_history_bound_is_rejected():
    with pytest.raises(ValueError, match="max_turns must be non-negative"):
        bound_history([], -1)


def test_router_rejects_negative_history_configuration():
    structured = StructuredGenerator(FakeLLM([]))

    with pytest.raises(ValueError, match="max_history_turns must be non-negative"):
        QueryRouter(structured, max_history_turns=-1)


def test_router_returns_validated_route():
    router = QueryRouter(StructuredGenerator(FakeLLM(['{"route":"faq"}'])))

    assert router.route("How do returns work?", []) is Route.FAQ


def test_router_prompt_contains_supported_scope():
    llm = FakeLLM(['{"route":"unsupported"}'])

    QueryRouter(StructuredGenerator(llm)).route("Write Python code", [])

    assert '"faq", "product", or "unsupported"' in llm.calls[0]
