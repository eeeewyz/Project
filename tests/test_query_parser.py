import pytest

from fashion_rag.llm import StructuredGenerator, StructuredOutputError
from fashion_rag.query_parser import ProductQueryParser
from fashion_rag.schemas import ChatTurn, TaskNature
from tests.fakes import FakeLLM


def test_parser_keeps_upper_price_without_minimum():
    """Catch a parser that invents a lower price bound for an "under" request."""

    llm = FakeLLM(
        [
            '{"nature":"technical","requested_count":3,"filters":'
            '{"article_type":["T-shirt"],"base_colour":["Blue"],'
            '"max_price":100}}'
        ]
    )

    parsed = ProductQueryParser(StructuredGenerator(llm)).parse(
        "Show three blue T-shirts under $100", []
    )

    assert parsed.nature is TaskNature.TECHNICAL
    assert parsed.filters.min_price is None
    assert parsed.filters.max_price == 100


def test_parser_uses_null_for_unspecified_catalog_filters():
    """Catch accepting the old Any sentinel instead of an absent filter."""

    llm = FakeLLM(
        [
            '{"nature":"creative","requested_count":3,"filters":'
            '{"usage":["Formal"],"season":null}}'
        ]
    )

    parsed = ProductQueryParser(StructuredGenerator(llm)).parse(
        "Create a formal look", []
    )

    assert parsed.filters.season is None
    assert parsed.filters.gender is None


def test_parser_rejects_empty_categorical_filter_lists():
    """Catch an LLM response that encodes an absent filter as an empty list."""

    invalid_response = (
        '{"nature":"technical","requested_count":1,"filters":{"season":[]}}'
    )
    llm = FakeLLM([invalid_response, invalid_response])

    with pytest.raises(StructuredOutputError):
        ProductQueryParser(StructuredGenerator(llm)).parse("Show a shirt", [])


def test_parser_bounds_history_and_explains_query_nature_to_the_generator():
    """Catch leaking old turns or sending an ambiguous nature classification prompt."""

    llm = FakeLLM(
        ['{"nature":"technical","requested_count":1,"filters":{}}']
    )
    history = [ChatTurn(role="user", content=f"turn-{index}") for index in range(6)]

    ProductQueryParser(StructuredGenerator(llm)).parse("Show a shirt", history)

    prompt = llm.calls[0]
    assert "turn-0" not in prompt
    assert "turn-1" not in prompt
    assert "turn-2" in prompt
    assert "turn-5" in prompt
    assert "technical: explicit catalog constraints" in prompt
    assert "creative: open-ended outfit or style recommendation" in prompt
