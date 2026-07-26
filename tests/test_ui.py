from __future__ import annotations

import csv
import importlib
import json
import sys

import gradio as gr
import pytest

from fashion_rag.config import MissingAPIKeyError, Settings
from fashion_rag.llm import LLMServiceError
from fashion_rag.pipeline import ApplicationError, GroundingError
from fashion_rag.schemas import (
    ChatTurn,
    PipelineResponse,
    ProductFilters,
    RetrievalHit,
    Route,
    TaskNature,
)
from fashion_rag.ui import (
    build_application,
    build_demo,
    format_retrieval_details,
    load_faqs,
    load_products,
)


def test_details_show_route_filters_ids_scores_and_relaxation():
    response = PipelineResponse(
        answer="Try P0001.",
        route=Route.PRODUCT,
        nature=TaskNature.TECHNICAL,
        applied_filters=ProductFilters(
            article_type=["T-shirt"],
            base_colour=["Blue"],
            max_price=100.0,
        ),
        hits=[
            RetrievalHit(
                record_id="P0001",
                text="Blue T-shirt",
                score=0.91,
                metadata={"price": 80},
                relaxed_filters=["usage"],
            )
        ],
        latency_ms=123.4,
    )

    markdown = format_retrieval_details(response)

    assert "**Route:** product" in markdown
    assert "**Nature:** technical" in markdown
    assert "P0001" in markdown
    assert "0.910" in markdown
    assert "usage" in markdown
    assert '"max_price":100.0' in markdown


def test_details_use_placeholders_and_escape_table_breaking_evidence():
    response = PipelineResponse(
        answer="See FAQ-01.",
        route=Route.FAQ,
        hits=[
            RetrievalHit(
                record_id="FAQ-01",
                text="Returns | exchanges\nwithin 30 days",
                score=0.5,
                metadata={},
            )
        ],
        latency_ms=2.0,
    )

    markdown = format_retrieval_details(response)

    assert "**Nature:** —" in markdown
    assert "**Applied filters:** —" in markdown
    assert r"Returns \| exchanges within 30 days" in markdown
    assert "| FAQ-01 |" in markdown


def test_loaders_parse_public_data_into_strict_models(tmp_path):
    products_path = tmp_path / "products.csv"
    with products_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "product_id",
                "name",
                "gender",
                "master_category",
                "article_type",
                "base_colour",
                "season",
                "usage",
                "price",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product_id": "P0001",
                "name": "Blue tee",
                "gender": "Unisex",
                "master_category": "Apparel",
                "article_type": "T-shirt",
                "base_colour": "Blue",
                "season": "Summer",
                "usage": "Casual",
                "price": "43.0",
            }
        )
    faqs_path = tmp_path / "faq.json"
    faqs_path.write_text(
        json.dumps(
            [
                {
                    "faq_id": "FAQ-01",
                    "question": "Can I return an item?",
                    "answer": "Yes, within 30 days.",
                    "topic": "returns",
                }
            ]
        ),
        encoding="utf-8",
    )

    products = load_products(products_path)
    faqs = load_faqs(faqs_path)

    assert products[0].product_id == "P0001"
    assert products[0].price == 43.0
    assert faqs[0].faq_id == "FAQ-01"


class StubPipeline:
    def __init__(self, result: PipelineResponse | Exception):
        self.result = result
        self.calls: list[tuple[str, list[ChatTurn]]] = []

    def answer(
        self,
        query: str,
        history: list[ChatTurn],
    ) -> PipelineResponse:
        self.calls.append((query, history))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _respond_function(demo: gr.Blocks):
    return next(
        dependency.fn
        for dependency in demo.fns.values()
        if dependency.api_name == "respond"
    )


def test_demo_submit_uses_four_turns_and_returns_explainable_result():
    response = PipelineResponse(
        answer="Try P0001.",
        route=Route.PRODUCT,
        nature=TaskNature.TECHNICAL,
        applied_filters=ProductFilters(
            article_type=["T-shirt"],
            max_price=100.0,
        ),
        hits=[
            RetrievalHit(
                record_id="P0001",
                text="Blue T-shirt",
                score=0.91,
                metadata={"price": 80},
                relaxed_filters=["usage"],
            )
        ],
        latency_ms=10.0,
    )
    pipeline = StubPipeline(response)
    demo = build_demo(pipeline)
    respond = _respond_function(demo)
    stored = [
        {"role": "user", "content": f"turn {index}"} for index in range(6)
    ]

    cleared, messages, history, details = respond(
        "Show me a blue T-shirt",
        [{"role": "assistant", "content": "Welcome"}],
        stored,
    )

    assert cleared == ""
    assert messages[-2:] == [
        {"role": "user", "content": "Show me a blue T-shirt"},
        {"role": "assistant", "content": "Try P0001."},
    ]
    assert history[-2:] == messages[-2:]
    assert pipeline.calls[0][1] == [
        ChatTurn.model_validate(turn) for turn in stored[-4:]
    ]
    assert "P0001" in details
    assert "0.910" in details
    assert "usage" in details


@pytest.mark.parametrize(
    ("error", "public_message"),
    [
        (
            MissingAPIKeyError("secret provider detail"),
            "not configured",
        ),
        (
            LLMServiceError("secret provider detail"),
            "temporarily unavailable",
        ),
        (
            GroundingError("secret provider detail"),
            "verify a grounded answer",
        ),
        (
            ApplicationError("secret provider detail"),
            "could not process",
        ),
    ],
)
def test_demo_submit_maps_typed_errors_to_safe_messages(
    error: Exception,
    public_message: str,
):
    demo = build_demo(StubPipeline(error))
    respond = _respond_function(demo)

    _, messages, _, details = respond("Help me", [], [])

    assert public_message in messages[-1]["content"]
    assert "secret provider detail" not in messages[-1]["content"]
    assert "secret provider detail" not in details


def test_demo_contains_approved_layout_and_examples():
    response = PipelineResponse(
        answer="Welcome.",
        route=Route.UNSUPPORTED,
        latency_ms=0.0,
    )

    demo = build_demo(StubPipeline(response))
    config = demo.get_config_file()
    values = [
        component.get("props", {}).get("value")
        for component in config["components"]
    ]
    labels = {
        component.get("props", {}).get("label")
        for component in config["components"]
    }

    assert config["title"] == "Fashion RAG Assistant"
    assert (
        "An explainable RAG demo for store FAQs, catalog search, "
        "and outfit recommendations."
    ) in values
    assert "What is your return policy?" in values
    assert "Show me three blue T-shirts under $100." in values
    assert "Create a summer wedding look for a man." in values
    assert "Retrieval Details" in labels
    assert any(component["type"] == "chatbot" for component in config["components"])
    assert any(component["type"] == "textbox" for component in config["components"])
    assert any(component["type"] == "state" for component in config["components"])


def test_build_application_without_key_returns_configuration_page():
    demo = build_application(
        Settings(
            together_api_key=None,
            together_model="Qwen/Qwen3.5-9B",
            embedding_model="unused",
            top_k=5,
            max_history_turns=4,
        )
    )

    assert isinstance(demo, gr.Blocks)
    values = [
        component.get("props", {}).get("value")
        for component in demo.get_config_file()["components"]
    ]
    startup_message = " ".join(str(value) for value in values)
    assert "not configured" in startup_message
    assert "TOGETHER_API_KEY" not in startup_message


def test_space_entrypoint_imports_without_api_key(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    sys.modules.pop("app", None)

    module = importlib.import_module("app")

    assert isinstance(module.demo, gr.Blocks)
