from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.metrics import (
    constraint_compliance,
    field_accuracy,
    groundedness,
    hit_at_k,
    route_accuracy,
)
from evaluation.run_evaluation import (
    aggregate_results,
    baseline_product_search,
    evaluate_cases,
    load_dataset,
    main,
    write_results,
)
from fashion_rag.retriever import ProductRetriever
from fashion_rag.schemas import (
    ChatTurn,
    PipelineResponse,
    Product,
    ProductFilters,
    ProductQuery,
    RetrievalHit,
    Route,
    TaskNature,
)
from tests.fakes import FakeEmbedder


DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "evaluation" / "eval_dataset.json"
)
CANONICAL_CASE_ID = re.compile(r"(?:F|P|U)-\d{2}")
CANONICAL_PRODUCT_ID = re.compile(r"P\d{4}")
CANONICAL_FAQ_ID = re.compile(r"FAQ-\d{2}")


def _hit(
    record_id: str,
    *,
    price: float = 80.0,
    base_colour: str = "Blue",
    relaxed_filters: list[str] | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        record_id=record_id,
        text="",
        score=1.0,
        metadata={
            "article_type": "T-shirt",
            "base_colour": base_colour,
            "price": price,
        },
        relaxed_filters=relaxed_filters or [],
    )


def test_hit_at_k() -> None:
    assert hit_at_k(["P2"], ["P1", "P2", "P3"], 2) == 1.0
    assert hit_at_k(["P4"], ["P1", "P2", "P3"], 3) == 0.0


def test_field_accuracy_counts_missing_and_extra_values() -> None:
    expected = {"base_colour": ["Blue"], "max_price": 100}
    actual = {"base_colour": ["Blue"], "max_price": 120}
    assert field_accuracy(expected, actual) == 0.5


def test_field_accuracy_casefolds_categorical_lists() -> None:
    expected = {"base_colour": ["Blue"], "article_type": ["T-shirt"]}
    actual = {"base_colour": ["BLUE"], "article_type": ["t-SHIRT"]}
    assert field_accuracy(expected, actual) == 1.0


def test_groundedness_rejects_unknown_product_id() -> None:
    assert groundedness("Use P0001 and P9999.", {"P0001"}) == 0.5


def test_groundedness_accepts_no_citations_only_without_retrieved_ids() -> None:
    assert groundedness("No matching products were found.", set()) == 1.0
    assert groundedness("Here is a recommendation.", {"P0001"}) == 0.0


def test_groundedness_extracts_only_canonical_complete_ids() -> None:
    answer = "Use P0001. Ignore P00001, p0001, and FAQ-001."
    assert groundedness(answer, {"P0001"}) == 1.0


def test_constraint_compliance_checks_upper_only_price() -> None:
    hits = [
        _hit("P1", price=80.0),
        _hit("P2", price=120.0),
    ]
    filters = ProductFilters(article_type=["T-shirt"], max_price=100.0)
    assert constraint_compliance(hits, filters) == 0.5


def test_constraint_compliance_ignores_filters_relaxed_by_retrieval() -> None:
    hits = [
        _hit(
            "P0001",
            base_colour="Red",
            relaxed_filters=["base_colour"],
        )
    ]
    filters = ProductFilters(
        article_type=["T-shirt"],
        base_colour=["Blue"],
    )
    assert constraint_compliance(hits, filters) == 1.0


def test_constraint_compliance_distinguishes_expected_empty_result() -> None:
    filters = ProductFilters(
        article_type=["T-shirt"],
        base_colour=["Blue"],
        max_price=10.0,
    )
    assert constraint_compliance([], filters, expected_ids=[]) == 1.0
    assert constraint_compliance([], filters, expected_ids=["P0001"]) == 0.0


def test_route_accuracy_uses_expected_and_actual_routes() -> None:
    rows = [
        {"expected_route": "faq", "actual_route": "faq"},
        {"expected_route": "product", "actual_route": "unsupported"},
    ]
    assert route_accuracy(rows) == 0.5


def test_eval_dataset_has_exact_inventory_and_canonical_ids() -> None:
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    assert len(rows) == 30
    assert [row["case_id"] for row in rows] == [
        *(f"F-{index:02d}" for index in range(1, 9)),
        *(f"P-{index:02d}" for index in range(1, 18)),
        *(f"U-{index:02d}" for index in range(1, 6)),
    ]
    assert all(CANONICAL_CASE_ID.fullmatch(row["case_id"]) for row in rows)
    assert all(
        CANONICAL_PRODUCT_ID.fullmatch(product_id)
        for row in rows
        for product_id in row["expected_product_ids"]
    )
    assert all(
        CANONICAL_FAQ_ID.fullmatch(faq_id)
        for row in rows
        for faq_id in row["expected_faq_ids"]
    )

    p17 = next(row for row in rows if row["case_id"] == "P-17")
    assert p17["expected_product_ids"] == []
    assert p17["expected_filters"] == {
        "article_type": ["T-shirt"],
        "base_colour": ["Blue"],
        "max_price": 10,
    }
    assert (
        sum(
            row["expected_route"] == "product"
            and bool(row["expected_product_ids"])
            for row in rows
        )
        == 11
    )
    assert sum(row["expected_route"] == "faq" for row in rows) == 8


def test_load_dataset_validates_the_published_case_set() -> None:
    rows = load_dataset(DATASET_PATH)

    assert len(rows) == 30
    assert rows[0]["case_id"] == "F-01"
    assert rows[-1]["case_id"] == "U-05"


def test_load_dataset_rejects_noncanonical_evidence_ids(
    tmp_path: Path,
) -> None:
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    rows[8]["expected_product_ids"] = ["P-0001"]
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical product ID"):
        load_dataset(invalid_path)


def test_baseline_product_search_does_not_apply_metadata_filters() -> None:
    products = [
        Product(
            product_id="P0001",
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
            product_id="P0002",
            name="Red Dress",
            gender="Women",
            master_category="Apparel",
            article_type="Dress",
            base_colour="Red",
            season="Spring",
            usage="Formal",
            price=120.0,
        ),
    ]
    retriever = ProductRetriever(products, FakeEmbedder())

    hits = baseline_product_search(retriever, "blue T-shirt", top_k=2)

    assert [hit.record_id for hit in hits] == ["P0001", "P0002"]


def test_aggregate_results_uses_only_applicable_measurements() -> None:
    per_case = [
        {
            "case_id": "F-01",
            "expected_route": "faq",
            "actual_route": "faq",
            "baseline_hit_at_5": None,
            "baseline_constraint_compliance": None,
            "metadata_field_accuracy": None,
            "improved_hit_at_5": None,
            "faq_hit_at_3": 1.0,
            "improved_constraint_compliance": None,
            "groundedness": 1.0,
            "latency_ms": 10.0,
        },
        {
            "case_id": "P-01",
            "expected_route": "product",
            "actual_route": "unsupported",
            "baseline_hit_at_5": 0.0,
            "baseline_constraint_compliance": 0.0,
            "metadata_field_accuracy": 0.5,
            "improved_hit_at_5": 1.0,
            "faq_hit_at_3": None,
            "improved_constraint_compliance": 1.0,
            "groundedness": 0.5,
            "latency_ms": 30.0,
        },
    ]

    results = aggregate_results(
        per_case,
        evaluated_at="2026-07-25T12:00:00Z",
    )

    assert results == {
        "evaluated_at": "2026-07-25T12:00:00Z",
        "case_count": 2,
        "baseline": {
            "hit_at_5": 0.0,
            "constraint_compliance": 0.0,
        },
        "metric_case_counts": {
            "product_hit_at_5": 1,
            "faq_hit_at_3": 1,
        },
        "improved": {
            "route_accuracy": 0.5,
            "metadata_field_accuracy": 0.5,
            "hit_at_5": 1.0,
            "faq_hit_at_3": 1.0,
            "constraint_compliance": 1.0,
            "groundedness": 0.75,
            "mean_latency_ms": 20.0,
        },
    }


def test_aggregate_results_rejects_different_product_hit_populations() -> None:
    per_case = [
        {
            "case_id": "P-01",
            "expected_route": "product",
            "actual_route": "product",
            "baseline_hit_at_5": 1.0,
            "improved_hit_at_5": None,
        },
        {
            "case_id": "F-01",
            "expected_route": "faq",
            "actual_route": "faq",
            "baseline_hit_at_5": None,
            "improved_hit_at_5": 1.0,
        },
    ]

    with pytest.raises(ValueError, match="same product cases"):
        aggregate_results(
            per_case,
            evaluated_at="2026-07-25T12:00:00Z",
        )


def test_write_results_uses_measured_values(tmp_path: Path) -> None:
    results = {
        "evaluated_at": "2026-07-25T12:00:00Z",
        "case_count": 30,
        "baseline": {
            "hit_at_5": 0.4,
            "constraint_compliance": 0.6,
        },
        "metric_case_counts": {
            "product_hit_at_5": 11,
            "faq_hit_at_3": 8,
        },
        "improved": {
            "route_accuracy": 0.9,
            "metadata_field_accuracy": 0.8,
            "hit_at_5": 0.75,
            "faq_hit_at_3": 0.875,
            "constraint_compliance": 0.95,
            "groundedness": 1.0,
            "mean_latency_ms": 123.4,
        },
    }
    json_path = tmp_path / "results.json"
    markdown_path = tmp_path / "results.md"

    write_results(results, json_path, markdown_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == results
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| Product Hit@5 (n=11) | 0.400 | 0.750 |" in markdown
    assert "| FAQ Hit@3 (n=8) | — | 0.875 |" in markdown
    assert "| Constraint compliance | 0.600 | 0.950 |" in markdown
    assert "| Mean latency (ms) | — | 123.4 |" in markdown


def test_main_without_api_key_does_not_create_result_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    json_path = tmp_path / "results.json"
    markdown_path = tmp_path / "results.md"

    exit_code = main(
        [
            "--dataset",
            str(DATASET_PATH),
            "--output-json",
            str(json_path),
            "--output-markdown",
            str(markdown_path),
        ]
    )

    assert exit_code == 2
    assert not json_path.exists()
    assert not markdown_path.exists()


def test_documented_direct_script_command_runs_without_editable_install(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("TOGETHER_API_KEY", None)
    json_path = tmp_path / "results.json"
    markdown_path = tmp_path / "results.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(DATASET_PATH.parent / "run_evaluation.py"),
            "--dataset",
            str(DATASET_PATH),
            "--output-json",
            str(json_path),
            "--output-markdown",
            str(markdown_path),
        ],
        cwd=DATASET_PATH.parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "TOGETHER_API_KEY is not configured" in completed.stderr
    assert not json_path.exists()
    assert not markdown_path.exists()


def test_evaluate_cases_scores_expected_empty_p17_as_compliant() -> None:
    products = [
        Product(
            product_id="P0001",
            name="Blue Tee",
            gender="Men",
            master_category="Apparel",
            article_type="T-shirt",
            base_colour="Blue",
            season="Summer",
            usage="Casual",
            price=80.0,
        )
    ]
    retriever = ProductRetriever(products, FakeEmbedder())
    parsed = ProductQuery(
        nature=TaskNature.TECHNICAL,
        requested_count=3,
        filters=ProductFilters(
            article_type=["T-shirt"],
            base_colour=["Blue"],
            max_price=10.0,
        ),
    )

    class Router:
        def route(self, query: str, history: list[ChatTurn]) -> Route:
            return Route.PRODUCT

    class Parser:
        def parse(
            self,
            query: str,
            history: list[ChatTurn],
        ) -> ProductQuery:
            return parsed

    class Pipeline:
        router = Router()
        product_parser = Parser()
        product_retriever = retriever

        def answer(
            self,
            query: str,
            history: list[ChatTurn],
        ) -> PipelineResponse:
            return PipelineResponse(
                answer="No matching products were found.",
                route=Route.PRODUCT,
                nature=TaskNature.TECHNICAL,
                applied_filters=ProductFilters(
                    article_type=["T-shirt"],
                    max_price=10.0,
                ),
                hits=[],
                latency_ms=5.0,
            )

    rows = [
        {
            "case_id": "P-17",
            "query": "Find a blue T-shirt under $10.",
            "history": [],
            "expected_route": "product",
            "expected_nature": "technical",
            "expected_filters": {
                "article_type": ["T-shirt"],
                "base_colour": ["Blue"],
                "max_price": 10,
            },
            "expected_faq_ids": [],
            "expected_product_ids": [],
        }
    ]

    result = evaluate_cases(rows, Pipeline())

    assert result[0]["baseline_constraint_compliance"] == 0.0
    assert result[0]["improved_constraint_compliance"] == 1.0
    assert result[0]["groundedness"] == 1.0
    assert result[0]["baseline_returned_ids"] == ["P0001"]
    assert result[0]["returned_ids"] == []
    assert result[0]["answer"] == "No matching products were found."


def test_evaluate_cases_keeps_product_and_faq_hit_populations_separate() -> None:
    product = Product(
        product_id="P0001",
        name="Blue Tee",
        gender="Men",
        master_category="Apparel",
        article_type="T-shirt",
        base_colour="Blue",
        season="Summer",
        usage="Casual",
        price=80.0,
    )
    retriever = ProductRetriever([product], FakeEmbedder())
    parsed = ProductQuery(
        nature=TaskNature.TECHNICAL,
        requested_count=3,
        filters=ProductFilters(
            article_type=["T-shirt"],
            base_colour=["Blue"],
        ),
    )
    product_evidence = RetrievalHit(
        record_id="P0001",
        text=product.search_text,
        score=1.0,
        metadata=product.model_dump(),
    )
    faq_evidence = RetrievalHit(
        record_id="FAQ-01",
        text="Return an item within 30 days.",
        score=1.0,
        metadata={
            "faq_id": "FAQ-01",
            "question": "How do I return an item?",
            "answer": "Return an item within 30 days.",
            "topic": "returns",
        },
    )

    class Parser:
        def parse(
            self,
            query: str,
            history: list[ChatTurn],
        ) -> ProductQuery:
            return parsed

    class Pipeline:
        product_parser = Parser()
        product_retriever = retriever

        def answer(
            self,
            query: str,
            history: list[ChatTurn],
        ) -> PipelineResponse:
            if query.startswith("How"):
                return PipelineResponse(
                    answer="Return within 30 days [FAQ-01].",
                    route=Route.FAQ,
                    hits=[faq_evidence],
                    latency_ms=5.0,
                )
            return PipelineResponse(
                answer="Try P0001.",
                route=Route.PRODUCT,
                nature=TaskNature.TECHNICAL,
                applied_filters=parsed.filters,
                hits=[product_evidence],
                latency_ms=5.0,
            )

    rows = [
        {
            "case_id": "F-01",
            "query": "How do I return an item?",
            "history": [],
            "expected_route": "faq",
            "expected_nature": None,
            "expected_filters": {},
            "expected_faq_ids": ["FAQ-01"],
            "expected_product_ids": [],
        },
        {
            "case_id": "P-01",
            "query": "Show me a blue T-shirt.",
            "history": [],
            "expected_route": "product",
            "expected_nature": "technical",
            "expected_filters": {
                "article_type": ["T-shirt"],
                "base_colour": ["Blue"],
            },
            "expected_faq_ids": [],
            "expected_product_ids": ["P0001"],
        },
    ]

    results = evaluate_cases(rows, Pipeline())

    faq_result, product_result = results
    assert faq_result["improved_hit_at_5"] is None
    assert faq_result["faq_hit_at_3"] == 1.0
    assert product_result["improved_hit_at_5"] == 1.0
    assert product_result["faq_hit_at_3"] is None
