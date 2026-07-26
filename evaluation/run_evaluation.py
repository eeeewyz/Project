"""Run a measured baseline-vs-improved evaluation.

This module deliberately writes result artifacts only after all 30 live cases
complete. A Together API key is required for routing, parsing, and grounded
answer generation; deterministic metric tests do not use that provider.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluation.metrics import (  # noqa: E402
    constraint_compliance,
    field_accuracy,
    groundedness,
    hit_at_k,
    route_accuracy,
)
from fashion_rag.config import MissingAPIKeyError, Settings  # noqa: E402
from fashion_rag.retriever import ProductRetriever  # noqa: E402
from fashion_rag.schemas import (  # noqa: E402
    ChatTurn,
    ProductFilters,
    RetrievalHit,
    Route,
)


_CASE_IDS = [
    *(f"F-{index:02d}" for index in range(1, 9)),
    *(f"P-{index:02d}" for index in range(1, 18)),
    *(f"U-{index:02d}" for index in range(1, 6)),
]
_REQUIRED_KEYS = {
    "case_id",
    "query",
    "history",
    "expected_route",
    "expected_nature",
    "expected_filters",
    "expected_faq_ids",
    "expected_product_ids",
}
_PRODUCT_ID = re.compile(r"P\d{4}")
_FAQ_ID = re.compile(r"FAQ-\d{2}")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the fixed, public 30-case evaluation inventory."""

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("evaluation dataset must be a JSON array")
    if len(rows) != 30:
        raise ValueError("evaluation dataset must contain exactly 30 cases")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("each evaluation case must be a JSON object")
    if [row.get("case_id") for row in rows] != _CASE_IDS:
        raise ValueError("evaluation case IDs must match the fixed inventory")

    for row in rows:
        if set(row) != _REQUIRED_KEYS:
            raise ValueError(
                f"{row['case_id']} must contain the required case fields"
            )
        if row["expected_route"] not in {route.value for route in Route}:
            raise ValueError(f"{row['case_id']} has an invalid expected route")
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError(f"{row['case_id']} must have a non-empty query")
        if not isinstance(row["history"], list):
            raise ValueError(f"{row['case_id']} history must be a list")
        for turn in row["history"]:
            ChatTurn.model_validate(turn)
        if not isinstance(row["expected_filters"], dict):
            raise ValueError(f"{row['case_id']} filters must be an object")
        if not all(
            isinstance(record_id, str)
            and _PRODUCT_ID.fullmatch(record_id)
            for record_id in row["expected_product_ids"]
        ):
            raise ValueError(
                f"{row['case_id']} contains a non-canonical product ID"
            )
        if not all(
            isinstance(record_id, str) and _FAQ_ID.fullmatch(record_id)
            for record_id in row["expected_faq_ids"]
        ):
            raise ValueError(
                f"{row['case_id']} contains a non-canonical FAQ ID"
            )
    return rows


def baseline_product_search(
    retriever: ProductRetriever,
    query: str,
    *,
    top_k: int = 5,
) -> list[RetrievalHit]:
    """Search every product by embedding similarity, without metadata filters."""

    return retriever.search(query, ProductFilters(), top_k=top_k)


def _expected_ids(row: Mapping[str, Any]) -> list[str]:
    return [*row["expected_faq_ids"], *row["expected_product_ids"]]


def _empty_result_label(row: Mapping[str, Any]) -> Sequence[str] | None:
    """Return an explicit empty label only for the known no-match case."""

    if row["case_id"] == "P-17":
        return []
    expected = row["expected_product_ids"]
    return expected or None


def evaluate_cases(
    rows: Sequence[Mapping[str, Any]],
    pipeline: Any,
) -> list[dict[str, Any]]:
    """Evaluate each labeled case with baseline retrieval and the full system."""

    per_case: list[dict[str, Any]] = []
    for row in rows:
        query = row["query"]
        history = [
            ChatTurn.model_validate(turn) for turn in row["history"]
        ]
        expected_route = row["expected_route"]
        expected_filters = ProductFilters.model_validate(
            row["expected_filters"]
        )

        metadata_score: float | None = None
        baseline_hit_score: float | None = None
        baseline_compliance: float | None = None
        baseline_returned_ids: list[str] | None = None

        if expected_route == Route.PRODUCT.value:
            product_query = pipeline.product_parser.parse(query, history)
            metadata_score = field_accuracy(
                row["expected_filters"],
                product_query.filters,
            )
            baseline_hits = baseline_product_search(
                pipeline.product_retriever,
                query,
                top_k=5,
            )
            baseline_returned_ids = [
                hit.record_id for hit in baseline_hits
            ]
            product_ids = row["expected_product_ids"]
            if product_ids:
                baseline_hit_score = hit_at_k(
                    product_ids,
                    baseline_returned_ids,
                    5,
                )
            baseline_compliance = constraint_compliance(
                baseline_hits,
                expected_filters,
                expected_ids=_empty_result_label(row),
            )

        response = pipeline.answer(query, history)
        returned_ids = [hit.record_id for hit in response.hits]
        expected_ids = _expected_ids(row)
        improved_hit_score = (
            hit_at_k(
                row["expected_product_ids"],
                returned_ids,
                5,
            )
            if (
                expected_route == Route.PRODUCT.value
                and row["expected_product_ids"]
            )
            else None
        )
        faq_hit_score = (
            hit_at_k(
                row["expected_faq_ids"],
                returned_ids,
                3,
            )
            if (
                expected_route == Route.FAQ.value
                and row["expected_faq_ids"]
            )
            else None
        )
        improved_compliance = (
            constraint_compliance(
                response.hits,
                expected_filters,
                expected_ids=_empty_result_label(row),
            )
            if expected_route == Route.PRODUCT.value
            else None
        )
        per_case.append(
            {
                "case_id": row["case_id"],
                "expected_route": expected_route,
                "actual_route": response.route.value,
                "expected_nature": row["expected_nature"],
                "actual_nature": (
                    response.nature.value
                    if response.nature is not None
                    else None
                ),
                "expected_ids": expected_ids,
                "baseline_returned_ids": baseline_returned_ids,
                "returned_ids": returned_ids,
                "answer": response.answer,
                "baseline_hit_at_5": baseline_hit_score,
                "baseline_constraint_compliance": baseline_compliance,
                "metadata_field_accuracy": metadata_score,
                "improved_hit_at_5": improved_hit_score,
                "faq_hit_at_3": faq_hit_score,
                "improved_constraint_compliance": improved_compliance,
                "groundedness": groundedness(
                    response.answer,
                    returned_ids,
                ),
                "latency_ms": response.latency_ms,
            }
        )
    return per_case


def _applicable_mean(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> float:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        raise ValueError(f"no applicable measurements for {key}")
    return float(fmean(values))


def aggregate_results(
    per_case: Sequence[Mapping[str, Any]],
    *,
    evaluated_at: str,
) -> dict[str, Any]:
    """Aggregate measured per-case values into the published result schema."""

    if not per_case:
        raise ValueError("cannot aggregate an empty evaluation")
    baseline_product_cases = {
        row["case_id"]
        for row in per_case
        if row.get("baseline_hit_at_5") is not None
    }
    improved_product_cases = {
        row["case_id"]
        for row in per_case
        if row.get("improved_hit_at_5") is not None
    }
    if baseline_product_cases != improved_product_cases:
        raise ValueError(
            "baseline and improved Hit@5 must use the same product cases"
        )
    faq_cases = {
        row["case_id"]
        for row in per_case
        if row.get("faq_hit_at_3") is not None
    }
    return {
        "evaluated_at": evaluated_at,
        "case_count": len(per_case),
        "baseline": {
            "hit_at_5": _applicable_mean(
                per_case,
                "baseline_hit_at_5",
            ),
            "constraint_compliance": _applicable_mean(
                per_case,
                "baseline_constraint_compliance",
            ),
        },
        "metric_case_counts": {
            "product_hit_at_5": len(baseline_product_cases),
            "faq_hit_at_3": len(faq_cases),
        },
        "improved": {
            "route_accuracy": route_accuracy(per_case),
            "metadata_field_accuracy": _applicable_mean(
                per_case,
                "metadata_field_accuracy",
            ),
            "hit_at_5": _applicable_mean(
                per_case,
                "improved_hit_at_5",
            ),
            "faq_hit_at_3": _applicable_mean(
                per_case,
                "faq_hit_at_3",
            ),
            "constraint_compliance": _applicable_mean(
                per_case,
                "improved_constraint_compliance",
            ),
            "groundedness": _applicable_mean(
                per_case,
                "groundedness",
            ),
            "mean_latency_ms": _applicable_mean(
                per_case,
                "latency_ms",
            ),
        },
    }


def _render_markdown(results: Mapping[str, Any]) -> str:
    baseline = results["baseline"]
    improved = results["improved"]
    counts = results["metric_case_counts"]
    return "\n".join(
        [
            "# Fashion RAG Evaluation Results",
            "",
            f"- Evaluated at: {results['evaluated_at']}",
            f"- Cases: {results['case_count']}",
            "",
            "| Metric | Baseline | Improved |",
            "|---|---:|---:|",
            (
                "| Route accuracy | — "
                f"| {improved['route_accuracy']:.3f} |"
            ),
            (
                "| Metadata field accuracy | — "
                f"| {improved['metadata_field_accuracy']:.3f} |"
            ),
            (
                "| Product Hit@5 "
                f"(n={counts['product_hit_at_5']}) "
                f"| {baseline['hit_at_5']:.3f} "
                f"| {improved['hit_at_5']:.3f} |"
            ),
            (
                f"| FAQ Hit@3 (n={counts['faq_hit_at_3']}) | — "
                f"| {improved['faq_hit_at_3']:.3f} |"
            ),
            (
                "| Constraint compliance "
                f"| {baseline['constraint_compliance']:.3f} "
                f"| {improved['constraint_compliance']:.3f} |"
            ),
            (
                "| Groundedness | — "
                f"| {improved['groundedness']:.3f} |"
            ),
            (
                "| Mean latency (ms) | — "
                f"| {improved['mean_latency_ms']:.1f} |"
            ),
            "",
        ]
    )


def write_results(
    results: Mapping[str, Any],
    output_json: str | Path,
    output_markdown: str | Path,
) -> None:
    """Write JSON and Markdown views of the same measured result object."""

    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _render_markdown(results),
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and improved Fashion RAG retrieval."
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/eval_dataset.json",
    )
    parser.add_argument(
        "--output-json",
        default="evaluation/results.json",
    )
    parser.add_argument(
        "--output-markdown",
        default="evaluation/results.md",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run all live cases and write results only after successful completion."""

    args = _parse_args(argv)
    rows = load_dataset(args.dataset)
    settings = Settings.from_env()
    try:
        settings.require_api_key()
        from fashion_rag.ui import create_pipeline

        pipeline = create_pipeline(settings)
    except MissingAPIKeyError as error:
        print(str(error), file=sys.stderr)
        return 2

    per_case = evaluate_cases(rows, pipeline)
    for row in per_case:
        print(json.dumps(row, sort_keys=True))
    results = aggregate_results(
        per_case,
        evaluated_at=datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    )
    write_results(results, args.output_json, args.output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
