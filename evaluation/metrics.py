"""Pure metrics used by the offline and live evaluation runner."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fashion_rag.retriever import SOFT_FILTER_ORDER
from fashion_rag.schemas import ProductFilters, RetrievalHit


_CANONICAL_ID = re.compile(
    r"(?<![A-Za-z0-9_-])(?:P\d{4}|FAQ-\d{2})(?![A-Za-z0-9_-])"
)
_CATEGORICAL_FIELDS = (
    "gender",
    "master_category",
    "article_type",
    "base_colour",
    "usage",
    "season",
)


def _route_value(value: object) -> object:
    return getattr(value, "value", value)


def route_accuracy(rows: Iterable[Mapping[str, object]]) -> float:
    """Return the fraction of rows whose actual route matches its label."""

    values = list(rows)
    if not values:
        raise ValueError("route accuracy requires at least one row")
    matches = sum(
        _route_value(row["expected_route"])
        == _route_value(row["actual_route"])
        for row in values
    )
    return matches / len(values)


def _normalized_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, frozenset)):
        return sorted(
            item.casefold() if isinstance(item, str) else item
            for item in value
        )
    return value.casefold() if isinstance(value, str) else value


def field_accuracy(
    expected: Mapping[str, object],
    actual: Mapping[str, object] | ProductFilters,
) -> float:
    """Compare every labeled field, case-folding categorical list values."""

    if isinstance(actual, ProductFilters):
        actual_values: Mapping[str, object] = actual.model_dump(
            exclude_none=True
        )
    else:
        actual_values = actual
    if not expected:
        return 1.0
    matches = sum(
        _normalized_value(expected_value)
        == _normalized_value(actual_values.get(field))
        for field, expected_value in expected.items()
    )
    return matches / len(expected)


def hit_at_k(
    expected_ids: Sequence[str],
    returned_ids: Sequence[str],
    k: int,
) -> float:
    """Return one when any expected ID occurs in the first *k* results."""

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be greater than zero")
    expected = set(expected_ids)
    return float(bool(expected.intersection(returned_ids[:k])))


def _hit_complies(hit: RetrievalHit, filters: ProductFilters) -> bool:
    relaxed = set(hit.relaxed_filters).intersection(SOFT_FILTER_ORDER)
    for field in _CATEGORICAL_FIELDS:
        expected = getattr(filters, field)
        if not expected or field in relaxed:
            continue
        actual = hit.metadata.get(field)
        if not isinstance(actual, str):
            return False
        accepted = {value.casefold() for value in expected}
        if actual.casefold() not in accepted:
            return False

    price = hit.metadata.get("price")
    if filters.min_price is not None:
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            return False
        if float(price) < filters.min_price:
            return False
    if filters.max_price is not None:
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            return False
        if float(price) > filters.max_price:
            return False
    return True


def constraint_compliance(
    hits: Sequence[RetrievalHit],
    filters: ProductFilters,
    *,
    expected_ids: Sequence[str] | None = None,
) -> float:
    """Return the fraction of hits satisfying hard and active soft filters.

    Empty retrievals need their label to be meaningful: an empty result earns
    full credit only when the case is explicitly labeled with no matching IDs.
    """

    if not hits:
        return float(expected_ids is not None and not expected_ids)
    return sum(_hit_complies(hit, filters) for hit in hits) / len(hits)


def groundedness(answer: str, retrieved_ids: Iterable[str]) -> float:
    """Return the proportion of canonical cited IDs found in retrieval."""

    retrieved = set(retrieved_ids)
    mentioned = _CANONICAL_ID.findall(answer)
    if not mentioned:
        return float(not retrieved)
    return sum(record_id in retrieved for record_id in mentioned) / len(
        mentioned
    )
