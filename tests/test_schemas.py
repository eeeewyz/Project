import pytest
from pydantic import ValidationError

from fashion_rag.schemas import ProductFilters, Route


def test_route_has_only_supported_values():
    assert Route.FAQ.value == "faq"
    assert Route.PRODUCT.value == "product"
    assert Route.UNSUPPORTED.value == "unsupported"


def test_price_bounds_must_be_ordered():
    with pytest.raises(ValidationError):
        ProductFilters(min_price=120, max_price=80)


def test_empty_filters_do_not_use_any_sentinel():
    filters = ProductFilters()
    assert filters.model_dump(exclude_none=True) == {}


def test_filters_reject_any_sentinel():
    with pytest.raises(ValidationError, match="Any"):
        ProductFilters(base_colour=["Any"])


def test_filters_reject_unknown_catalogue_values():
    with pytest.raises(ValidationError):
        ProductFilters(article_type=["Hat"])


def test_schema_models_forbid_extra_fields():
    with pytest.raises(ValidationError):
        ProductFilters(base_colour=["Blue"], accidental_field=True)


def test_price_bounds_do_not_coerce_strings_to_numbers():
    with pytest.raises(ValidationError):
        ProductFilters(max_price="100")
