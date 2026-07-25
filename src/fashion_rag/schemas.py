from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator


class StrictModel(BaseModel):
    """Shared boundary for data crossing application-module boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Route(StrEnum):
    FAQ = "faq"
    PRODUCT = "product"
    UNSUPPORTED = "unsupported"


class TaskNature(StrEnum):
    TECHNICAL = "technical"
    CREATIVE = "creative"


Gender = Literal["Men", "Women", "Unisex"]
MasterCategory = Literal["Apparel", "Footwear"]
ArticleType = Literal["T-shirt", "Dress", "Shirt", "Sneakers"]
BaseColour = Literal["Blue", "Black", "White", "Red"]
Usage = Literal["Casual", "Formal", "Sports"]
Season = Literal["Summer", "Spring", "Fall"]


class ChatTurn(StrictModel):
    role: str
    content: str


class Product(StrictModel):
    product_id: str
    name: str
    gender: Gender
    master_category: MasterCategory
    article_type: ArticleType
    base_colour: BaseColour
    season: Season
    usage: Usage
    price: StrictFloat = Field(gt=0)

    @property
    def search_text(self) -> str:
        return (
            f"{self.name}. {self.gender} {self.base_colour} "
            f"{self.article_type} for {self.usage} use in {self.season}. "
            f"Price ${self.price:.2f}."
        )


class FAQEntry(StrictModel):
    faq_id: str
    question: str
    answer: str
    topic: str

    @property
    def search_text(self) -> str:
        return f"{self.question} {self.answer}"


class ProductFilters(StrictModel):
    gender: list[Gender] | None = None
    master_category: list[MasterCategory] | None = None
    article_type: list[ArticleType] | None = None
    base_colour: list[BaseColour] | None = None
    usage: list[Usage] | None = None
    season: list[Season] | None = None
    min_price: StrictFloat | None = Field(default=None, ge=0)
    max_price: StrictFloat | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_price_range(self) -> "ProductFilters":
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must not exceed max_price")
        return self


class RouteDecision(StrictModel):
    route: Route


class ProductQuery(StrictModel):
    nature: TaskNature
    requested_count: int = Field(default=3, ge=1, le=5)
    filters: ProductFilters


class RetrievalHit(StrictModel):
    record_id: str
    text: str
    score: StrictFloat
    metadata: dict[str, Any]
    relaxed_filters: list[str] = Field(default_factory=list)


class PipelineResponse(StrictModel):
    answer: str
    route: Route
    nature: TaskNature | None = None
    applied_filters: ProductFilters | None = None
    hits: list[RetrievalHit] = Field(default_factory=list)
    latency_ms: StrictFloat = Field(ge=0)
