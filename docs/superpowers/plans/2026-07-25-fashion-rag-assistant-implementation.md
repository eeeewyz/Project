# Fashion RAG Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independently implemented, interview-ready Fashion RAG Assistant with a public GitHub repository, measured retrieval improvements, and a live Gradio demo on Hugging Face Spaces.

**Architecture:** A bounded conversation enters an LLM router and then follows either an FAQ semantic-retrieval path or a product path that combines validated metadata filters with FAISS similarity search. A grounded generator may cite only the retrieved FAQ IDs or product IDs, while Gradio exposes the answer and retrieval details.

**Tech Stack:** Python 3.11, Pydantic 2.13.4, Together SDK 2.28.0, Qwen/Qwen3.5-9B, Sentence Transformers 5.6.1 with BAAI/bge-small-en-v1.5, FAISS CPU 1.14.3, Gradio 6.20.0, pytest 9.1.1, GitHub Actions, Hugging Face Spaces.

## Global Constraints

- Public code, comments, tests, UI labels, and the main README body are in English; the README begins with a short Japanese summary.
- Do not commit the raw Coursera notebook, `unittests.py`, `utils.py`, `flask_app.py`, `weaviate_server.py`, course proxy URLs, or course datasets.
- Use only the independently generated synthetic product catalog and independently written FAQ data.
- Read `TOGETHER_API_KEY` from the environment; never log or commit it.
- Use `Qwen/Qwen3.5-9B` through Together AI and disable thinking output for direct application responses.
- Keep at most four previous user/assistant turns when analyzing a follow-up.
- Price bounds and core `article_type` are hard filters and are never relaxed.
- Soft-filter relaxation order is `usage → season → base_colour → gender`.
- Product answers may reference only product IDs present in the retrieved context; FAQ answers may cite only retrieved FAQ IDs.
- Retry one time after a transient LLM failure or invalid structured output, then return a typed application error.
- GitHub Actions runs deterministic tests without live API credentials.
- Do not place evaluation numbers in the README until the evaluation command has produced them.
- Target a public Hugging Face Gradio Space on CPU Basic hardware.

---

## File Responsibility Map

- `app.py`: Hugging Face Space entry point; creates and queues the Gradio app.
- `pyproject.toml`: package metadata and editable-install configuration.
- `requirements.txt`: pinned runtime dependencies, including `-e .`.
- `requirements-dev.txt`: pinned test and notebook-generation dependencies.
- `src/fashion_rag/config.py`: environment-backed immutable settings.
- `src/fashion_rag/schemas.py`: all shared Pydantic domain and response models.
- `src/fashion_rag/llm.py`: provider-neutral protocol, Together adapter, JSON validation, one-retry policy.
- `src/fashion_rag/prompts.py`: prompt builders only; no API calls.
- `src/fashion_rag/router.py`: FAQ/product/unsupported routing.
- `src/fashion_rag/query_parser.py`: product-query metadata extraction.
- `src/fashion_rag/retriever.py`: metadata filtering, controlled relaxation, embedding, and FAISS search.
- `src/fashion_rag/pipeline.py`: orchestration, grounded generation, latency measurement, and application errors.
- `src/fashion_rag/ui.py`: Gradio callbacks and retrieval-detail formatting.
- `scripts/generate_demo_data.py`: deterministic synthetic product and FAQ generation.
- `scripts/build_index.py`: reproducible local embedding-cache build.
- `scripts/create_notebook.py`: creates the cleaned public exploration notebook from package APIs.
- `data/sample_products.csv`: generated synthetic catalog.
- `data/faq.json`: independently written FAQ records.
- `evaluation/eval_dataset.json`: labeled route, filter, retrieval, and grounding cases.
- `evaluation/metrics.py`: pure metric functions.
- `evaluation/run_evaluation.py`: baseline/improved evaluation runner and JSON/Markdown output.
- `tests/fakes.py`: deterministic fake LLM and embedder implementations.
- `tests/`: one focused test module per production responsibility.
- `.github/workflows/tests.yml`: Python 3.11 deterministic CI.
- `README.md`: Japanese introduction, English portfolio narrative, verified results, setup, and deployment.

---

### Task 1: Repository scaffold and environment-safe configuration

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/fashion_rag/__init__.py`
- Create: `src/fashion_rag/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.from_env() -> Settings`
- Produces: `Settings.require_api_key() -> str`
- Consumes: environment variables `TOGETHER_API_KEY`, `TOGETHER_MODEL`, `EMBEDDING_MODEL`, `TOP_K`, and `MAX_HISTORY_TURNS`

- [ ] **Step 1: Create packaging and dependency files**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "fashion-rag-assistant"
version = "0.1.0"
description = "An explainable FAQ and product-catalog RAG assistant."
requires-python = ">=3.11,<3.12"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Create `requirements.txt`:

```text
-e .
faiss-cpu==1.14.3
gradio==6.20.0
numpy>=2.1,<3
pydantic==2.13.4
sentence-transformers==5.6.1
together==2.28.0
```

Create `requirements-dev.txt`:

```text
-r requirements.txt
nbformat==5.10.4
pytest==9.1.1
```

Create `.env.example`:

```dotenv
TOGETHER_API_KEY=replace-with-your-key
TOGETHER_MODEL=Qwen/Qwen3.5-9B
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
TOP_K=5
MAX_HISTORY_TURNS=4
```

Create `.gitignore`:

```gitignore
.DS_Store
.env
.models/
.pytest_cache/
.ruff_cache/
__pycache__/
*.py[cod]
data/index/
evaluation/results.json
evaluation/results.md
```

- [ ] **Step 2: Write configuration tests**

Create `tests/test_config.py`:

```python
import pytest

from fashion_rag.config import MissingAPIKeyError, Settings


def test_defaults_do_not_require_api_key_at_startup(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    settings = Settings.from_env()
    assert settings.together_model == "Qwen/Qwen3.5-9B"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.top_k == 5
    assert settings.max_history_turns == 4
    assert settings.together_api_key is None


def test_require_api_key_returns_secret_without_logging_it(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "secret-value")
    assert Settings.from_env().require_api_key() == "secret-value"


def test_require_api_key_raises_typed_error(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="TOGETHER_API_KEY"):
        Settings.from_env().require_api_key()
```

- [ ] **Step 3: Run the focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'fashion_rag.config'`.

- [ ] **Step 4: Implement immutable settings**

Create `src/fashion_rag/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass


class MissingAPIKeyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    together_api_key: str | None
    together_model: str
    embedding_model: str
    top_k: int
    max_history_turns: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            together_api_key=os.getenv("TOGETHER_API_KEY") or None,
            together_model=os.getenv("TOGETHER_MODEL", "Qwen/Qwen3.5-9B"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
            ),
            top_k=int(os.getenv("TOP_K", "5")),
            max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "4")),
        )

    def require_api_key(self) -> str:
        if self.together_api_key is None:
            raise MissingAPIKeyError(
                "TOGETHER_API_KEY is not configured. Add it as an environment "
                "variable or Hugging Face Space secret."
            )
        return self.together_api_key
```

Create `src/fashion_rag/__init__.py`:

```python
from fashion_rag.config import Settings

__all__ = ["Settings"]
```

- [ ] **Step 5: Run the focused tests and verify pass**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt .env.example .gitignore src/fashion_rag tests/test_config.py
git commit -m "build: scaffold the Fashion RAG package"
```

---

### Task 2: Typed domain models and deterministic synthetic data

**Files:**
- Create: `src/fashion_rag/schemas.py`
- Create: `scripts/generate_demo_data.py`
- Create: `data/sample_products.csv`
- Create: `data/faq.json`
- Create: `tests/test_schemas.py`
- Create: `tests/test_demo_data.py`

**Interfaces:**
- Produces: `Route`, `TaskNature`, `ChatTurn`, `Product`, `FAQEntry`, `ProductFilters`, `ProductQuery`, `RetrievalHit`, and `PipelineResponse`
- Produces: `generate_products() -> list[Product]`
- Produces: `generate_faqs() -> list[FAQEntry]`

- [ ] **Step 1: Write schema tests**

Create `tests/test_schemas.py`:

```python
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
```

- [ ] **Step 2: Run schema tests and verify failure**

Run:

```bash
python -m pytest tests/test_schemas.py -q
```

Expected: import fails because `fashion_rag.schemas` does not exist.

- [ ] **Step 3: Implement shared models**

Create `src/fashion_rag/schemas.py` with these exact fields:

```python
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Route(StrEnum):
    FAQ = "faq"
    PRODUCT = "product"
    UNSUPPORTED = "unsupported"


class TaskNature(StrEnum):
    TECHNICAL = "technical"
    CREATIVE = "creative"


class ChatTurn(BaseModel):
    role: str
    content: str


class Product(BaseModel):
    product_id: str
    name: str
    gender: str
    master_category: str
    article_type: str
    base_colour: str
    season: str
    usage: str
    price: float = Field(gt=0)

    @property
    def search_text(self) -> str:
        return (
            f"{self.name}. {self.gender} {self.base_colour} "
            f"{self.article_type} for {self.usage} use in {self.season}. "
            f"Price ${self.price:.2f}."
        )


class FAQEntry(BaseModel):
    faq_id: str
    question: str
    answer: str
    topic: str

    @property
    def search_text(self) -> str:
        return f"{self.question} {self.answer}"


class ProductFilters(BaseModel):
    gender: list[str] | None = None
    master_category: list[str] | None = None
    article_type: list[str] | None = None
    base_colour: list[str] | None = None
    usage: list[str] | None = None
    season: list[str] | None = None
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_price_range(self) -> "ProductFilters":
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must not exceed max_price")
        return self


class RouteDecision(BaseModel):
    route: Route


class ProductQuery(BaseModel):
    nature: TaskNature
    requested_count: int = Field(default=3, ge=1, le=5)
    filters: ProductFilters


class RetrievalHit(BaseModel):
    record_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    relaxed_filters: list[str] = Field(default_factory=list)


class PipelineResponse(BaseModel):
    answer: str
    route: Route
    nature: TaskNature | None = None
    applied_filters: ProductFilters | None = None
    hits: list[RetrievalHit] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)
```

- [ ] **Step 4: Write deterministic data-generation tests**

Create `tests/test_demo_data.py`:

```python
from scripts.generate_demo_data import generate_faqs, generate_products


def test_catalog_is_deterministic_and_has_filter_coverage():
    first = generate_products()
    second = generate_products()
    assert first == second
    assert len(first) == 96
    assert len({p.product_id for p in first}) == 96
    assert {"Blue", "Black", "White", "Red"} <= {p.base_colour for p in first}
    assert any(p.article_type == "T-shirt" and p.price < 100 for p in first)


def test_faqs_are_independently_identified():
    faqs = generate_faqs()
    assert len(faqs) == 12
    assert len({f.faq_id for f in faqs}) == 12
    assert {"returns", "shipping", "support"} <= {f.topic for f in faqs}
```

- [ ] **Step 5: Implement deterministic data generation**

Create `scripts/generate_demo_data.py`. Generate 96 products from the exact cross-product below, using the index to assign IDs and a deterministic price:

```python
from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

from fashion_rag.schemas import FAQEntry, Product

COLOURS = ["Blue", "Black", "White", "Red"]
ARTICLE_TYPES = [
    ("T-shirt", "Apparel", "Casual", "Summer"),
    ("Dress", "Apparel", "Formal", "Spring"),
    ("Shirt", "Apparel", "Formal", "Fall"),
    ("Sneakers", "Footwear", "Sports", "Summer"),
]
GENDERS = ["Men", "Women", "Unisex"]
PRICE_LEVELS = [39.0, 79.0]


def generate_products() -> list[Product]:
    records: list[Product] = []
    for index, (colour, item, gender, base_price) in enumerate(
        product(COLOURS, ARTICLE_TYPES, GENDERS, PRICE_LEVELS), start=1
    ):
        article_type, category, usage, season = item
        price = base_price + (index % 5) * 4
        records.append(
            Product(
                product_id=f"P{index:04d}",
                name=f"{colour} {gender} {article_type}",
                gender=gender,
                master_category=category,
                article_type=article_type,
                base_colour=colour,
                season=season,
                usage=usage,
                price=price,
            )
        )
    return records


def generate_faqs() -> list[FAQEntry]:
    rows = [
        ("FAQ-01", "How do I return an item?", "Start a return within 30 days of delivery using your order page.", "returns"),
        ("FAQ-02", "When will I receive my refund?", "Approved refunds are issued to the original payment method within 5–10 business days.", "returns"),
        ("FAQ-03", "Can I exchange a size?", "You can request one size exchange within 30 days while stock is available.", "returns"),
        ("FAQ-04", "How long does standard shipping take?", "Standard shipping normally takes 3–5 business days.", "shipping"),
        ("FAQ-05", "Do you offer express shipping?", "Express shipping is available at checkout for eligible destinations.", "shipping"),
        ("FAQ-06", "Can I track my order?", "A tracking link is emailed after the order leaves the warehouse.", "shipping"),
        ("FAQ-07", "How can I contact support?", "Support is available through the website contact form from Monday to Friday.", "support"),
        ("FAQ-08", "What are support hours?", "Support operates Monday to Friday, 9:00–18:00 JST.", "support"),
        ("FAQ-09", "Can I cancel an order?", "Orders can be cancelled before warehouse processing begins.", "orders"),
        ("FAQ-10", "Which payment methods are accepted?", "The demo store accepts major credit cards and digital wallets.", "payments"),
        ("FAQ-11", "How do I find my size?", "Use the size guide linked on each product page.", "sizing"),
        ("FAQ-12", "Are sale items returnable?", "Sale items follow the same 30-day policy unless marked final sale.", "returns"),
    ]
    return [FAQEntry(faq_id=i, question=q, answer=a, topic=t) for i, q, a, t in rows]


def write_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    products = [item.model_dump() for item in generate_products()]
    with (output_dir / "sample_products.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(products[0]))
        writer.writeheader()
        writer.writerows(products)
    (output_dir / "faq.json").write_text(
        json.dumps([item.model_dump() for item in generate_faqs()], indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_data(Path("data"))
```

- [ ] **Step 6: Generate and validate public data**

Run:

```bash
python scripts/generate_demo_data.py
python -m pytest tests/test_schemas.py tests/test_demo_data.py -q
```

Expected: `5 passed`; `data/sample_products.csv` has 97 lines including its header, and `data/faq.json` contains 12 entries.

- [ ] **Step 7: Commit**

```bash
git add src/fashion_rag/schemas.py scripts/generate_demo_data.py data tests/test_schemas.py tests/test_demo_data.py
git commit -m "feat: add typed synthetic catalog and FAQ data"
```

---

### Task 3: Provider-neutral LLM adapter with validated structured output

**Files:**
- Create: `src/fashion_rag/llm.py`
- Create: `tests/fakes.py`
- Create: `tests/test_llm.py`

**Interfaces:**
- Produces: `LLMClient.complete(prompt, *, temperature, max_tokens) -> str`
- Produces: `TogetherLLM.complete(...) -> str`
- Produces: `StructuredGenerator.generate(prompt, schema, *, temperature, max_tokens) -> BaseModel`
- Produces: `LLMServiceError` and `StructuredOutputError`

- [ ] **Step 1: Write retry and validation tests**

Create `tests/fakes.py`:

```python
from collections.abc import Iterable


class FakeLLM:
    def __init__(self, responses: Iterable[str | Exception]):
        self.responses = iter(responses)
        self.calls: list[str] = []

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.calls.append(prompt)
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result
```

Create `tests/test_llm.py`:

```python
import pytest

from fashion_rag.config import Settings
from fashion_rag.llm import (
    StructuredGenerator,
    StructuredOutputError,
    TogetherLLM,
)
from fashion_rag.schemas import Route, RouteDecision
from tests.fakes import FakeLLM


def test_structured_generator_accepts_fenced_json():
    llm = FakeLLM(['```json\n{"route":"faq"}\n```'])
    result = StructuredGenerator(llm).generate(
        "route this", RouteDecision, temperature=0, max_tokens=40
    )
    assert result.route is Route.FAQ


def test_structured_generator_retries_once_after_invalid_json():
    llm = FakeLLM(["not-json", '{"route":"product"}'])
    result = StructuredGenerator(llm).generate(
        "route this", RouteDecision, temperature=0, max_tokens=40
    )
    assert result.route is Route.PRODUCT
    assert len(llm.calls) == 2
    assert "Return corrected JSON only" in llm.calls[1]


def test_structured_generator_stops_after_second_invalid_response():
    llm = FakeLLM(["bad", "still bad"])
    with pytest.raises(StructuredOutputError):
        StructuredGenerator(llm).generate(
            "route this", RouteDecision, temperature=0, max_tokens=40
        )


def test_together_adapter_retries_one_transient_failure():
    from types import SimpleNamespace

    class StubCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary timeout")
            message = SimpleNamespace(content="recovered")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    completions = StubCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    settings = Settings(
        together_api_key="test-key",
        together_model="Qwen/Qwen3.5-9B",
        embedding_model="BAAI/bge-small-en-v1.5",
        top_k=5,
        max_history_turns=4,
    )
    result = TogetherLLM(settings, client=client, retry_delay=0).complete(
        "hello", temperature=0, max_tokens=20
    )
    assert result == "recovered"
    assert completions.calls == 2
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_llm.py -q
```

Expected: import fails because `fashion_rag.llm` does not exist.

- [ ] **Step 3: Implement LLM protocol, Together adapter, and one retry**

Implement `src/fashion_rag/llm.py`:

```python
from __future__ import annotations

import re
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError
from together import Together

from fashion_rag.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMServiceError(RuntimeError):
    pass


class StructuredOutputError(LLMServiceError):
    pass


class LLMClient(Protocol):
    def complete(
        self, prompt: str, *, temperature: float, max_tokens: int
    ) -> str: ...


class TogetherLLM:
    def __init__(
        self,
        settings: Settings,
        client: Together | None = None,
        retry_delay: float = 0.25,
    ):
        self.settings = settings
        self.client = client or Together(api_key=settings.require_api_key())
        self.retry_delay = retry_delay

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.together_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    chat_template_kwargs={"enable_thinking": False},
                )
                break
            except Exception as exc:
                if attempt == 1:
                    raise LLMServiceError(
                        "The language-model service is unavailable."
                    ) from exc
                time.sleep(self.retry_delay)
        content = response.choices[0].message.content
        if not content:
            raise LLMServiceError("The language-model service returned no text.")
        return content


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    return match.group(1) if match else stripped


class StructuredGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(
        self,
        prompt: str,
        schema: type[T],
        *,
        temperature: float,
        max_tokens: int,
    ) -> T:
        current_prompt = prompt
        for attempt in range(2):
            raw = self.llm.complete(
                current_prompt, temperature=temperature, max_tokens=max_tokens
            )
            try:
                return schema.model_validate_json(strip_json_fence(raw))
            except ValidationError as exc:
                if attempt == 1:
                    raise StructuredOutputError(
                        f"Model output did not match {schema.__name__}."
                    ) from exc
                current_prompt = (
                    f"{prompt}\n\nThe previous response failed validation: {exc}.\n"
                    "Return corrected JSON only."
                )
        raise AssertionError("unreachable")
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_llm.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/fashion_rag/llm.py tests/fakes.py tests/test_llm.py
git commit -m "feat: add validated Together LLM adapter"
```

---

### Task 4: FAQ/product router with bounded history

**Files:**
- Create: `src/fashion_rag/prompts.py`
- Create: `src/fashion_rag/router.py`
- Create: `tests/test_router.py`

**Interfaces:**
- Consumes: `StructuredGenerator.generate(...)`
- Produces: `bound_history(history, max_turns) -> list[ChatTurn]`
- Produces: `QueryRouter.route(query, history) -> Route`

- [ ] **Step 1: Write router tests**

Create `tests/test_router.py`:

```python
from fashion_rag.llm import StructuredGenerator
from fashion_rag.router import QueryRouter, bound_history
from fashion_rag.schemas import ChatTurn, Route
from tests.fakes import FakeLLM


def test_history_is_bounded_to_four_latest_turns():
    history = [ChatTurn(role="user", content=str(i)) for i in range(7)]
    assert [turn.content for turn in bound_history(history, 4)] == ["3", "4", "5", "6"]


def test_router_returns_validated_route():
    router = QueryRouter(StructuredGenerator(FakeLLM(['{"route":"faq"}'])))
    assert router.route("How do returns work?", []) is Route.FAQ


def test_router_prompt_contains_supported_scope():
    llm = FakeLLM(['{"route":"unsupported"}'])
    QueryRouter(StructuredGenerator(llm)).route("Write Python code", [])
    assert '"faq", "product", or "unsupported"' in llm.calls[0]
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_router.py -q
```

Expected: import fails because `fashion_rag.router` does not exist.

- [ ] **Step 3: Add prompt builders and router**

Add to `src/fashion_rag/prompts.py`:

```python
from fashion_rag.schemas import ChatTurn


def format_history(history: list[ChatTurn]) -> str:
    if not history:
        return "(no previous conversation)"
    return "\n".join(f"{turn.role}: {turn.content}" for turn in history)


def build_router_prompt(query: str, history: list[ChatTurn]) -> str:
    return f"""Classify a fashion-store assistant query.
Return JSON with exactly one route: "faq", "product", or "unsupported".
faq: store policy, delivery, returns, payment, sizing, order, or support.
product: catalog search, product comparison, or outfit recommendation.
unsupported: everything outside those scopes.

Recent conversation:
{format_history(history)}

Current query:
{query}

Return JSON only, for example {{"route":"faq"}}."""
```

Create `src/fashion_rag/router.py`:

```python
from fashion_rag.llm import StructuredGenerator
from fashion_rag.prompts import build_router_prompt
from fashion_rag.schemas import ChatTurn, Route, RouteDecision


def bound_history(history: list[ChatTurn], max_turns: int) -> list[ChatTurn]:
    return history[-max_turns:]


class QueryRouter:
    def __init__(self, structured: StructuredGenerator, max_history_turns: int = 4):
        self.structured = structured
        self.max_history_turns = max_history_turns

    def route(self, query: str, history: list[ChatTurn]) -> Route:
        prompt = build_router_prompt(
            query, bound_history(history, self.max_history_turns)
        )
        return self.structured.generate(
            prompt, RouteDecision, temperature=0, max_tokens=40
        ).route
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_router.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/fashion_rag/prompts.py src/fashion_rag/router.py tests/test_router.py
git commit -m "feat: route FAQ and product questions"
```

---

### Task 5: Validated product-query parser

**Files:**
- Create: `src/fashion_rag/query_parser.py`
- Modify: `src/fashion_rag/prompts.py`
- Create: `tests/test_query_parser.py`

**Interfaces:**
- Consumes: `StructuredGenerator`
- Produces: `ProductQueryParser.parse(query, history) -> ProductQuery`

- [ ] **Step 1: Write parser tests**

Create `tests/test_query_parser.py`:

```python
from fashion_rag.llm import StructuredGenerator
from fashion_rag.query_parser import ProductQueryParser
from fashion_rag.schemas import TaskNature
from tests.fakes import FakeLLM


def test_parser_keeps_upper_price_without_minimum():
    llm = FakeLLM(
        ['{"nature":"technical","requested_count":3,"filters":'
         '{"article_type":["T-shirt"],"base_colour":["Blue"],"max_price":100}}']
    )
    parsed = ProductQueryParser(StructuredGenerator(llm)).parse(
        "Show three blue T-shirts under $100", []
    )
    assert parsed.nature is TaskNature.TECHNICAL
    assert parsed.filters.min_price is None
    assert parsed.filters.max_price == 100


def test_parser_uses_null_instead_of_any():
    llm = FakeLLM(
        ['{"nature":"creative","requested_count":3,"filters":'
         '{"usage":["Formal"],"season":null}}']
    )
    parsed = ProductQueryParser(StructuredGenerator(llm)).parse(
        "Create a formal look", []
    )
    assert parsed.filters.season is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_query_parser.py -q
```

Expected: import fails because `fashion_rag.query_parser` does not exist.

- [ ] **Step 3: Add the exact extraction prompt**

Append to `src/fashion_rag/prompts.py`:

```python
def build_product_query_prompt(query: str, history: list[ChatTurn]) -> str:
    return f"""Extract a product-catalog request as JSON.
Allowed nature values: "technical" or "creative".
requested_count must be an integer from 1 to 5; use 3 when unspecified.
Allowed filter keys: gender, master_category, article_type, base_colour,
usage, season, min_price, max_price.
Categorical filters are JSON lists or null. Never output "Any".
Price bounds are numbers or null.

Catalog values:
gender: Men, Women, Unisex
master_category: Apparel, Footwear
article_type: T-shirt, Dress, Shirt, Sneakers
base_colour: Blue, Black, White, Red
usage: Casual, Formal, Sports
season: Spring, Summer, Fall

Recent conversation:
{format_history(history)}

Current query:
{query}

Return JSON only."""
```

- [ ] **Step 4: Implement parser**

Create `src/fashion_rag/query_parser.py`:

```python
from fashion_rag.llm import StructuredGenerator
from fashion_rag.prompts import build_product_query_prompt
from fashion_rag.router import bound_history
from fashion_rag.schemas import ChatTurn, ProductQuery


class ProductQueryParser:
    def __init__(self, structured: StructuredGenerator, max_history_turns: int = 4):
        self.structured = structured
        self.max_history_turns = max_history_turns

    def parse(self, query: str, history: list[ChatTurn]) -> ProductQuery:
        prompt = build_product_query_prompt(
            query, bound_history(history, self.max_history_turns)
        )
        return self.structured.generate(
            prompt, ProductQuery, temperature=0, max_tokens=300
        )
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_query_parser.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/fashion_rag/prompts.py src/fashion_rag/query_parser.py tests/test_query_parser.py
git commit -m "feat: extract validated catalog filters"
```

---

### Task 6: Product filtering, controlled relaxation, and FAISS ranking

**Files:**
- Create: `src/fashion_rag/retriever.py`
- Create: `scripts/build_index.py`
- Create: `tests/test_retriever.py`

**Interfaces:**
- Produces: `Embedder.encode(texts: list[str]) -> np.ndarray`
- Produces: `SentenceTransformerEmbedder`
- Produces: `apply_product_filters(products, filters, relaxed) -> list[Product]`
- Produces: `ProductRetriever.search(query, filters, top_k) -> list[RetrievalHit]`

- [ ] **Step 1: Add a deterministic fake embedder**

Append to `tests/fakes.py`:

```python
import numpy as np


class FakeEmbedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("blue" in lowered),
                    float("t-shirt" in lowered),
                    float("dress" in lowered),
                    float("return" in lowered),
                ]
            )
        array = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)
```

- [ ] **Step 2: Write hard-filter and relaxation tests**

Create `tests/test_retriever.py`:

```python
from fashion_rag.retriever import ProductRetriever, apply_product_filters
from fashion_rag.schemas import Product, ProductFilters
from tests.fakes import FakeEmbedder


PRODUCTS = [
    Product(product_id="P1", name="Blue Tee", gender="Men", master_category="Apparel", article_type="T-shirt", base_colour="Blue", season="Summer", usage="Casual", price=80),
    Product(product_id="P2", name="Blue Premium Tee", gender="Women", master_category="Apparel", article_type="T-shirt", base_colour="Blue", season="Fall", usage="Casual", price=130),
    Product(product_id="P3", name="Red Tee", gender="Men", master_category="Apparel", article_type="T-shirt", base_colour="Red", season="Summer", usage="Sports", price=70),
    Product(product_id="P4", name="Blue Dress", gender="Women", master_category="Apparel", article_type="Dress", base_colour="Blue", season="Spring", usage="Formal", price=90),
]


def test_upper_price_bound_works_without_lower_bound():
    result = apply_product_filters(
        PRODUCTS,
        ProductFilters(article_type=["T-shirt"], max_price=100),
        relaxed=set(),
    )
    assert {p.product_id for p in result} == {"P1", "P3"}


def test_article_type_and_price_are_never_relaxed():
    retriever = ProductRetriever(PRODUCTS, FakeEmbedder(), min_results=2)
    hits = retriever.search(
        "blue formal product",
        ProductFilters(
            article_type=["T-shirt"],
            base_colour=["Blue"],
            usage=["Formal"],
            max_price=100,
        ),
        top_k=5,
    )
    assert [hit.record_id for hit in hits] == ["P1"]
    assert all(hit.metadata["article_type"] == "T-shirt" for hit in hits)
    assert all(hit.metadata["price"] <= 100 for hit in hits)


def test_soft_filters_relax_in_declared_order():
    retriever = ProductRetriever(PRODUCTS, FakeEmbedder(), min_results=2)
    hits = retriever.search(
        "blue T-shirt",
        ProductFilters(
            article_type=["T-shirt"],
            base_colour=["Blue"],
            usage=["Formal"],
            season=["Summer"],
            gender=["Men"],
            max_price=100,
        ),
        top_k=5,
    )
    assert hits[0].relaxed_filters[:2] == ["usage", "season"]
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_retriever.py -q
```

Expected: import fails because `fashion_rag.retriever` does not exist.

- [ ] **Step 4: Implement filtering and candidate-local FAISS search**

Create `src/fashion_rag/retriever.py`. Use case-insensitive exact matching for categorical values. The core loop must be:

```python
SOFT_FILTER_ORDER = ("usage", "season", "base_colour", "gender")


def apply_product_filters(
    products: list[Product],
    filters: ProductFilters,
    relaxed: set[str],
) -> list[Product]:
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
            if field not in relaxed and expected:
                if actual.casefold() not in {value.casefold() for value in expected}:
                    return False
        if filters.min_price is not None and product.price < filters.min_price:
            return False
        if filters.max_price is not None and product.price > filters.max_price:
            return False
        return True

    return [product for product in products if allowed(product)]
```

Implement these classes and signatures in the same file:

```python
class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")


class ProductRetriever:
    def __init__(
        self,
        products: list[Product],
        embedder: Embedder,
        min_results: int = 5,
    ):
        self.products = products
        self.embedder = embedder
        self.embeddings = embedder.encode([p.search_text for p in products])
        self.min_results = min_results

    def search(
        self, query: str, filters: ProductFilters, top_k: int
    ) -> list[RetrievalHit]:
        relaxed: list[str] = []
        candidates = apply_product_filters(self.products, filters, set(relaxed))
        for field in SOFT_FILTER_ORDER:
            if len(candidates) >= self.min_results:
                break
            if getattr(filters, field):
                relaxed.append(field)
                candidates = apply_product_filters(
                    self.products, filters, set(relaxed)
                )
        if not candidates:
            return []
        positions = [self.products.index(item) for item in candidates]
        matrix = self.embeddings[positions]
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        query_vector = self.embedder.encode([query])
        scores, local_positions = index.search(
            query_vector, min(top_k, len(candidates))
        )
        return [
            RetrievalHit(
                record_id=candidates[pos].product_id,
                text=candidates[pos].search_text,
                score=float(score),
                metadata=candidates[pos].model_dump(),
                relaxed_filters=relaxed.copy(),
            )
            for score, pos in zip(scores[0], local_positions[0], strict=True)
            if pos >= 0
        ]
```

Import `faiss`, `numpy as np`, `Protocol`, `SentenceTransformer`, and the schema types explicitly. Do not catch model-download failures in this module; the application factory handles them once.

- [ ] **Step 5: Add reproducible index inspection script**

Create `scripts/build_index.py`:

```python
import csv
from pathlib import Path

import numpy as np

from fashion_rag.config import Settings
from fashion_rag.retriever import SentenceTransformerEmbedder
from fashion_rag.schemas import Product


def main() -> None:
    with Path("data/sample_products.csv").open(encoding="utf-8") as handle:
        products = [Product.model_validate(row) for row in csv.DictReader(handle)]
    embedder = SentenceTransformerEmbedder(Settings.from_env().embedding_model)
    embeddings = embedder.encode([product.search_text for product in products])
    output = Path("data/index")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "product_embeddings.npy", embeddings)
    print(f"built {len(products)} embeddings with dimension {embeddings.shape[1]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run retrieval tests**

Run:

```bash
python -m pytest tests/test_retriever.py -q
```

Expected: `3 passed`.

- [ ] **Step 7: Commit**

```bash
git add src/fashion_rag/retriever.py scripts/build_index.py tests/fakes.py tests/test_retriever.py
git commit -m "feat: add filtered FAISS product retrieval"
```

---

### Task 7: FAQ semantic retrieval

**Files:**
- Modify: `src/fashion_rag/retriever.py`
- Create: `tests/test_faq_retriever.py`

**Interfaces:**
- Consumes: `FAQEntry`, `Embedder`
- Produces: `FAQRetriever.search(query, top_k=3) -> list[RetrievalHit]`

- [ ] **Step 1: Write FAQ retrieval test**

Create `tests/test_faq_retriever.py`:

```python
from fashion_rag.retriever import FAQRetriever
from fashion_rag.schemas import FAQEntry
from tests.fakes import FakeEmbedder


def test_return_question_retrieves_return_faq_first():
    faqs = [
        FAQEntry(faq_id="FAQ-01", question="How do I return an item?", answer="Return within 30 days.", topic="returns"),
        FAQEntry(faq_id="FAQ-02", question="How fast is shipping?", answer="Three to five days.", topic="shipping"),
    ]
    hits = FAQRetriever(faqs, FakeEmbedder()).search("I need to return it", top_k=2)
    assert hits[0].record_id == "FAQ-01"
    assert hits[0].metadata["topic"] == "returns"
```

- [ ] **Step 2: Run focused test and verify failure**

Run:

```bash
python -m pytest tests/test_faq_retriever.py -q
```

Expected: import fails because `FAQRetriever` is not defined.

- [ ] **Step 3: Implement FAQRetriever**

Append to `src/fashion_rag/retriever.py`:

```python
class FAQRetriever:
    def __init__(self, faqs: list[FAQEntry], embedder: Embedder):
        self.faqs = faqs
        self.embedder = embedder
        self.embeddings = embedder.encode([faq.search_text for faq in faqs])
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 3) -> list[RetrievalHit]:
        query_vector = self.embedder.encode([query])
        scores, positions = self.index.search(
            query_vector, min(top_k, len(self.faqs))
        )
        return [
            RetrievalHit(
                record_id=self.faqs[pos].faq_id,
                text=self.faqs[pos].search_text,
                score=float(score),
                metadata=self.faqs[pos].model_dump(),
            )
            for score, pos in zip(scores[0], positions[0], strict=True)
            if pos >= 0
        ]
```

Add `FAQEntry` to the schema imports in that module.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/test_faq_retriever.py tests/test_retriever.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/fashion_rag/retriever.py tests/test_faq_retriever.py
git commit -m "feat: add FAQ semantic retrieval"
```

---

### Task 8: Grounded answer generation and end-to-end pipeline

**Files:**
- Modify: `src/fashion_rag/prompts.py`
- Create: `src/fashion_rag/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `QueryRouter`, `ProductQueryParser`, `ProductRetriever`, `FAQRetriever`, `LLMClient`
- Produces: `FashionRAGPipeline.answer(query, history) -> PipelineResponse`
- Produces: `GroundingError` and `ApplicationError`

- [ ] **Step 1: Write pipeline behavior tests**

Create `tests/test_pipeline.py` with these collaborators and helpers before the four tests:

```python
from fashion_rag.pipeline import FashionRAGPipeline
from fashion_rag.schemas import (
    ProductFilters,
    ProductQuery,
    RetrievalHit,
    Route,
    TaskNature,
)
from tests.fakes import FakeLLM


class StubRouter:
    def __init__(self, route: Route):
        self.value = route

    def route(self, query, history):
        return self.value


class StubParser:
    def parse(self, query, history):
        return ProductQuery(
            nature=TaskNature.TECHNICAL,
            requested_count=3,
            filters=ProductFilters(),
        )


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, *args, **kwargs):
        return self.hits


def product_hit(record_id: str) -> RetrievalHit:
    return RetrievalHit(
        record_id=record_id,
        text=f"{record_id} Blue T-shirt",
        score=0.9,
        metadata={
            "product_id": record_id,
            "article_type": "T-shirt",
            "base_colour": "Blue",
            "price": 80,
        },
    )


def faq_hit(record_id: str) -> RetrievalHit:
    return RetrievalHit(
        record_id=record_id,
        text="How do I return an item? Return within 30 days.",
        score=0.9,
        metadata={
            "faq_id": record_id,
            "question": "How do I return an item?",
            "answer": "Return within 30 days.",
            "topic": "returns",
        },
    )


def make_pipeline(
    *,
    route: Route,
    product_hits=None,
    faq_hits=None,
    llm=None,
):
    active_llm = llm or FakeLLM([])
    return (
        FashionRAGPipeline(
            router=StubRouter(route),
            product_parser=StubParser(),
            product_retriever=StubRetriever(product_hits or []),
            faq_retriever=StubRetriever(faq_hits or []),
            llm=active_llm,
            top_k=5,
        ),
        active_llm,
    )


def test_unsupported_route_does_not_call_generator():
    pipeline, llm = make_pipeline(route=Route.UNSUPPORTED)
    response = pipeline.answer("Write Python code", [])
    assert response.route is Route.UNSUPPORTED
    assert response.hits == []
    assert "fashion products and store policies" in response.answer
    assert llm.calls == []


def test_empty_product_result_is_deterministic():
    pipeline, llm = make_pipeline(route=Route.PRODUCT, product_hits=[])
    response = pipeline.answer("Find a purple hat", [])
    assert "could not find a matching product" in response.answer.lower()
    assert llm.calls == []


def test_product_answer_retries_when_it_cites_unknown_id():
    llm = FakeLLM([
        "Try product P9999.",
        "Try product P0001.",
    ])
    pipeline, _ = make_pipeline(
        route=Route.PRODUCT,
        product_hits=[product_hit("P0001")],
        llm=llm,
    )
    response = pipeline.answer("Show a blue T-shirt", [])
    assert "P0001" in response.answer
    assert len(llm.calls) == 2


def test_faq_answer_may_cite_only_retrieved_faq_ids():
    llm = FakeLLM(["Returns are accepted within 30 days [FAQ-01]."])
    pipeline, _ = make_pipeline(
        route=Route.FAQ,
        faq_hits=[faq_hit("FAQ-01")],
        llm=llm,
    )
    response = pipeline.answer("How do returns work?", [])
    assert response.route is Route.FAQ
    assert response.hits[0].record_id == "FAQ-01"
```

Do not patch internal pipeline methods; tests use only constructor-injected collaborators.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_pipeline.py -q
```

Expected: import fails because `fashion_rag.pipeline` does not exist.

- [ ] **Step 3: Add grounded prompt builders**

Append to `src/fashion_rag/prompts.py`:

```python
def build_faq_answer_prompt(query: str, hits: list[RetrievalHit]) -> str:
    context = "\n".join(
        f"[{hit.record_id}] {hit.metadata['question']} — {hit.metadata['answer']}"
        for hit in hits
    )
    return f"""Answer only from the FAQ context below.
Cite the supporting FAQ ID in square brackets.
If the context does not answer the question, say so.

FAQ context:
{context}

Question: {query}"""


def build_product_answer_prompt(
    query: str,
    product_query: ProductQuery,
    hits: list[RetrievalHit],
) -> str:
    context = "\n".join(
        f"[{hit.record_id}] {hit.text}" for hit in hits
    )
    temperature_instruction = (
        "Be concise and factual."
        if product_query.nature is TaskNature.TECHNICAL
        else "Create a coherent outfit while staying within the supplied catalog."
    )
    return f"""Answer only from the product context below.
Mention every recommended product by its exact product ID.
Recommend at most {product_query.requested_count} products.
{temperature_instruction}

Product context:
{context}

Question: {query}"""
```

Import `ProductQuery`, `RetrievalHit`, and `TaskNature`.

- [ ] **Step 4: Implement orchestration and grounding check**

Create `src/fashion_rag/pipeline.py` with:

```python
PRODUCT_ID_PATTERN = re.compile(r"\bP\d{4}\b")
FAQ_ID_PATTERN = re.compile(r"\bFAQ-\d{2}\b")


class ApplicationError(RuntimeError):
    pass


class GroundingError(ApplicationError):
    pass


def _is_grounded(answer: str, allowed_ids: set[str], pattern: re.Pattern[str]) -> bool:
    mentioned = set(pattern.findall(answer))
    return bool(mentioned) and mentioned <= allowed_ids


class FashionRAGPipeline:
    def __init__(
        self,
        *,
        router: QueryRouter,
        product_parser: ProductQueryParser,
        product_retriever: ProductRetriever,
        faq_retriever: FAQRetriever,
        llm: LLMClient,
        top_k: int = 5,
    ):
        self.router = router
        self.product_parser = product_parser
        self.product_retriever = product_retriever
        self.faq_retriever = faq_retriever
        self.llm = llm
        self.top_k = top_k
```

Implement `FashionRAGPipeline.answer` with this order:

1. start `time.perf_counter()`;
2. route the query;
3. return the deterministic unsupported response when needed;
4. retrieve top three FAQs and generate with temperature `0.1`;
5. or parse a product query, retrieve products, and generate with temperature `0.2` for technical / `0.8` for creative;
6. validate cited IDs against retrieved IDs;
7. if invalid, call the LLM once more with `Use only these IDs: ...`;
8. after a second invalid answer, raise `GroundingError("The model cited items outside the retrieved context.")`;
9. return `PipelineResponse` with elapsed milliseconds.

Use these deterministic no-result messages:

```python
UNSUPPORTED_MESSAGE = (
    "I can help with fashion products and store policies. "
    "Please ask about the catalog, outfits, shipping, returns, sizing, or support."
)
NO_PRODUCT_MESSAGE = (
    "I could not find a matching product without violating your core "
    "product-type or price constraints. Try changing a color, season, or usage."
)
```

- [ ] **Step 5: Run pipeline tests**

Run:

```bash
python -m pytest tests/test_pipeline.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Run the complete deterministic suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass without `TOGETHER_API_KEY`.

- [ ] **Step 7: Commit**

```bash
git add src/fashion_rag/prompts.py src/fashion_rag/pipeline.py tests/test_pipeline.py
git commit -m "feat: orchestrate grounded FAQ and product answers"
```

---

### Task 9: Application factory and explainable Gradio UI

**Files:**
- Create: `src/fashion_rag/ui.py`
- Create: `app.py`
- Create: `tests/test_ui.py`

**Interfaces:**
- Produces: `load_products(path) -> list[Product]`
- Produces: `load_faqs(path) -> list[FAQEntry]`
- Produces: `create_pipeline(settings) -> FashionRAGPipeline`
- Produces: `format_retrieval_details(response) -> str`
- Produces: `build_demo(pipeline_or_error) -> gr.Blocks`

- [ ] **Step 1: Write pure UI-formatting tests**

Create `tests/test_ui.py`:

```python
from fashion_rag.schemas import PipelineResponse, ProductFilters, RetrievalHit, Route
from fashion_rag.ui import format_retrieval_details


def test_details_show_route_filters_ids_scores_and_relaxation():
    response = PipelineResponse(
        answer="Try P0001.",
        route=Route.PRODUCT,
        applied_filters=ProductFilters(
            article_type=["T-shirt"], base_colour=["Blue"], max_price=100
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
    assert "product" in markdown
    assert "P0001" in markdown
    assert "0.910" in markdown
    assert "usage" in markdown
    assert "100" in markdown
```

- [ ] **Step 2: Run focused test and verify failure**

Run:

```bash
python -m pytest tests/test_ui.py -q
```

Expected: import fails because `fashion_rag.ui` does not exist.

- [ ] **Step 3: Implement data loaders and Markdown details**

In `src/fashion_rag/ui.py`, implement CSV loading with `csv.DictReader` and FAQ loading with `json.loads`. Parse each row through `Product.model_validate` or `FAQEntry.model_validate`.

Implement `format_retrieval_details` using this stable output structure:

```markdown
**Route:** product  
**Nature:** technical  
**Latency:** 123 ms  
**Applied filters:** `{"article_type":["T-shirt"],...}`

| ID | Score | Relaxed filters |
|---|---:|---|
| P0001 | 0.910 | usage |
```

Use `json.dumps(..., ensure_ascii=False)` for the filters. Use `None` as `—`. Escape table-breaking pipe characters in record text.

- [ ] **Step 4: Implement the Gradio Blocks app**

Use `gr.Blocks(title="Fashion RAG Assistant")`, `gr.Chatbot(type="messages")`, `gr.Textbox`, `gr.Markdown`, `gr.State`, and `gr.Accordion("Retrieval Details")`.

The submit callback has this signature:

```python
def respond(
    message: str,
    messages: list[dict[str, str]],
    history_state: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]], list[dict[str, str]], str]:
```

It must:

- convert the last four stored dictionaries to `ChatTurn`;
- call `pipeline.answer`;
- append user and assistant messages;
- clear the textbox;
- return formatted retrieval details;
- catch `MissingAPIKeyError`, `LLMServiceError`, `GroundingError`, and `ApplicationError` separately;
- show a concise error message without exception internals or secrets.

Add three buttons that submit the approved example prompts. Add this subtitle:

```text
An explainable RAG demo for store FAQs, catalog search, and outfit recommendations.
```

- [ ] **Step 5: Add the Space entry point**

Create `app.py`:

```python
from fashion_rag.config import Settings
from fashion_rag.ui import build_application

demo = build_application(Settings.from_env())

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch()
```

`build_application` constructs the embedder, product/FAQ retrievers, Together client, router, parser, pipeline, and UI exactly once. If model or data initialization fails, it returns a Gradio page containing a short startup-error Markdown message and does not reveal a stack trace.

- [ ] **Step 6: Run tests and a no-key startup smoke check**

Run:

```bash
python -m pytest tests/test_ui.py -q
python -c "import app; print(type(app.demo).__name__)"
```

Expected: UI test passes and the smoke check prints `Blocks` even without an API key.

- [ ] **Step 7: Run the app with a real local secret**

Run:

```bash
TOGETHER_API_KEY="your-local-key" python app.py
```

Manually verify all three example buttons, the retrieval-details accordion, one unsupported question, and one impossible price/type combination. Stop the server after verification. Do not paste the key into shell history when executing outside an ephemeral development environment; use a local `.env` loader or exported session variable.

- [ ] **Step 8: Commit**

```bash
git add app.py src/fashion_rag/ui.py tests/test_ui.py
git commit -m "feat: add explainable Gradio chat interface"
```

---

### Task 10: Baseline-vs-improved evaluation

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/eval_dataset.json`
- Create: `evaluation/metrics.py`
- Create: `evaluation/run_evaluation.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces: `route_accuracy(rows) -> float`
- Produces: `field_accuracy(expected, actual) -> float`
- Produces: `hit_at_k(expected_ids, returned_ids, k) -> float`
- Produces: `constraint_compliance(hits, filters) -> float`
- Produces: `groundedness(answer, retrieved_ids) -> float`
- Produces: `evaluation/results.json` and `evaluation/results.md`

- [ ] **Step 1: Create the labeled evaluation set**

Create `evaluation/eval_dataset.json` with exactly 30 records:

- 8 FAQ cases covering returns, refund timing, exchange, shipping time, tracking, support contact, support hours, and payment;
- 17 product cases covering upper-only price, lower-only price, price range, color, article type, gender, usage, season, technical requests, creative outfit requests, and two follow-ups whose `history` resolves “blue ones” or “cheaper options”;
- 5 unsupported cases covering coding, weather, mathematics, medical advice, and unrelated travel.

Each record uses:

```json
{
  "case_id": "P-01",
  "query": "Show me three blue T-shirts under $100.",
  "history": [],
  "expected_route": "product",
  "expected_nature": "technical",
  "expected_filters": {
    "article_type": ["T-shirt"],
    "base_colour": ["Blue"],
    "max_price": 100
  },
  "expected_faq_ids": [],
  "expected_product_ids": []
}
```

For product cases, `expected_product_ids` may be empty when the assertion is entirely constraint-based. FAQ cases include one expected FAQ ID. Unsupported cases leave nature, filters, and IDs null/empty.

Use this exact case inventory when writing the JSON; list-valued filters use the catalog spelling defined in Task 2:

| ID | Query / history | Expected route | Expected details |
|---|---|---|---|
| F-01 | How do I return an item? | faq | `FAQ-01` |
| F-02 | When will my refund arrive? | faq | `FAQ-02` |
| F-03 | Can I exchange for another size? | faq | `FAQ-03` |
| F-04 | How long does standard shipping take? | faq | `FAQ-04` |
| F-05 | Where can I track my order? | faq | `FAQ-06` |
| F-06 | How can I contact customer support? | faq | `FAQ-07` |
| F-07 | What time is support available? | faq | `FAQ-08` |
| F-08 | Which payment methods can I use? | faq | `FAQ-10` |
| P-01 | Show me three blue T-shirts under $100. | product | technical; `article_type=["T-shirt"]`, `base_colour=["Blue"]`, `max_price=100`; IDs `P0001`–`P0006` |
| P-02 | Show white sneakers costing at least $50. | product | technical; `article_type=["Sneakers"]`, `base_colour=["White"]`, `min_price=50`; IDs `P0068,P0069,P0070,P0072` |
| P-03 | Find black sneakers between $50 and $100. | product | technical; `article_type=["Sneakers"]`, `base_colour=["Black"]`, `min_price=50`, `max_price=100`; IDs `P0043,P0044,P0046,P0048` |
| P-04 | Show women's red dresses. | product | technical; `gender=["Women"]`, `article_type=["Dress"]`, `base_colour=["Red"]`; IDs `P0081,P0082` |
| P-05 | I need a white shirt for a man. | product | technical; `gender=["Men"]`, `article_type=["Shirt"]`, `base_colour=["White"]`; IDs `P0061,P0062` |
| P-06 | Do you have blue unisex sneakers? | product | technical; `gender=["Unisex"]`, `article_type=["Sneakers"]`, `base_colour=["Blue"]`; IDs `P0023,P0024` |
| P-07 | Give me summer T-shirts. | product | technical; `article_type=["T-shirt"]`, `season=["Summer"]` |
| P-08 | Show formal dresses. | product | technical; `article_type=["Dress"]`, `usage=["Formal"]` |
| P-09 | Find sneakers for sports. | product | technical; `article_type=["Sneakers"]`, `usage=["Sports"]` |
| P-10 | Create a summer wedding look for a man. | product | creative; `gender=["Men"]`, `season=["Summer"]`, `usage=["Formal"]` |
| P-11 | Create a black outfit for a woman. | product | creative; `gender=["Women"]`, `base_colour=["Black"]` |
| P-12 | Give me three red T-shirts. | product | technical; count 3; `article_type=["T-shirt"]`, `base_colour=["Red"]`; IDs `P0073`–`P0078` |
| P-13 | Show two blue dresses below $100. | product | technical; count 2; `article_type=["Dress"]`, `base_colour=["Blue"]`, `max_price=100`; IDs `P0007`–`P0012` |
| P-14 | Show one white pair of sneakers. | product | technical; count 1; `article_type=["Sneakers"]`, `base_colour=["White"]`; IDs `P0067`–`P0072` |
| P-15 | History user: “Show red T-shirts.” Current: “What about the blue ones?” | product | technical; `article_type=["T-shirt"]`, `base_colour=["Blue"]`; IDs `P0001`–`P0006` |
| P-16 | History user: “Show black dresses under $120.” Current: “Any cheaper options under $80?” | product | technical; `article_type=["Dress"]`, `base_colour=["Black"]`, `max_price=80`; IDs `P0031,P0033,P0035` |
| P-17 | Find a blue T-shirt under $10. | product | technical; `article_type=["T-shirt"]`, `base_colour=["Blue"]`, `max_price=10`; no matching ID |
| U-01 | Write Python code for quicksort. | unsupported | no retrieval |
| U-02 | What is the weather in Tokyo? | unsupported | no retrieval |
| U-03 | Solve x squared minus one. | unsupported | no retrieval |
| U-04 | Diagnose my headache. | unsupported | no retrieval |
| U-05 | Plan a five-day trip to Paris. | unsupported | no retrieval |

Expand ID ranges into individual strings in the JSON. Compute Hit@5 only for cases with at least one expected ID; compute constraint compliance for all product cases, including the expected-empty P-17 case.

- [ ] **Step 2: Write pure metric tests**

Create `tests/test_metrics.py`:

```python
from evaluation.metrics import (
    constraint_compliance,
    field_accuracy,
    groundedness,
    hit_at_k,
)
from fashion_rag.schemas import ProductFilters, RetrievalHit


def test_hit_at_k():
    assert hit_at_k(["P2"], ["P1", "P2", "P3"], 2) == 1.0
    assert hit_at_k(["P4"], ["P1", "P2", "P3"], 3) == 0.0


def test_field_accuracy_counts_missing_and_extra_values():
    expected = {"base_colour": ["Blue"], "max_price": 100}
    actual = {"base_colour": ["Blue"], "max_price": 120}
    assert field_accuracy(expected, actual) == 0.5


def test_groundedness_rejects_unknown_product_id():
    assert groundedness("Use P0001 and P9999.", {"P0001"}) == 0.5


def test_constraint_compliance_checks_upper_only_price():
    hits = [
        RetrievalHit(record_id="P1", text="", score=1, metadata={"article_type":"T-shirt","price":80}),
        RetrievalHit(record_id="P2", text="", score=.9, metadata={"article_type":"T-shirt","price":120}),
    ]
    filters = ProductFilters(article_type=["T-shirt"], max_price=100)
    assert constraint_compliance(hits, filters) == 0.5
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
python -m pytest tests/test_metrics.py -q
```

Expected: import fails because `evaluation.metrics` does not exist.

- [ ] **Step 4: Implement metrics**

Implement each named function as a pure function. `field_accuracy` compares every expected key after normalizing categorical list values with case-folding. `groundedness` extracts `P\d{4}` and `FAQ-\d{2}` IDs; return `1.0` when no IDs are expected and none are mentioned, otherwise return the proportion of mentioned IDs in the retrieved set. `constraint_compliance` reuses the same hard and active-soft checks as `apply_product_filters` and returns the fraction of hits that comply. When retrieval correctly returns no hits for a case labeled with no matching product IDs, constraint compliance is `1.0`; an unexpected empty result for a case with expected IDs is `0.0`.

- [ ] **Step 5: Implement baseline and improved evaluation runner**

`evaluation/run_evaluation.py` accepts:

```text
--dataset evaluation/eval_dataset.json
--output-json evaluation/results.json
--output-markdown evaluation/results.md
```

The baseline uses embedding similarity over all products without metadata filters. The improved system uses `ProductRetriever.search`. For each case, collect route correctness, metadata field accuracy, Hit@5, constraint compliance, groundedness, and latency. Aggregate with arithmetic means and write:

```json
{
  "evaluated_at": "ISO-8601 UTC timestamp",
  "case_count": 30,
  "baseline": {
    "hit_at_5": 0.0,
    "constraint_compliance": 0.0
  },
  "improved": {
    "route_accuracy": 0.0,
    "metadata_field_accuracy": 0.0,
    "hit_at_5": 0.0,
    "constraint_compliance": 0.0,
    "groundedness": 0.0,
    "mean_latency_ms": 0.0
  }
}
```

The numeric values above describe the output schema only; the runner replaces them with measured values. `results.md` renders the same values as a baseline/improved table.

- [ ] **Step 6: Run deterministic metric tests**

Run:

```bash
python -m pytest tests/test_metrics.py -q
```

Expected: `4 passed`.

- [ ] **Step 7: Run live evaluation and inspect every failure**

Run:

```bash
TOGETHER_API_KEY="your-local-key" python evaluation/run_evaluation.py
```

Expected: both result files are created, `case_count` is 30, every metric is between `0.0` and `1.0` except latency, and no answer cites an ID absent from its retrieved set. Inspect per-case output before accepting aggregate scores; fix code or labels when the evidence shows a real issue, then rerun.

- [ ] **Step 8: Commit evaluation code and labeled data**

Do not commit generated result files yet; they remain ignored until Task 11 deliberately force-adds the verified files.

```bash
git add evaluation/__init__.py evaluation/eval_dataset.json evaluation/metrics.py evaluation/run_evaluation.py tests/test_metrics.py
git commit -m "test: add baseline and improved RAG evaluation"
```

---

### Task 11: CI, cleaned notebook, README, license, and verified evidence

**Files:**
- Create: `.github/workflows/tests.yml`
- Create: `scripts/create_notebook.py`
- Create: `notebooks/rag_pipeline_exploration.ipynb`
- Delete: `README`
- Create: `README.md`
- Create: `LICENSE`
- Create after live verification: `evaluation/results.json`
- Create after live verification: `evaluation/results.md`

**Interfaces:**
- CI command: `python -m pytest -q`
- Notebook command: `python scripts/create_notebook.py`

- [ ] **Step 1: Add deterministic GitHub Actions**

Create `.github/workflows/tests.yml`:

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -r requirements-dev.txt
      - run: python -m pytest -q
```

- [ ] **Step 2: Create a clean reproducible notebook generator**

Create `scripts/create_notebook.py` with `nbformat`. It writes five sections:

1. project objective and architecture;
2. load the synthetic catalog and FAQ data;
3. demonstrate FAQ retrieval;
4. demonstrate product filters plus FAISS retrieval;
5. load `evaluation/results.json` and display the measured baseline/improved table.

The generated notebook imports only public package modules. Set its kernel to Python 3 and clear all outputs before writing:

```python
notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
nbformat.write(notebook, "notebooks/rag_pipeline_exploration.ipynb")
```

Do not copy Markdown or code from the Coursera notebook.

- [ ] **Step 3: Write the portfolio README**

Create `README.md` in this exact order:

1. `# Fashion RAG Assistant`
2. two-sentence Japanese summary;
3. a local-run badge and test badge; do not add a live-demo badge or screenshot link until Task 12 has verified their real targets;
4. short outcome-first description;
5. problem statement;
6. architecture diagram;
7. key features;
8. baseline vs improved table copied from verified `evaluation/results.md`;
9. local setup commands;
10. Hugging Face deployment and `TOGETHER_API_KEY` secret instructions;
11. limitations and future work;
12. data/model/course-inspiration attribution.

Include exactly:

> The project was inspired by concepts learned in a RAG course, while the application architecture, evaluation pipeline, deployment, and implementation were independently redesigned.

The local setup block is:

```bash
git clone https://github.com/eeeewyz/fashion-rag-assistant.git
cd fashion-rag-assistant
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export TOGETHER_API_KEY="your-key"
python app.py
```

State that demo data is synthetic and contains no real inventory or prices.

Remove the extensionless empty README before creating the Markdown file:

```bash
git rm README
```

- [ ] **Step 4: Add MIT license**

Create `LICENSE` using the standard MIT license text with:

```text
Copyright (c) 2026 Wang Yuzhe
```

- [ ] **Step 5: Generate notebook and verify it contains no course material**

Run:

```bash
python scripts/create_notebook.py
rg -n -i "coursera|grader|unittests|dlai.link|jovyan" notebooks README.md src tests data
```

Expected: the only `coursera` occurrence is the approved high-level course-inspiration disclosure if that term is used; there are no grader, proxy, absolute-course-path, or course-test references.

- [ ] **Step 6: Add measured results intentionally**

Review `evaluation/results.json` and `evaluation/results.md`, then force-add them because they are normally ignored:

```bash
git add -f evaluation/results.json evaluation/results.md
```

Copy the measured result table into README without changing the numbers. If the live evaluation has not run successfully, omit the result table and do not claim improvement.

- [ ] **Step 7: Run full local verification**

Run:

```bash
python -m pytest -q
python scripts/generate_demo_data.py
python scripts/create_notebook.py
python -m compileall -q src app.py evaluation scripts
git status --short
```

Expected: tests pass, compileall exits zero, generated files are stable, and status contains only the intended Task 11 files.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/tests.yml scripts/create_notebook.py notebooks/rag_pipeline_exploration.ipynb README.md LICENSE
git add -f evaluation/results.json evaluation/results.md
git commit -m "docs: add portfolio narrative and verified evaluation"
```

---

### Task 12: Rename, publish, deploy, and perform final verification

**Files:**
- Create: `assets/demo_screenshot.png`
- Modify after deployment: `README.md`

**External targets:**
- Rename GitHub repository: `eeeewyz/Project` → `eeeewyz/fashion-rag-assistant`
- Hugging Face Space: `fashion-rag-assistant` under the authenticated user's verified Hugging Face username
- Hugging Face secret: `TOGETHER_API_KEY`

- [ ] **Step 1: Verify the branch before publishing**

Run:

```bash
git status --short
git log --oneline --decorate -12
python -m pytest -q
```

Expected: clean worktree, intentional task-sized commits, and all tests passing.

- [ ] **Step 2: Rename the GitHub repository**

In GitHub repository settings, change the repository name from `Project` to `fashion-rag-assistant`. Confirm the resulting URL is:

```text
https://github.com/eeeewyz/fashion-rag-assistant
```

Update the local remote:

```bash
git remote set-url origin https://github.com/eeeewyz/fashion-rag-assistant.git
git remote -v
```

Expected: fetch and push URLs both use the new repository name.

- [ ] **Step 3: Push the implementation branch and open a draft PR**

Use branch name:

```text
feat/fashion-rag-assistant
```

Push it and open a draft PR titled:

```text
Build interview-ready Fashion RAG Assistant
```

The PR body lists: modular pipeline, validated structured output, filtered FAISS retrieval, Gradio UI, evaluation, CI, and course-asset exclusion.

- [ ] **Step 4: Create the Hugging Face Space**

Run `hf auth whoami` and record the returned username as the task-specific value `FASHION_HF_OWNER`. Create a public Gradio Space named `fashion-rag-assistant` under that exact account, using Python 3.11 and CPU Basic. Push the exact reviewed source state to the Space repository. Add `TOGETHER_API_KEY` under Space **Settings → Variables and secrets → Secrets**; never add the value as a normal public variable.

- [ ] **Step 5: Verify the live demo**

Open:

```text
https://huggingface.co/spaces/${FASHION_HF_OWNER}/fashion-rag-assistant
```

Verify:

1. Space reaches `Running`;
2. all three example prompts return answers;
3. retrieval details show route, filters, IDs, scores, and relaxation;
4. “Write Python code for quicksort” returns the unsupported response;
5. an impossible hard-constrained product request returns the deterministic no-match response;
6. Space logs contain no API key, authorization header, or full conversation history.

- [ ] **Step 6: Capture the real application screenshot**

Capture the running UI at a desktop width, crop browser chrome, and save as:

```text
assets/demo_screenshot.png
```

The screenshot must show one product answer and the expanded retrieval details without exposing secrets or personal data.

- [ ] **Step 7: Update and verify README links**

Confirm README links point to:

```text
GitHub: https://github.com/eeeewyz/fashion-rag-assistant
Demo: the verified Space URL opened successfully in Step 5
```

Add the verified demo badge and the screenshot directly below the Japanese summary:

```markdown
[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-yellow)](the verified Space URL)

![Fashion RAG Assistant demo](assets/demo_screenshot.png)
```

Replace the parenthesized link target with the exact URL verified in Step 5 before committing; do not leave the prose target in the Markdown file.

Run:

```bash
test -s assets/demo_screenshot.png
rg -n "eeeewyz/(Project|fashion-rag-assistant)" README.md
python -m pytest -q
```

Expected: screenshot is non-empty, no active README link uses `eeeewyz/Project`, and all tests pass.

- [ ] **Step 8: Commit the deployment evidence**

```bash
git add assets/demo_screenshot.png README.md
git commit -m "docs: link live demo and add verified screenshot"
git push
```

- [ ] **Step 9: Mark the PR ready only after remote checks pass**

Inspect GitHub Actions on the PR. If checks pass and the Space remains healthy after a cold restart, mark the draft PR ready for review. Do not merge automatically; leave the final merge decision to the repository owner.

---

## Final Acceptance Checklist

- [ ] Public repository contains no raw Coursera assets or unclear-license course data.
- [ ] `python -m pytest -q` passes without an API key.
- [ ] Local app imports without an API key and provides a configuration message.
- [ ] Live Space runs with the secret configured.
- [ ] FAQ and product paths both retrieve before generation.
- [ ] Price and article-type hard constraints are never relaxed.
- [ ] Product and FAQ citations are subsets of retrieved IDs.
- [ ] Retrieval details expose route, filters, scores, IDs, and relaxed fields.
- [ ] Evaluation contains 30 labeled cases and reports measured baseline/improved results.
- [ ] README is English-first with a Japanese introduction, real screenshot, verified numbers, live-demo link, limitations, and provenance.
- [ ] GitHub Actions passes on Python 3.11.
