"""Generate the clean, reproducible portfolio exploration notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


OUTPUT_PATH = Path("notebooks/rag_pipeline_exploration.ipynb")


def _markdown(source: str, cell_id: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=source, id=cell_id)


def _code(source: str, cell_id: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=source, id=cell_id)


def build_notebook() -> nbformat.NotebookNode:
    """Build the five-section notebook without executing any cells."""

    cells = [
        _markdown(
            """# 1. Project objective and architecture

This notebook explores an explainable RAG assistant over a synthetic fashion
catalog and store FAQ collection.

```text
query → route → parse constraints → retrieve evidence → grounded answer
                         ↘ retrieval details and citations
```

The product path combines metadata constraints with FAISS cosine-similarity
ranking. The FAQ path uses semantic top-k retrieval. The application layer then
generates an answer from retrieved evidence and validates every cited ID.""",
            "objective",
        ),
        _markdown(
            """# 2. Load the synthetic data

The repository ships a deterministic synthetic catalog and FAQ collection.
They contain no real inventory, customers, or prices.""",
            "load-data-heading",
        ),
        _code(
            """import csv
import json
from pathlib import Path

from fashion_rag.schemas import FAQEntry, Product

project_root = Path.cwd()

with (project_root / "data/sample_products.csv").open(
    encoding="utf-8", newline=""
) as handle:
    products = [
        Product.model_validate({**row, "price": float(row["price"])})
        for row in csv.DictReader(handle)
    ]

faq_records = json.loads(
    (project_root / "data/faq.json").read_text(encoding="utf-8")
)
faqs = [FAQEntry.model_validate(record) for record in faq_records]

print(f"Loaded {len(products)} products and {len(faqs)} FAQs.")""",
            "load-data-code",
        ),
        _markdown(
            """# 3. FAQ semantic retrieval

FAQ questions and answers are embedded together. FAISS ranks the normalized
vectors by inner product, which is cosine similarity after normalization.""",
            "faq-heading",
        ),
        _code(
            """from fashion_rag.config import Settings
from fashion_rag.retriever import FAQRetriever, SentenceTransformerEmbedder

settings = Settings.from_env()
embedder = SentenceTransformerEmbedder(settings.embedding_model)
faq_retriever = FAQRetriever(faqs, embedder)

faq_hits = faq_retriever.search("When can I return an order?", top_k=3)
[
    {"id": hit.record_id, "score": round(hit.score, 3), "text": hit.text}
    for hit in faq_hits
]""",
            "faq-code",
        ),
        _markdown(
            """# 4. Product filters and FAISS retrieval

Article type and price remain hard constraints. When too few products match,
the retriever relaxes only soft filters in the documented order and reports
that decision with each result.""",
            "product-heading",
        ),
        _code(
            """from fashion_rag.retriever import ProductRetriever
from fashion_rag.schemas import ProductFilters

product_retriever = ProductRetriever(products, embedder)
filters = ProductFilters(
    article_type=["T-shirt"],
    base_colour=["Blue"],
    max_price=100.0,
)
product_hits = product_retriever.search(
    "comfortable blue T-shirts under $100",
    filters,
    top_k=5,
)
[
    {
        "id": hit.record_id,
        "price": hit.metadata["price"],
        "score": round(hit.score, 3),
        "relaxed_filters": hit.relaxed_filters,
    }
    for hit in product_hits
]""",
            "product-code",
        ),
        _markdown(
            """# 5. Measured evaluation

Live evaluation requires `TOGETHER_API_KEY` because routing, query parsing,
and answer generation call the configured model provider. No metrics are
published until all 30 labeled cases complete successfully.

Run the evaluation from the repository root:

```bash
export TOGETHER_API_KEY="your-key"
python -m evaluation.run_evaluation
```

Then regenerate this notebook with `python scripts/create_notebook.py`.
The cell below displays the measured baseline/improved table when
`evaluation/results.json` exists; otherwise it prints the same run command.""",
            "evaluation-heading",
        ),
        _code(
            """import json

results_path = project_root / "evaluation/results.json"
if not results_path.exists():
    print(
        "No verified live results are available. Run:\\n"
        'export TOGETHER_API_KEY="your-key"\\n'
        "python -m evaluation.run_evaluation"
    )
else:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    baseline = results["baseline"]
    improved = results["improved"]
    counts = results["metric_case_counts"]
    rows = [
        ("Route accuracy", "—", improved["route_accuracy"]),
        (
            "Metadata field accuracy",
            "—",
            improved["metadata_field_accuracy"],
        ),
        (
            f"Product Hit@5 (n={counts['product_hit_at_5']})",
            baseline["hit_at_5"],
            improved["hit_at_5"],
        ),
        (
            f"FAQ Hit@3 (n={counts['faq_hit_at_3']})",
            "—",
            improved["faq_hit_at_3"],
        ),
        (
            "Constraint compliance",
            baseline["constraint_compliance"],
            improved["constraint_compliance"],
        ),
        ("Groundedness", "—", improved["groundedness"]),
        ("Mean latency (ms)", "—", improved["mean_latency_ms"]),
    ]
    print(f"Measured on {results['case_count']} cases at {results['evaluated_at']}")
    print(f"{'Metric':<36} {'Baseline':>10} {'Improved':>10}")
    print("-" * 58)
    for metric, baseline_value, improved_value in rows:
        baseline_text = (
            baseline_value
            if isinstance(baseline_value, str)
            else f"{baseline_value:.3f}"
        )
        improved_text = (
            f"{improved_value:.1f}"
            if metric == "Mean latency (ms)"
            else f"{improved_value:.3f}"
        )
        print(f"{metric:<36} {baseline_text:>10} {improved_text:>10}")""",
            "evaluation-code",
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {
        "name": "python",
        "version": "3.11",
    }
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return notebook


def main() -> None:
    """Write the generated notebook at the documented repository path."""

    notebook = build_notebook()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT_PATH)


if __name__ == "__main__":
    main()
