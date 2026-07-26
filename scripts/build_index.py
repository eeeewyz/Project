from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from fashion_rag.config import Settings
from fashion_rag.retriever import SentenceTransformerEmbedder
from fashion_rag.schemas import Product


def main() -> None:
    with Path("data/sample_products.csv").open(encoding="utf-8") as handle:
        products = [
            Product.model_validate({**row, "price": float(row["price"])})
            for row in csv.DictReader(handle)
        ]
    embedder = SentenceTransformerEmbedder(
        Settings.from_env().embedding_model
    )
    embeddings = embedder.encode(
        [product.search_text for product in products]
    )
    output = Path("data/index")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "product_embeddings.npy", embeddings)
    print(
        f"built {len(products)} embeddings with dimension "
        f"{embeddings.shape[1]}"
    )


if __name__ == "__main__":
    main()
