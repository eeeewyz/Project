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
        ("FAQ-02", "When will I receive my refund?", "Approved refunds are issued to the original payment method within 5-10 business days.", "returns"),
        ("FAQ-03", "Can I exchange a size?", "You can request one size exchange within 30 days while stock is available.", "returns"),
        ("FAQ-04", "How long does standard shipping take?", "Standard shipping normally takes 3-5 business days.", "shipping"),
        ("FAQ-05", "Do you offer express shipping?", "Express shipping is available at checkout for eligible destinations.", "shipping"),
        ("FAQ-06", "Can I track my order?", "A tracking link is emailed after the order leaves the warehouse.", "shipping"),
        ("FAQ-07", "How can I contact support?", "Support is available through the website contact form from Monday to Friday.", "support"),
        ("FAQ-08", "What are support hours?", "Support operates Monday to Friday, 9:00-18:00 JST.", "support"),
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
        writer = csv.DictWriter(
            handle, fieldnames=list(products[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(products)
    (output_dir / "faq.json").write_text(
        json.dumps([item.model_dump() for item in generate_faqs()], indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_data(Path("data"))
