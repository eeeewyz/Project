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
