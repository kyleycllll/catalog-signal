import numpy as np
import pytest

from product_discovery.embeddings import InMemoryDenseIndex
from product_discovery.hard_negatives import build_training_variants, validate_query_disjoint


def _row(query_id, product_id, label):
    return {
        "query_id": query_id,
        "query": "trail pack",
        "label": label,
        "product": {"id": product_id, "title": product_id, "category": "bag"},
    }


def test_hard_negative_variant_keeps_label_mix_and_prefers_retrieved_negative():
    rows = [
        _row("q1", "exact", "E"),
        _row("q1", "substitute", "S"),
        _row("q1", "complement", "C"),
        _row("q1", "irrelevant-low", "I"),
        _row("q1", "irrelevant-high", "I"),
    ]
    index = InMemoryDenseIndex(
        ["exact", "substitute", "complement", "irrelevant-low", "irrelevant-high"],
        np.array([[0, 1], [0, 1], [0, 1], [0, 1], [1, 0]], dtype=np.float32),
        {"trail pack": np.array([1, 0], dtype=np.float32)},
    )
    variants, manifest = build_training_variants(rows, index, examples_per_label=1)
    assert manifest["variants"]["baseline"]["label_counts"] == manifest["variants"]["hard_negative"]["label_counts"]
    mined_irrelevant = [row for row in variants["hard_negative"] if row["label"] == "I"]
    assert mined_irrelevant[0]["product"]["id"] == "irrelevant-high"
    assert manifest["negative_policy"].startswith("only observed ESCI")


def test_query_leakage_is_rejected():
    with pytest.raises(ValueError):
        validate_query_disjoint({"train": [_row("q1", "p1", "E")], "test": [_row("q1", "p2", "I")]})
