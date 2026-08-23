"""Known-label hard-negative selection for ESCI reranker training."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from .retrieval import DenseIndex, retrieve
from .schemas import Constraints, Product


def validate_query_disjoint(partitions: dict[str, list[dict[str, Any]]]) -> None:
    assigned: dict[str, str] = {}
    for split, rows in partitions.items():
        for row in rows:
            query_id = str(row["query_id"])
            previous = assigned.setdefault(query_id, split)
            if previous != split:
                raise ValueError(f"Query {query_id} leaks between {previous} and {split}")


def _products(rows: list[dict[str, Any]]) -> list[Product]:
    by_id: dict[str, Product] = {}
    for row in rows:
        product = Product.model_validate(row["product"])
        existing = by_id.setdefault(product.id, product)
        if existing != product:
            raise ValueError(f"Product {product.id} has inconsistent metadata in labelled rows")
    return list(by_id.values())


def _targets(rows: list[dict[str, Any]], examples_per_label: int | None) -> dict[str, int]:
    counts = Counter(str(row["label"]) for row in rows)
    if set(counts) - {"E", "S", "C", "I"}:
        raise ValueError("Hard-negative mining requires ESCI-labelled rows")
    if examples_per_label is None:
        return dict(counts)
    if examples_per_label < 1:
        raise ValueError("examples_per_label must be positive")
    return {label: min(count, examples_per_label) for label, count in counts.items()}


def _random_balanced(
    rows: list[dict[str, Any]], targets: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label, count in sorted(targets.items()):
        pool = [row for row in rows if row["label"] == label]
        selected.extend(random.Random(f"{seed}:{label}:baseline").sample(pool, count))
    random.Random(f"{seed}:baseline-shuffle").shuffle(selected)
    return selected


def _hybrid_hardness(rows: list[dict[str, Any]], dense_index: DenseIndex) -> dict[tuple[str, str], float]:
    products = _products(rows)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    scores: dict[tuple[str, str], float] = {}
    for query_id, query_rows in by_query.items():
        candidate_set = retrieve(
            query_rows[0]["query"],
            products,
            Constraints(),
            strategy="hybrid",
            limit=len(products),
            dense_index=dense_index,
        )
        for row in query_rows:
            product_id = str(row["product"]["id"])
            scores[(query_id, product_id)] = candidate_set.scores.get(product_id, {}).get("rrf", 0.0)
    return scores


def _mined_balanced(
    rows: list[dict[str, Any]],
    targets: dict[str, int],
    hardness: dict[tuple[str, str], float],
    seed: int,
    max_per_query: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label, count in sorted(targets.items()):
        pool = [row for row in rows if row["label"] == label]
        if label == "E":
            # The ablation changes difficult negatives, not the positive distribution.
            chosen = random.Random(f"{seed}:{label}:positive").sample(pool, count)
        else:
            ordered = sorted(
                pool,
                key=lambda row: (
                    -hardness[(str(row["query_id"]), str(row["product"]["id"]))],
                    str(row["query_id"]),
                    str(row["product"]["id"]),
                ),
            )
            chosen: list[dict[str, Any]] = []
            per_query: Counter[str] = Counter()
            for row in ordered:
                query_id = str(row["query_id"])
                if per_query[query_id] < max_per_query:
                    chosen.append(row)
                    per_query[query_id] += 1
                if len(chosen) == count:
                    break
            if len(chosen) < count:
                chosen_ids = {id(row) for row in chosen}
                chosen.extend(row for row in ordered if id(row) not in chosen_ids)
                chosen = chosen[:count]
        selected.extend(chosen)
    random.Random(f"{seed}:hard-negative-shuffle").shuffle(selected)
    return selected


def build_training_variants(
    rows: list[dict[str, Any]],
    dense_index: DenseIndex,
    seed: int = 42,
    examples_per_label: int | None = None,
    max_per_query: int = 3,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return equal-size/equal-label baseline and hard-negative training variants.

    Only explicitly judged S/C/I rows may become negatives. Unjudged retrieved products
    are never assigned a negative label.
    """
    if max_per_query < 1:
        raise ValueError("max_per_query must be positive")
    targets = _targets(rows, examples_per_label)
    hardness = _hybrid_hardness(rows, dense_index)
    baseline = _random_balanced(rows, targets, seed)
    hard_negative = _mined_balanced(rows, targets, hardness, seed, max_per_query)
    variants = {"baseline": baseline, "hard_negative": hard_negative}
    manifest = {
        "seed": seed,
        "target_label_counts": targets,
        "max_per_query_per_label": max_per_query,
        "hardness": "hybrid RRF rank over the labelled training catalog",
        "negative_policy": "only observed ESCI S/C/I labels; no unjudged products are negatives",
        "variants": {name: {"rows": len(values), "label_counts": dict(Counter(row["label"] for row in values))} for name, values in variants.items()},
    }
    return variants, manifest
