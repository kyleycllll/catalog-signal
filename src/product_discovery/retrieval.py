"""Transparent reference candidate generation for offline retrieval baselines."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .schemas import Constraints, Product, RetrievalStrategy


RRF_K = 60


class DenseIndex(Protocol):
    metadata: object

    def search(
        self, query: str, top_k: int = 20, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]: ...


@dataclass(frozen=True)
class CandidateSet:
    products: list[Product]
    scores: dict[str, dict[str, float]]
    strategy: RetrievalStrategy


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def product_text(product: Product) -> str:
    return " ".join(
        filter(
            None,
            [
                product.title,
                product.description,
                product.bullet_points or "",
                product.category,
                product.colour or "",
                product.brand or "",
                product.material or "",
                *product.attributes,
            ],
        )
    )


def hard_filter(products: list[Product], constraints: Constraints) -> list[Product]:
    filtered: list[Product] = []
    for product in products:
        haystack = set(tokens(product_text(product)))
        if (
            constraints.category
            and constraints.category != "unknown"
            and product.category not in {constraints.category, "unknown"}
        ):
            continue
        if constraints.max_price is not None and product.price is not None and product.price > constraints.max_price:
            continue
        if constraints.min_price is not None and product.price is not None and product.price < constraints.min_price:
            continue
        if any(not set(tokens(attribute)).issubset(haystack) for attribute in constraints.required_attributes):
            continue
        if any(set(tokens(attribute)).issubset(haystack) for attribute in constraints.excluded_attributes):
            continue
        filtered.append(product)
    return filtered


def bm25_scores(query: str, products: list[Product], k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    query_terms = tokens(query)
    documents = {product.id: tokens(product_text(product)) for product in products}
    if not query_terms or not documents:
        return {product.id: 0.0 for product in products}
    avg_length = sum(map(len, documents.values())) / len(documents)
    document_frequency = Counter(
        term for term in set(query_terms) for document in documents.values() if term in document
    )
    scores: dict[str, float] = {}
    for product_id, document in documents.items():
        counts = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            idf = math.log(
                1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * len(document) / max(avg_length, 1))
            score += idf * frequency * (k1 + 1) / denominator
        scores[product_id] = score
    return scores


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, product_id in enumerate(ranking, start=1):
            fused[product_id] = fused.get(product_id, 0.0) + 1.0 / (k + rank)
    return fused


def _ordered_products(
    eligible: list[Product], ordered_ids: list[str], limit: int
) -> list[Product]:
    by_id = {product.id: product for product in eligible}
    return [by_id[product_id] for product_id in ordered_ids if product_id in by_id][:limit]


def _bm25_candidate_set(query: str, eligible: list[Product], limit: int) -> CandidateSet:
    raw = bm25_scores(query, eligible)
    ordered_ids = [product.id for product in sorted(eligible, key=lambda row: (-raw[row.id], row.id))]
    return CandidateSet(
        products=_ordered_products(eligible, ordered_ids, limit),
        scores={product.id: {"bm25_raw": raw[product.id], "retrieval": raw[product.id]} for product in eligible},
        strategy="bm25",
    )


def retrieve_candidates(
    query: str, products: list[Product], constraints: Constraints, limit: int = 40
) -> tuple[list[Product], dict[str, float]]:
    """Backward-compatible BM25-only candidate retrieval."""
    result = _bm25_candidate_set(query, hard_filter(products, constraints), limit)
    return result.products, {product_id: row["bm25_raw"] for product_id, row in result.scores.items()}


def retrieve(
    query: str,
    products: list[Product],
    constraints: Constraints,
    strategy: RetrievalStrategy = "hybrid",
    limit: int = 40,
    dense_index: DenseIndex | None = None,
) -> CandidateSet:
    """Retrieve candidates with BM25, dense cosine search, or deterministic RRF fusion."""
    eligible = hard_filter(products, constraints)
    if not eligible:
        return CandidateSet([], {}, strategy)
    if strategy == "bm25":
        return _bm25_candidate_set(query, eligible, limit)
    if dense_index is None:
        from .embeddings import DenseIndexUnavailable

        raise DenseIndexUnavailable("The requested dense retrieval index is unavailable; build and configure it first.")

    allowed_ids = {product.id for product in eligible}
    dense_rows = dense_index.search(query, top_k=len(eligible), allowed_ids=allowed_ids)
    dense_scores = dict(dense_rows)
    dense_order = [product_id for product_id, _ in dense_rows]
    if strategy == "dense":
        return CandidateSet(
            products=_ordered_products(eligible, dense_order, limit),
            scores={
                product.id: {"dense_cosine": dense_scores.get(product.id, float("-inf")), "retrieval": dense_scores.get(product.id, float("-inf"))}
                for product in eligible
            },
            strategy="dense",
        )

    bm25 = bm25_scores(query, eligible)
    bm25_order = [product.id for product in sorted(eligible, key=lambda row: (-bm25[row.id], row.id))]
    rrf = reciprocal_rank_fusion([bm25_order, dense_order])
    fused_order = sorted(set(bm25_order) | set(dense_order), key=lambda product_id: (-rrf[product_id], product_id))
    return CandidateSet(
        products=_ordered_products(eligible, fused_order, limit),
        scores={
            product.id: {
                "bm25_raw": bm25.get(product.id, 0.0),
                "dense_cosine": dense_scores.get(product.id, float("-inf")),
                "rrf": rrf.get(product.id, 0.0),
                "retrieval": rrf.get(product.id, 0.0),
            }
            for product in eligible
        },
        strategy="hybrid",
    )
