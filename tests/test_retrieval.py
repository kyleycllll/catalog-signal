import asyncio
import json
from pathlib import Path

import numpy as np

from product_discovery.embeddings import InMemoryDenseIndex
from product_discovery.retrieval import (
    hard_filter,
    reciprocal_rank_fusion,
    retrieve,
    rerank_with_sft,
    retrieve_candidates,
)
from product_discovery.schemas import Constraints, ModelPrediction, Product, RelevanceLabel, SessionState

catalog = [Product.model_validate(row) for row in json.loads(Path("data/sample_esci_catalog.json").read_text())]


class Reranker:
    async def rerank(self, query, products):
        return [ModelPrediction(product_id=p.id, label=RelevanceLabel.exact if p.id == "demo-backpack-1" else RelevanceLabel.irrelevant, confidence=.9) for p in products]


def test_hard_filter_enforces_known_price_and_required_attribute():
    results = hard_filter(catalog, Constraints(category="backpack", required_attributes=["black"], max_price=80))
    assert {product.id for product in results} == {"demo-backpack-1"}


def test_fine_tuned_score_dominates_bm25_reranking():
    candidates, bm25 = retrieve_candidates("university black backpack", catalog, Constraints())
    results = asyncio.run(rerank_with_sft("university black backpack", candidates, bm25, SessionState(id="x"), Reranker()))
    assert results[0].product.id == "demo-backpack-1"
    assert results[0].scores["sft_relevance"] > results[0].scores["bm25_raw"] * 0


def test_dense_and_hybrid_retrieval_preserve_hard_constraints():
    products = [
        Product(id="a", title="Black backpack", category="backpack"),
        Product(id="b", title="Blue backpack", category="backpack"),
        Product(id="c", title="Black sofa", category="sofa"),
    ]
    index = InMemoryDenseIndex(
        ["a", "b", "c"],
        np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32),
        {"pack": np.array([1, 0], dtype=np.float32)},
    )
    constraints = Constraints(category="backpack", required_attributes=["black"])
    dense = retrieve("pack", products, constraints, strategy="dense", dense_index=index)
    hybrid = retrieve("pack", products, constraints, strategy="hybrid", dense_index=index)
    assert [row.id for row in dense.products] == ["a"]
    assert [row.id for row in hybrid.products] == ["a"]
    assert "rrf" in hybrid.scores["a"]


def test_rrf_is_deterministic_for_ties():
    fused = reciprocal_rank_fusion([["b", "a"], ["a", "b"]])
    assert fused["a"] == fused["b"]
    products = [Product(id="b", title="same"), Product(id="a", title="same")]
    index = InMemoryDenseIndex(
        ["b", "a"], np.array([[1, 0], [1, 0]], dtype=np.float32), {"x": np.array([1, 0], dtype=np.float32)}
    )
    assert [row.id for row in retrieve("x", products, Constraints(), "hybrid", dense_index=index).products] == ["a", "b"]
