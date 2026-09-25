"""The single in-process production search pipeline.

The serving path is deliberately narrow:

``query understanding -> indexed hybrid retrieval -> MiniLM cross-encoder ->
constraint adjustment -> ranked products``.

Reference retrieval baselines remain separate from serving and are not imported
or initialized by this pipeline.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .adaptive_retrieval import (
    AdaptiveRetrievalDecision,
    FieldLexicalMatch,
    choose_retrieval_weights,
    lexical_fields_match,
)
from .cross_encoder_reranker import CrossEncoderReranker
from .embeddings import FaissTextEmbeddingIndex, catalog_fingerprint
from .indexed_retrieval import BM25Index, IndexedRetriever, RetrievalResult
from .query_understanding import QueryAnalysis, analyze_query
from .retrieval import RRF_K, product_text, tokens
from .schemas import Product, RankedResult, RelevanceLabel
from .serving_catalog import ArrowServingCatalog, row_order_hash


CANDIDATE_COUNT = 40
FUSION_DEPTH = 1000
FIELD_BOOST_DEPTH = 120
EXCLUSION_PENALTY = 0.75
DEFAULT_CATALOG_METADATA_PATH = "data/indexes/esci_us_product_metadata.arrow"
DEFAULT_DENSE_INDEX_PATH = "data/indexes/esci_us_minilm.faiss"
DEFAULT_BM25_INDEX_PATH = "data/indexes/esci_us_bm25"
DEFAULT_CROSS_ENCODER_PATH = "models/ablation/ce_A_balanced_random/epoch-3"


class ServingCatalog(Protocol):
    def __len__(self) -> int: ...

    def product_at(self, row: int, product_id: str) -> Product: ...

    def lexical_fields_at(self, row: int, product_id: str) -> Mapping[str, str | None]: ...


class CrossEncoderScorer(Protocol):
    model_dir: str

    def score_products(
        self, query: str, products: list[Product]
    ) -> tuple[dict[str, str], dict[str, float], dict[str, list[float]]]: ...


@dataclass
class ProductListCatalog:
    """Small JSON-catalog adapter for explicit local development fixtures."""

    products: list[Product]

    def __post_init__(self) -> None:
        self.products = sorted(self.products, key=lambda product: product.id)
        self.ids = [product.id for product in self.products]

    def __len__(self) -> int:
        return len(self.products)

    def fingerprint(self) -> str:
        return catalog_fingerprint(self.products)

    def product_at(self, row: int, product_id: str) -> Product:
        product = self.products[row]
        if product.id != product_id:
            raise RuntimeError("Fixture catalog row does not match the retrieval index ID")
        return product

    def lexical_fields_at(self, row: int, product_id: str) -> Mapping[str, str | None]:
        product = self.product_at(row, product_id)
        return {"title": product.title, "brand": product.brand, "colour": product.colour}


@dataclass
class PipelineSearchResult:
    analysis: QueryAnalysis
    retrieval_decision: AdaptiveRetrievalDecision
    results: list[RankedResult]
    candidate_count: int
    stage_latency_ms: dict[str, float]


class SearchPipeline:
    """Owns the loaded catalog, retrieval indexes and one cross-encoder."""

    def __init__(
        self,
        catalog: ServingCatalog,
        catalog_ids: Sequence[str],
        retriever: IndexedRetriever,
        reranker: CrossEncoderScorer,
        index_version: str,
        model_version: str,
        candidate_count: int = CANDIDATE_COUNT,
    ):
        self.catalog = catalog
        self.catalog_ids = catalog_ids
        if len(self.catalog_ids) != len(catalog):
            raise ValueError("Catalog metadata row count does not match retrieval index IDs")
        self.retriever = retriever
        self.reranker = reranker
        self.index_version = index_version
        self.model_version = model_version
        self.candidate_count = candidate_count

    def search(self, query: str) -> PipelineSearchResult:
        understanding_started = time.perf_counter()
        analysis = analyze_query(query)
        understanding_ms = (time.perf_counter() - understanding_started) * 1000

        lexical_rarity = self.retriever.lexical_rarity(analysis.retrieval_query)
        decision = choose_retrieval_weights(analysis, lexical_rarity=lexical_rarity)
        retrieval = self.retriever.hybrid_search(
            analysis.retrieval_query,
            self.candidate_count,
            bm25_weight=decision.bm25_weight,
            dense_weight=decision.dense_weight,
            field_boost=lambda rows: self._field_aware_bm25_adjustments(analysis, rows),
            field_boost_depth=FIELD_BOOST_DEPTH,
        )
        candidates = [
            self.catalog.product_at(row, self.catalog_ids[row]) for row in retrieval.rows
        ]

        rerank_started = time.perf_counter()
        labels, expected_gains, _ = self.reranker.score_products(analysis.normalized_query, candidates)
        results = rank_cross_encoder_candidates(
            analysis=analysis,
            candidates=candidates,
            retrieval=retrieval,
            labels=labels,
            expected_gains=expected_gains,
            retrieval_decision=decision,
        )
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        retrieval_timings = retrieval.timings_ms or {}
        stage_latency_ms = {
            "query_understanding": round(understanding_ms, 2),
            "bm25_retrieval": round(retrieval_timings.get("bm25_top_depth", 0.0), 2),
            "dense_retrieval": round(
                retrieval_timings.get("query_embedding", 0.0)
                + retrieval_timings.get("faiss_top_depth", 0.0),
                2,
            ),
            "fusion": round(retrieval_timings.get("exact_rank_lookup_and_fusion", 0.0), 2),
            "cross_encoder_reranking": round(rerank_ms, 2),
        }
        return PipelineSearchResult(
            analysis=analysis,
            retrieval_decision=decision,
            results=results[:10],
            candidate_count=len(candidates),
            stage_latency_ms=stage_latency_ms,
        )

    def _field_aware_bm25_adjustments(
        self, analysis: QueryAnalysis, rows: Sequence[int]
    ) -> dict[int, float]:
        """Re-score a fixed lexical prefix using only memory-mapped identifier fields."""
        adjustments: dict[int, float] = {}
        for raw_row in rows:
            row = int(raw_row)
            fields = self.catalog.lexical_fields_at(row, self.catalog_ids[row])
            match = lexical_fields_match(analysis, fields)
            if match.boost > 0:
                adjustments[row] = match.boost
        return adjustments


def rank_cross_encoder_candidates(
    *,
    analysis: QueryAnalysis,
    candidates: Sequence[Product],
    retrieval: RetrievalResult,
    labels: dict[str, str],
    expected_gains: dict[str, float],
    retrieval_decision: AdaptiveRetrievalDecision | None = None,
) -> list[RankedResult]:
    """Rank fixed hybrid candidates by expected ESCI gain and exclusions.

    The cross-encoder's probability-weighted ESCI gain is the ranking signal.
    A deterministic penalty only adjusts products that contain a customer-stated
    exclusion; it does not replace retrieval or introduce a second reranker.
    """
    component_ranks = retrieval.component_ranks or {}
    bm25_ranks = component_ranks.get("bm25", [])
    dense_ranks = component_ranks.get("dense", [])
    field_score_adjustments = retrieval.field_score_adjustments or {}
    ranked: list[tuple[int, RankedResult]] = []
    for position, product in enumerate(candidates):
        expected_gain = float(expected_gains[product.id])
        violations = exclusion_violations(product, analysis.exclusions)
        penalty = EXCLUSION_PENALTY * len(violations)
        final_score = expected_gain - penalty
        label = RelevanceLabel(labels[product.id])
        retrieval_score = float(retrieval.scores[position])
        field_boost = float(field_score_adjustments.get(retrieval.rows[position], 0.0))
        field_match = lexical_fields_match(
            analysis,
            {
                "title": product.title,
                "brand": product.brand,
                "colour": product.colour,
            },
        )
        scores = {
            "bm25_rank": float(bm25_ranks[position]) if position < len(bm25_ranks) else 0.0,
            "dense_rank": float(dense_ranks[position]) if position < len(dense_ranks) else 0.0,
            "fusion_score": round(retrieval_score, 6),
            "cross_encoder_expected_gain": round(expected_gain, 6),
            "negation_penalty": round(penalty, 6),
            "final_score": round(final_score, 6),
            "field_lexical_boost": round(field_boost, 6),
        }
        if retrieval_decision is not None:
            scores.update(
                {
                    "adaptive_bm25_weight": retrieval_decision.bm25_weight,
                    "adaptive_dense_weight": retrieval_decision.dense_weight,
                }
            )
        ranked.append(
            (
                position,
                RankedResult(
                    product=product,
                    score=round(final_score, 6),
                    scores=scores,
                    relevance_label=label,
                    explanation=ranking_explanation(
                        product=product,
                        analysis=analysis,
                        expected_gain=expected_gain,
                        violations=violations,
                        dense_rank=int(scores["dense_rank"]),
                        field_match=field_match if field_boost else None,
                    ),
                ),
            )
        )
    # Retain the indexed hybrid order as the deterministic tie-breaker, then ID.
    return [row for _, row in sorted(ranked, key=lambda item: (-item[1].score, item[0], item[1].product.id))]


def exclusion_violations(product: Product, exclusions: Sequence[str]) -> tuple[str, ...]:
    if not exclusions:
        return ()
    product_tokens = set(tokens(product_text(product)))
    return tuple(exclusion for exclusion in exclusions if exclusion in product_tokens)


def ranking_explanation(
    *,
    product: Product,
    analysis: QueryAnalysis,
    expected_gain: float,
    violations: Sequence[str],
    dense_rank: int,
    field_match: FieldLexicalMatch | None,
) -> str:
    """Build a compact explanation only from signals used by the pipeline."""
    reasons: list[str] = []
    if field_match:
        reasons.extend(f"field-aware {reason}" for reason in field_match.reasons)
    product_brand = " ".join(tokens(product.brand or ""))
    if product_brand and product_brand in analysis.brands and not field_match:
        reasons.append(f"exact {product.brand} brand match")
    title_tokens = set(tokens(product.title))
    overlap = [token for token in analysis.tokens if token in title_tokens and token not in analysis.brands]
    if overlap and not field_match:
        reasons.append(f"title match for {', '.join(dict.fromkeys(overlap[:3]))}")
    matched_models = [token for token in analysis.model_tokens if token in title_tokens]
    if matched_models and not field_match:
        reasons.append(f"model-token match for {', '.join(matched_models)}")
    if product.colour and " ".join(tokens(product.colour)) in analysis.colors and not field_match:
        reasons.append(f"{product.colour} colour match")
    if dense_rank and dense_rank <= 10:
        reasons.append("strong dense-retrieval rank")
    reasons.append(f"cross-encoder expected ESCI gain {expected_gain:.2f}/3")
    if violations:
        reasons.append(f"contains excluded term {', '.join(violations)}")
    return "; ".join(reasons)


def load_production_pipeline() -> SearchPipeline:
    """Load all production assets once during FastAPI startup.

    Missing or incompatible indexes are startup/readiness failures.  The server
    never falls back to the slow reference scanner or a remote model service.
    """
    metadata_path = Path(os.getenv("PRODUCT_METADATA_PATH", DEFAULT_CATALOG_METADATA_PATH))
    dense_path = Path(os.getenv("DENSE_INDEX_PATH", DEFAULT_DENSE_INDEX_PATH))
    bm25_path = Path(os.getenv("BM25_INDEX_PATH", DEFAULT_BM25_INDEX_PATH))
    model_path = os.getenv("CROSS_ENCODER_MODEL_PATH", DEFAULT_CROSS_ENCODER_PATH)
    if not metadata_path.exists():
        raise RuntimeError(
            f"Serving metadata is missing at {metadata_path}. "
            "Run scripts/build_serving_metadata.py before starting the API."
        )
    dense = FaissTextEmbeddingIndex.load(dense_path)
    catalog = ArrowServingCatalog(
        metadata_path,
        expected_product_count=dense.metadata.product_count,
        expected_fingerprint=dense.metadata.catalog_fingerprint,
        expected_row_order_hash=row_order_hash(dense.ids),
    )
    bm25 = BM25Index.load(bm25_path, dense.metadata.catalog_fingerprint)
    if bm25.document_count != len(catalog):
        raise RuntimeError("BM25 index document count does not match the catalog")
    retriever = IndexedRetriever(
        tokenize=tokens,
        bm25=bm25,
        faiss_index=dense.index,
        encode_query=dense.query_vector,
        vectors=dense.vectors_view(),
        rrf_k=RRF_K,
        fusion_depth=FUSION_DEPTH,
    )
    reranker = CrossEncoderReranker(model_path)
    return SearchPipeline(
        catalog=catalog,
        catalog_ids=dense.ids,
        retriever=retriever,
        reranker=reranker,
        index_version=dense.metadata.version,
        model_version=f"cross-encoder:{model_path}",
    )
