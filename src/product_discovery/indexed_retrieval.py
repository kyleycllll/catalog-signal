"""Indexed retrieval that reproduces ``retrieval.retrieve`` exactly at catalog scale.

``retrieval.retrieve`` re-tokenizes every product for every query and fuses full
rankings, which is correct but does not run on a million-product catalog. This
module computes the *same* rankings with an inverted index:

* BM25: identical tokenizer (``[a-z0-9]+`` over lower-cased ``product_text``),
  identical Okapi formula and float64 operation order, duplicate query terms
  counted as in ``bm25_scores``, ties broken by product ID (catalog rows are
  sorted by ID, so row order == ID order), zero-score products included.
* Dense: the persisted FAISS ``IndexFlatIP`` over normalized MiniLM vectors.
* Hybrid: RRF ``score(d) = sum_i 1 / (60 + rank_i(d))`` over the *full* BM25 and
  dense rankings, like ``retrieve``. Only the union of each list's top ``depth``
  can reach the fused top K (a bound checked at run time), and the exact
  full-list rank of each such product in both lists is used for its score.

Unit tests assert equality with ``retrieval.retrieve`` on small catalogs.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .retrieval import RRF_K

TOKEN_PATTERN = r"[a-z0-9]+"


class BM25Index:
    """Okapi BM25 over a CSC term-document matrix (rows = catalog order)."""

    def __init__(self, matrix, vocabulary: dict[str, int], k1: float = 1.5, b: float = 0.75):
        from scipy import sparse

        self.matrix = sparse.csc_matrix(matrix)
        self.vocabulary = vocabulary
        self.k1 = k1
        self.b = b
        self.document_count = self.matrix.shape[0]
        self.document_lengths = np.asarray(self.matrix.sum(axis=1)).ravel().astype(np.int64)
        # Python's ``sum(lengths) / count`` in bm25_scores: exact integer sum, one division.
        self.average_length = int(self.document_lengths.sum()) / self.document_count
        self._length_factor = self.b * self.document_lengths.astype(np.float64) / max(self.average_length, 1)

    @classmethod
    def build(cls, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            lowercase=True, token_pattern=TOKEN_PATTERN, dtype=np.int32, strip_accents=None
        )
        matrix = vectorizer.fit_transform(texts)
        vocabulary = {term: int(column) for term, column in vectorizer.vocabulary_.items()}
        return cls(matrix, vocabulary, k1=k1, b=b)

    def save(self, directory: str | Path, catalog_fingerprint: str) -> None:
        from scipy import sparse

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(target / "term_document.npz", self.matrix, compressed=False)
        (target / "metadata.json").write_text(
            json.dumps(
                {
                    "catalog_fingerprint": catalog_fingerprint,
                    "k1": self.k1,
                    "b": self.b,
                    "token_pattern": TOKEN_PATTERN,
                    "documents": self.document_count,
                    "vocabulary": self.vocabulary,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path, catalog_fingerprint: str) -> "BM25Index":
        from scipy import sparse

        source = Path(directory)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata["catalog_fingerprint"] != catalog_fingerprint or metadata["token_pattern"] != TOKEN_PATTERN:
            raise ValueError("BM25 index was built for a different catalog or tokenizer")
        return cls(
            sparse.load_npz(source / "term_document.npz"),
            metadata["vocabulary"],
            k1=metadata["k1"],
            b=metadata["b"],
        )

    def scores(self, query_terms: Sequence[str]) -> np.ndarray:
        """Full-catalog BM25 scores, bit-identical to ``retrieval.bm25_scores``."""
        scores = np.zeros(self.document_count, dtype=np.float64)
        n = self.document_count
        for term in query_terms:  # duplicates contribute repeatedly, as in bm25_scores
            column = self.vocabulary.get(term)
            if column is None:
                continue
            start, end = self.matrix.indptr[column], self.matrix.indptr[column + 1]
            rows = self.matrix.indices[start:end]
            frequency = self.matrix.data[start:end].astype(np.float64)
            df = end - start
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denominator = frequency + self.k1 * (1 - self.b + self._length_factor[rows])
            scores[rows] += idf * frequency * (self.k1 + 1) / denominator
        return scores

    def lexical_rarity(self, query_terms: Sequence[str]) -> float | None:
        """Mean BM25 IDF for recognized query terms, without scoring documents."""
        values: list[float] = []
        n = self.document_count
        for term in dict.fromkeys(query_terms):
            column = self.vocabulary.get(term)
            if column is None:
                continue
            df = self.matrix.indptr[column + 1] - self.matrix.indptr[column]
            values.append(math.log(1 + (n - df + 0.5) / (df + 0.5)))
        return sum(values) / len(values) if values else None


def top_rows_by_score(scores: np.ndarray, k: int) -> np.ndarray:
    """Rows of the top ``k`` scores ordered by (-score, row), including zero scores."""
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k == len(scores):
        candidates = np.arange(len(scores))
    else:
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        candidates = np.flatnonzero(scores >= threshold)
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order][:k]


def rank_in_full_ordering(scores: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """1-based rank of ``rows`` in the (-score, row) ordering of all ``scores``."""
    rows = np.asarray(rows, dtype=np.int64)
    ascending = np.sort(scores)
    values = scores[rows]
    right = np.searchsorted(ascending, values, side="right")
    left = np.searchsorted(ascending, values, side="left")
    ranks = 1 + (len(scores) - right)
    tied = np.flatnonzero(right - left > 1)
    for value in np.unique(values[tied]):
        tied_rows = np.flatnonzero(scores == value)
        members = tied[values[tied] == value]
        ranks[members] += np.searchsorted(tied_rows, rows[members], side="left")
    return ranks


@dataclass
class RetrievalResult:
    rows: list[int]
    scores: list[float]
    component_ranks: dict[str, list[int]] | None = None
    timings_ms: dict[str, float] | None = None
    field_score_adjustments: dict[int, float] | None = None


class IndexedRetriever:
    """BM25, dense (FAISS) and hybrid RRF retrieval over one catalog."""

    def __init__(
        self,
        tokenize: Callable[[str], list[str]],
        bm25: BM25Index,
        faiss_index,
        encode_query: Callable[[str], np.ndarray],
        vectors: np.ndarray,
        faiss_row_of_id: Sequence[int] | None = None,
        rrf_k: int = RRF_K,
        fusion_depth: int = 1000,
    ):
        self.tokenize = tokenize
        self.bm25 = bm25
        self.faiss_index = faiss_index
        self.encode_query = encode_query
        self.vectors = vectors
        self.rrf_k = rrf_k
        self.fusion_depth = fusion_depth
        if faiss_row_of_id is not None and list(faiss_row_of_id) != list(range(len(faiss_row_of_id))):
            raise ValueError("FAISS rows must be in catalog row order")

    def bm25_search(self, query: str, k: int) -> RetrievalResult:
        scores = self.bm25.scores(self.tokenize(query))
        rows = top_rows_by_score(scores, k)
        return RetrievalResult(rows=rows.tolist(), scores=scores[rows].tolist())

    def _dense_top(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """FAISS top ``k`` ordered by (-score, row); over-fetches so boundary ties resolve by row."""
        n = int(self.faiss_index.ntotal)
        k = min(k, n)
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        fetch = min(n, max(2 * k, k + 64))
        while True:
            scores, positions = self.faiss_index.search(query, fetch)
            keep = positions[0] >= 0
            rows, values = positions[0][keep].astype(np.int64), scores[0][keep]
            order = np.lexsort((rows, -values))
            rows, values = rows[order], values[order]
            if fetch >= n or len(rows) <= k or values[k - 1] > values[-1]:
                return rows[:k], values[:k]
            fetch = min(n, fetch * 2)

    def dense_search(self, query: str, k: int) -> RetrievalResult:
        rows, scores = self._dense_top(self.encode_query(query), k)
        return RetrievalResult(rows=rows.tolist(), scores=[float(value) for value in scores])

    def lexical_rarity(self, query: str) -> float | None:
        """Expose a lightweight lexical signal to the deterministic router."""
        return self.bm25.lexical_rarity(self.tokenize(query))

    def hybrid_search(
        self,
        query: str,
        k: int,
        *,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        field_boost: Callable[[Sequence[int]], Mapping[int, float]] | None = None,
        field_boost_depth: int = 0,
    ) -> RetrievalResult:
        """Fuse indexed BM25 and dense rankings with optional bounded field boosts.

        The optional boost only re-scores the leading lexical candidates supplied
        to it; production uses that hook to inspect title, brand, and colour for
        at most a small fixed number of rows.  It never scans catalog metadata.
        """
        n = self.bm25.document_count
        depth = min(self.fusion_depth, n)
        if k > depth:
            raise ValueError("fusion_depth must be at least k")
        if bm25_weight < 0 or dense_weight < 0 or bm25_weight + dense_weight <= 0:
            raise ValueError("At least one non-negative retrieval weight is required")
        if field_boost_depth < 0:
            raise ValueError("field_boost_depth cannot be negative")
        clock = time.perf_counter()
        bm25_scores = self.bm25.scores(self.tokenize(query))
        field_score_adjustments: dict[int, float] = {}
        if field_boost is not None and field_boost_depth:
            boost_rows = top_rows_by_score(bm25_scores, min(field_boost_depth, depth))
            proposed = field_boost(boost_rows)
            for row, adjustment in proposed.items():
                row = int(row)
                value = float(adjustment)
                if row < 0 or row >= n or not math.isfinite(value) or value <= 0:
                    continue
                field_score_adjustments[row] = value
                bm25_scores[row] += value
        bm25_top = top_rows_by_score(bm25_scores, depth)
        after_bm25 = time.perf_counter()
        query_vector = self.encode_query(query)
        after_encode = time.perf_counter()
        dense_top, _ = self._dense_top(query_vector, depth)
        after_dense = time.perf_counter()

        bm25_rank = {int(row): rank for rank, row in enumerate(bm25_top, start=1)}
        dense_rank = {int(row): rank for rank, row in enumerate(dense_top, start=1)}
        candidates = np.array(sorted(set(bm25_rank) | set(dense_rank)), dtype=np.int64)
        missing_bm25 = np.array([row for row in candidates if row not in bm25_rank], dtype=np.int64)
        if len(missing_bm25):
            for row, rank in zip(missing_bm25, rank_in_full_ordering(bm25_scores, missing_bm25)):
                bm25_rank[int(row)] = int(rank)
        missing_dense = np.array([row for row in candidates if row not in dense_rank], dtype=np.int64)
        if len(missing_dense):
            dense_scores = self.vectors @ query_vector.astype(np.float32).ravel()
            for row, rank in zip(missing_dense, rank_in_full_ordering(dense_scores, missing_dense)):
                dense_rank[int(row)] = int(rank)

        fused = {
            int(row): (
                bm25_weight / (self.rrf_k + bm25_rank[int(row)])
                + dense_weight / (self.rrf_k + dense_rank[int(row)])
            )
            for row in candidates
        }
        ordered = sorted(fused, key=lambda row: (-fused[row], row))[:k]
        # Any product outside both top-depth lists scores at most this bound; the
        # k-th fused score must exceed it for the truncated fusion to be exact.
        outside_bound = (bm25_weight + dense_weight) / (self.rrf_k + depth + 1)
        if n > depth and len(ordered) == k and fused[ordered[-1]] <= outside_bound:
            raise RuntimeError("Fusion depth too shallow for an exact hybrid top-k")
        return RetrievalResult(
            rows=ordered,
            scores=[fused[row] for row in ordered],
            component_ranks={
                "bm25": [bm25_rank[row] for row in ordered],
                "dense": [dense_rank[row] for row in ordered],
            },
            timings_ms={
                "bm25_top_depth": (after_bm25 - clock) * 1000,
                "query_embedding": (after_encode - after_bm25) * 1000,
                "faiss_top_depth": (after_dense - after_encode) * 1000,
                "exact_rank_lookup_and_fusion": (time.perf_counter() - after_dense) * 1000,
            },
            field_score_adjustments=field_score_adjustments,
        )

    def search(self, strategy: str, query: str, k: int) -> RetrievalResult:
        if strategy == "bm25":
            return self.bm25_search(query, k)
        if strategy == "dense":
            return self.dense_search(query, k)
        if strategy == "hybrid":
            return self.hybrid_search(query, k)
        raise ValueError(f"Unknown strategy {strategy}")
