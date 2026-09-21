"""The indexed benchmark retriever must reproduce ``retrieval.retrieve`` exactly."""
import numpy as np
import pytest

from product_discovery.retrieval import bm25_scores, product_text, retrieve, tokens
from product_discovery.schemas import Constraints, Product

faiss = pytest.importorskip("faiss")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from product_discovery.embeddings import (  # noqa: E402
    FaissTextEmbeddingIndex,
    IndexMetadata,
    catalog_fingerprint,
)
from product_discovery.indexed_retrieval import (  # noqa: E402
    BM25Index,
    IndexedRetriever,
    rank_in_full_ordering,
    top_rows_by_score,
)

WORDS = "black leather wallet usb cable charger red running shoes men women kids phone case glass".split()
QUERIES = ["black leather wallet", "usb c cable cable", "red running shoes for women", "zzz unseen", "phone"]


def make_catalog(n: int = 40) -> list[Product]:
    rng = np.random.default_rng(7)
    products = []
    for index in range(n):
        words = rng.choice(WORDS, size=int(rng.integers(1, 7)))
        title = " ".join(words)
        if index % 9 == 0:
            title = "Black Leather Wallet"  # duplicate texts create exact BM25 ties
        products.append(Product(id=f"p{index:03d}", title=title, brand="Acme" if index % 2 else None))
    return sorted(products, key=lambda row: row.id)


class FakeEncoder:
    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions

    def vector(self, text: str) -> np.ndarray:
        seed = sum(ord(character) * (position + 1) for position, character in enumerate(text)) % (2**32)
        vector = np.random.default_rng(seed).normal(size=self.dimensions).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def encode(self, texts, normalize_embeddings=True, **_):
        return np.vstack([self.vector(text) for text in texts])


def build(products, fusion_depth=1000):
    encoder = FakeEncoder()
    vectors = encoder.encode([product_text(p) for p in products])
    metadata = IndexMetadata(
        model_name="test/fake", dimensions=vectors.shape[1], product_count=len(products),
        catalog_fingerprint=catalog_fingerprint(products),
    )
    dense = FaissTextEmbeddingIndex.from_vectors([p.id for p in products], vectors, metadata)
    dense._embedder = encoder
    retriever = IndexedRetriever(
        tokenize=tokens,
        bm25=BM25Index.build([product_text(p) for p in products]),
        faiss_index=dense.index,
        encode_query=lambda query: encoder.encode([query])[0],
        vectors=vectors,
        fusion_depth=fusion_depth,
    )
    return dense, retriever


def test_bm25_scores_are_bit_identical_to_reference():
    products = make_catalog()
    index = BM25Index.build([product_text(p) for p in products])
    for query in QUERIES:
        expected = bm25_scores(query, products)
        actual = index.scores(tokens(query))
        assert [expected[p.id] for p in products] == actual.tolist()


@pytest.mark.parametrize("strategy", ["bm25", "dense", "hybrid"])
@pytest.mark.parametrize("k", [1, 5, 10, 40])
def test_indexed_rankings_equal_reference_retrieve(strategy, k):
    products = make_catalog()
    dense, retriever = build(products)
    for query in QUERIES:
        expected = retrieve(query, products, Constraints(), strategy=strategy, limit=k, dense_index=dense)
        actual = retriever.search(strategy, query, k)
        assert [products[row].id for row in actual.rows] == [p.id for p in expected.products]


def test_truncated_hybrid_fusion_is_exact_with_full_rank_lookup():
    products = make_catalog(40)
    dense, retriever = build(products, fusion_depth=20)
    for query in QUERIES:
        expected = retrieve(query, products, Constraints(), strategy="hybrid", limit=5, dense_index=dense)
        actual = retriever.hybrid_search(query, 5)
        assert [products[row].id for row in actual.rows] == [p.id for p in expected.products]


def test_rank_helpers_break_ties_by_row():
    scores = np.array([0.0, 2.0, 1.0, 2.0, 0.0, 3.0])
    assert top_rows_by_score(scores, 4).tolist() == [5, 1, 3, 2]
    assert rank_in_full_ordering(scores, np.array([5, 1, 3, 2, 0, 4])).tolist() == [1, 2, 3, 4, 5, 6]


def test_index_rejects_catalog_mismatch():
    products = make_catalog()
    dense, _ = build(products)
    changed = products[:-1] + [products[-1].model_copy(update={"title": "different"})]
    with pytest.raises(Exception, match="fingerprint"):
        dense.validate_catalog_fingerprint(catalog_fingerprint(changed))
