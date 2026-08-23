import numpy as np
import pytest

from product_discovery.embeddings import (
    FaissTextEmbeddingIndex,
    IndexMetadata,
    TextEmbeddingIndex,
    catalog_fingerprint,
)
from product_discovery.schemas import Product


def test_hash_fallback_persists_and_retrieves(tmp_path):
    products = [Product(id="a", title="Black hiking backpack", description="waterproof", category="backpack"), Product(id="b", title="Red sofa", description="linen", category="sofa")]
    index = TextEmbeddingIndex.build(products, allow_fallback=True)
    path = tmp_path / "vectors.npz"; index.save(path)
    loaded = TextEmbeddingIndex.load(path)
    assert loaded.model_name == "hash-fallback/not-semantic"
    assert loaded.search("waterproof backpack", 1)[0][0] == "a"


def test_faiss_index_persists_provenance_without_loading_an_embedding_model(tmp_path):
    faiss = pytest.importorskip("faiss")
    products = [Product(id="a", title="A"), Product(id="b", title="B")]
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
    metadata = IndexMetadata(
        model_name="test/model", dimensions=2, product_count=2,
        catalog_fingerprint=catalog_fingerprint(products),
    )
    persisted = FaissTextEmbeddingIndex(["a", "b"], index, metadata)
    path = tmp_path / "products.faiss"
    persisted.save(path)
    loaded = FaissTextEmbeddingIndex.load(path, products)
    assert loaded.metadata.version == metadata.version
