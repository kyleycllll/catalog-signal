"""Local dense retrieval indexes with explicit provenance.

The production index is FAISS-backed. The small hash index is retained only for
offline unit tests and must never be selected by the search API.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .retrieval import product_text
from .schemas import Product


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseIndexUnavailable(RuntimeError):
    """Raised when a requested production dense index cannot be used."""


def hash_embed(text: str, dimensions: int = 384) -> np.ndarray:
    """Deterministic test-only embedding; it is not a semantic model."""
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in text.lower().split():
        digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vector[digest % dimensions] += -1 if digest & 1 else 1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def catalog_fingerprint(products: Iterable[Product]) -> str:
    digest = hashlib.sha256()
    for product in sorted(products, key=lambda row: row.id):
        digest.update(product.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(product_text(product).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class IndexMetadata:
    model_name: str
    dimensions: int
    product_count: int
    catalog_fingerprint: str
    backend: str = "faiss.IndexFlatIP"

    @property
    def version(self) -> str:
        return f"{self.model_name}:{self.catalog_fingerprint[:12]}"


class FaissTextEmbeddingIndex:
    """A persisted normalized-vector FAISS index for product text."""

    def __init__(self, ids: list[str], index: object, metadata: IndexMetadata):
        self.ids = ids
        self.index = index
        self.metadata = metadata
        self._embedder: object | None = None

    @staticmethod
    def _faiss() -> object:
        try:
            import faiss
        except ImportError as exc:
            raise DenseIndexUnavailable(
                "FAISS is required for dense retrieval. Install the project's retrieval extra."
            ) from exc
        return faiss

    @staticmethod
    def _embedder_for(model_name: str) -> object:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseIndexUnavailable(
                "sentence-transformers is required to build or query the dense index."
            ) from exc
        return SentenceTransformer(model_name)

    @classmethod
    def build(
        cls,
        products: list[Product],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "FaissTextEmbeddingIndex":
        if not products:
            raise ValueError("Cannot build a dense index for an empty catalog")
        faiss = cls._faiss()
        embedder = cls._embedder_for(model_name)
        vectors = np.asarray(
            embedder.encode(
                [product_text(product) for product in products],
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or not vectors.shape[1]:
            raise DenseIndexUnavailable("Embedding model returned invalid vectors")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors))
        metadata = IndexMetadata(
            model_name=model_name,
            dimensions=int(vectors.shape[1]),
            product_count=len(products),
            catalog_fingerprint=catalog_fingerprint(products),
        )
        return cls([product.id for product in products], index, metadata)

    @staticmethod
    def metadata_path(path: str | Path) -> Path:
        target = Path(path)
        return target.with_suffix(target.suffix + ".json")

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._faiss().write_index(self.index, str(output))
        self.metadata_path(output).write_text(
            json.dumps({"ids": self.ids, "metadata": asdict(self.metadata)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, path: str | Path, products: list[Product] | None = None
    ) -> "FaissTextEmbeddingIndex":
        source = Path(path)
        if not source.exists() or not cls.metadata_path(source).exists():
            raise DenseIndexUnavailable(f"Dense index or metadata is missing at {source}")
        raw = json.loads(cls.metadata_path(source).read_text(encoding="utf-8"))
        metadata = IndexMetadata(**raw["metadata"])
        ids = [str(value) for value in raw["ids"]]
        if len(ids) != metadata.product_count:
            raise DenseIndexUnavailable("Dense index metadata product count does not match its ID mapping")
        if products is not None and metadata.catalog_fingerprint != catalog_fingerprint(products):
            raise DenseIndexUnavailable(
                "Dense index catalog fingerprint does not match the loaded catalog; rebuild the index."
            )
        index = cls._faiss().read_index(str(source))
        if int(index.d) != metadata.dimensions or int(index.ntotal) != len(ids):
            raise DenseIndexUnavailable("Dense index data does not match its persisted metadata")
        return cls(ids, index, metadata)

    def _query_vector(self, query: str) -> np.ndarray:
        if self._embedder is None:
            self._embedder = self._embedder_for(self.metadata.model_name)
        vector = np.asarray(
            self._embedder.encode([query], normalize_embeddings=True), dtype=np.float32
        )
        if vector.shape != (1, self.metadata.dimensions):
            raise DenseIndexUnavailable("Embedding model output dimension differs from the persisted index")
        return np.ascontiguousarray(vector)

    def search(
        self, query: str, top_k: int = 20, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if top_k < 1:
            return []
        # Fetch all values when filtering so hard constraints cannot be bypassed by an over-fetch cap.
        fetch_k = self.metadata.product_count if allowed_ids is not None else min(
            self.metadata.product_count, max(top_k, top_k * 8)
        )
        scores, positions = self.index.search(self._query_vector(query), fetch_k)
        result: list[tuple[str, float]] = []
        for position, score in zip(positions[0], scores[0]):
            if position < 0:
                continue
            product_id = self.ids[int(position)]
            if allowed_ids is None or product_id in allowed_ids:
                result.append((product_id, float(score)))
            if len(result) == top_k:
                break
        return result


class InMemoryDenseIndex:
    """Small deterministic vector index for unit tests and local smoke fixtures."""

    def __init__(self, ids: list[str], vectors: np.ndarray, query_vectors: dict[str, np.ndarray] | None = None):
        self.ids = ids
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-12)
        self.query_vectors = query_vectors or {}
        self.metadata = IndexMetadata(
            model_name="test/in-memory",
            dimensions=int(vectors.shape[1]),
            product_count=len(ids),
            catalog_fingerprint="test-index",
            backend="in-memory",
        )

    def search(
        self, query: str, top_k: int = 20, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        vector = self.query_vectors.get(query, hash_embed(query, self.vectors.shape[1]))
        vector = np.asarray(vector, dtype=np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        ordered = sorted(
            zip(self.ids, self.vectors @ vector), key=lambda row: (-float(row[1]), row[0])
        )
        return [
            (product_id, float(score))
            for product_id, score in ordered
            if allowed_ids is None or product_id in allowed_ids
        ][:top_k]


class TextEmbeddingIndex:
    """Backward-compatible NPZ test fixture index; not used by production retrieval."""

    def __init__(self, ids: list[str], vectors: np.ndarray, model_name: str):
        self.ids, self.vectors, self.model_name = ids, vectors.astype(np.float32), model_name

    @classmethod
    def build(
        cls, products: list[Product], model_name: str = DEFAULT_EMBEDDING_MODEL, allow_fallback: bool = False
    ) -> "TextEmbeddingIndex":
        if not allow_fallback:
            raise DenseIndexUnavailable(
                "TextEmbeddingIndex is test-only. Build a FaissTextEmbeddingIndex for production retrieval."
            )
        return cls(
            [product.id for product in products],
            np.vstack([hash_embed(product_text(product)) for product in products]),
            "hash-fallback/not-semantic",
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        query_vector = hash_embed(query, self.vectors.shape[1])
        scores = self.vectors @ query_vector
        positions = np.argsort(-scores)[:top_k]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in positions]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.asarray(self.ids), vectors=self.vectors, model_name=np.asarray(self.model_name))

    @classmethod
    def load(cls, path: str | Path) -> "TextEmbeddingIndex":
        archive = np.load(path, allow_pickle=False)
        return cls(archive["ids"].tolist(), archive["vectors"], str(archive["model_name"].item()))


def build_image_index(products: list[Product], output: str | Path, model_name: str = "openai/clip-vit-base-patch32") -> None:
    """Create CLIP image embeddings. Fails explicitly when paths or ML dependencies are absent."""
    image_products = [p for p in products if p.image_path and Path(p.image_path).exists()]
    if not image_products:
        raise RuntimeError("No local product images found; cannot create an image index.")
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError("Install torch, pillow, and transformers for CLIP image embeddings.") from error
    processor, model = CLIPProcessor.from_pretrained(model_name), CLIPModel.from_pretrained(model_name)
    vectors = []
    for product in image_products:
        inputs = processor(images=Image.open(product.image_path).convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            vector = model.get_image_features(**inputs)[0].cpu().numpy()
        vectors.append(vector / np.linalg.norm(vector))
    TextEmbeddingIndex([p.id for p in image_products], np.vstack(vectors), model_name).save(output)
