"""Build the persisted MiniLM + FAISS IndexFlatIP text index with provenance metadata.

JSON catalogs (the application format) are built in one pass. Parquet catalogs
(the prepared ESCI evaluation corpus) are encoded in resumable chunks because a
full-locale catalog takes more than an hour to embed on a laptop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from product_discovery.data_pipeline import validate_catalog
from product_discovery.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    FaissTextEmbeddingIndex,
    IndexMetadata,
    build_image_index,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoder(model_name: str, device: str | None, dtype: str):
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if dtype == "float16":
        model.half()
    return model, device


def build_parquet_index(args: argparse.Namespace) -> None:
    from product_discovery.esci_catalog import load_catalog

    started = time.perf_counter()
    catalog = load_catalog(args.catalog)
    fingerprint = catalog.fingerprint()
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        expected = manifest["catalog"]["content_fingerprint"]
        if expected != fingerprint:
            raise SystemExit(f"Catalog fingerprint {fingerprint} differs from manifest {expected}")
    print(f"Loaded {len(catalog):,} products in {time.perf_counter() - started:.1f}s; fingerprint {fingerprint[:12]}")

    chunk_dir = Path(args.chunk_dir or f"{args.text_output}.chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "fingerprint.txt").write_text(fingerprint)
    model, device = encoder(args.embedding_model, args.device, args.dtype)
    chunks = range(0, len(catalog), args.chunk_size)
    for number, start in enumerate(chunks):
        path = chunk_dir / f"{number:05d}.npy"
        if path.exists():
            continue
        chunk_started = time.perf_counter()
        vectors = model.encode(
            catalog.texts[start : start + args.chunk_size],
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(path.with_suffix(".tmp.npy"), vectors)
        path.with_suffix(".tmp.npy").rename(path)
        elapsed = time.perf_counter() - chunk_started
        print(f"chunk {number + 1}/{len(chunks)}: {len(vectors)} vectors, {len(vectors) / elapsed:.0f}/s", flush=True)

    vectors = np.concatenate([np.load(chunk_dir / f"{number:05d}.npy") for number in range(len(chunks))])
    metadata = IndexMetadata(
        model_name=args.embedding_model,
        dimensions=int(vectors.shape[1]),
        product_count=len(catalog),
        catalog_fingerprint=fingerprint,
        encode_dtype=args.dtype,
        encode_device=device,
        max_seq_length=int(model.max_seq_length),
        built_at=datetime.now(timezone.utc).isoformat(),
        source_manifest=args.manifest,
        source_manifest_sha256=sha256_file(Path(args.manifest)) if args.manifest else None,
        catalog_path=str(args.catalog),
    )
    index = FaissTextEmbeddingIndex.from_vectors(catalog.ids, vectors, metadata)
    index.save(args.text_output)
    print(f"Indexed {len(catalog):,} products with FAISS at {args.text_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/processed/esci_products.json")
    parser.add_argument("--text-output", default="data/indexes/text_embeddings.faiss")
    parser.add_argument("--image-output", default="data/indexes/image_embeddings.npz")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--manifest", default=None, help="Data manifest whose catalog fingerprint must match")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--chunk-dir", default=None)
    parser.add_argument("--images", action="store_true")
    args = parser.parse_args()

    if str(args.catalog).endswith(".parquet"):
        build_parquet_index(args)
        return
    products = validate_catalog(args.catalog)
    FaissTextEmbeddingIndex.build(products, model_name=args.embedding_model).save(args.text_output)
    if args.images:
        build_image_index(products, args.image_output)
    print(f"Indexed {len(products)} products with FAISS at {args.text_output}")


if __name__ == "__main__":
    main()
