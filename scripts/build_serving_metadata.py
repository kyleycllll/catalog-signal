"""Build the memory-mapped product metadata artifact used by the search API.

Unlike the offline benchmark catalog loader, this script streams parquet batches
and does not materialize the 1.2M-product catalog in Python memory.
"""
from __future__ import annotations

import argparse

from product_discovery.embeddings import FaissTextEmbeddingIndex
from product_discovery.serving_catalog import build_serving_metadata, row_order_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/processed/esci_us_products.parquet")
    parser.add_argument("--dense-index", default="data/indexes/esci_us_minilm.faiss")
    parser.add_argument("--output", default="data/indexes/esci_us_product_metadata.arrow")
    args = parser.parse_args()

    dense = FaissTextEmbeddingIndex.load(args.dense_index)
    build_serving_metadata(
        args.catalog,
        args.output,
        expected_product_count=dense.metadata.product_count,
        expected_fingerprint=dense.metadata.catalog_fingerprint,
        expected_row_order_hash=row_order_hash(dense.ids),
    )
    print(f"Wrote memory-mapped serving metadata for {dense.metadata.product_count:,} products to {args.output}")


if __name__ == "__main__":
    main()
