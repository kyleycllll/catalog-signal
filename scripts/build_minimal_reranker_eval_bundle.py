#!/usr/bin/env python3
"""Build a compact, reproducible fixed-candidate reranker evaluation bundle.

The bundle intentionally contains only products reachable from a deterministic
subset of the saved hybrid candidate lists.  Product Parquet is scanned in
batches, so this utility never materializes the 1.2M-product catalog in RAM.
It is useful for a constrained external evaluation runtime where the complete
catalog and retrieval indexes are deliberately unavailable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PRODUCT_COLUMNS = ("id", "title", "description", "bullet_points", "brand", "colour", "locale")
LABEL_COLUMNS = ("query_id", "query", "product_id", "esci_label", "prepared_split")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-per-query", type=Path, required=True)
    parser.add_argument("--fixed-candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=40)
    args = parser.parse_args()

    if args.queries <= 0 or args.candidate_k <= 0:
        raise SystemExit("--queries and --candidate-k must be positive")

    historical_rows = list(read_jsonl(args.historical_per_query))[: args.queries]
    query_ids = [str(row["query_id"]) for row in historical_rows]
    if len(query_ids) != args.queries or len(set(query_ids)) != len(query_ids):
        raise SystemExit("Historical per-query file does not contain the requested unique query count")
    query_id_set = set(query_ids)

    fixed = {
        str(row["query_id"]): row
        for row in read_jsonl(args.fixed_candidates)
        if str(row["query_id"]) in query_id_set
    }
    if set(fixed) != query_id_set:
        raise SystemExit("Fixed candidate file is missing at least one requested query")
    product_ids = {
        str(product_id)
        for query_id in query_ids
        for product_id in fixed[query_id]["product_ids"][: args.candidate_k]
    }

    import pyarrow.parquet as pq

    labels: dict[str, dict[str, str]] = {query_id: {} for query_id in query_ids}
    queries: dict[str, str] = {}
    labels_file = pq.ParquetFile(args.labels)
    for batch in labels_file.iter_batches(columns=list(LABEL_COLUMNS), batch_size=50_000):
        for row in batch.to_pylist():
            query_id = str(row["query_id"])
            if query_id not in query_id_set or row["prepared_split"] != "test":
                continue
            query = str(row["query"])
            if query_id in queries and queries[query_id] != query:
                raise RuntimeError(f"Query {query_id} has inconsistent query text")
            queries[query_id] = query
            labels[query_id][str(row["product_id"])] = str(row["esci_label"])
    if set(queries) != query_id_set:
        missing = sorted(query_id_set - set(queries))[:5]
        raise SystemExit(f"Test labels are missing queries: {missing}")

    products: dict[str, dict[str, Any]] = {}
    catalog_file = pq.ParquetFile(args.catalog)
    for batch in catalog_file.iter_batches(columns=list(PRODUCT_COLUMNS), batch_size=50_000):
        columns = batch.to_pydict()
        for row_number, product_id in enumerate(columns["id"]):
            product_id = str(product_id)
            if product_id in product_ids:
                products[product_id] = {column: columns[column][row_number] for column in PRODUCT_COLUMNS}
    if set(products) != product_ids:
        missing = sorted(product_ids - set(products))[:5]
        raise SystemExit(f"Catalog is missing candidate products: {missing}")

    bundle_queries = []
    for query_id in query_ids:
        candidate_row = fixed[query_id]
        ids = [str(product_id) for product_id in candidate_row["product_ids"][: args.candidate_k]]
        scores = [float(score) for score in candidate_row["scores"][: args.candidate_k]]
        bundle_queries.append(
            {
                "query_id": query_id,
                "query": queries[query_id],
                "labels": labels[query_id],
                "candidates": [
                    {"product": products[product_id], "retrieval_score": score}
                    for product_id, score in zip(ids, scores, strict=True)
                ],
            }
        )

    bundle = {
        "format": "adaptive-product-search-fixed-candidate-rerank-v1",
        "queries": bundle_queries,
        "provenance": {
            "query_count": len(bundle_queries),
            "candidate_k": args.candidate_k,
            "historical_per_query_sha256": sha256_file(args.historical_per_query),
            "fixed_candidates_sha256": sha256_file(args.fixed_candidates),
            "labels_sha256": sha256_file(args.labels),
            "catalog_sha256": sha256_file(args.catalog),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(bundle, stream, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size, **bundle["provenance"]}, indent=2))


if __name__ == "__main__":
    main()
