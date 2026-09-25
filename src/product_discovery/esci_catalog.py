"""Columnar access to the prepared ESCI evaluation catalog and labels.

The benchmark catalog has more than a million products, so it is stored as a
Product-schema parquet file (sorted by product ID) rather than the application's
JSON catalog. Retrieval text is produced by the same ``product_text`` function
used by the application so BM25, embeddings and fingerprints stay consistent.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embeddings import catalog_fingerprint_from_texts
from .retrieval import product_text
from .schemas import Product

CATALOG_COLUMNS = ("id", "title", "description", "bullet_points", "brand", "colour", "locale")


@dataclass
class EsciCatalog:
    ids: list[str]
    texts: list[str]
    columns: dict[str, list[Any]]

    def __post_init__(self) -> None:
        if self.ids != sorted(self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError("Catalog rows must be sorted by unique product ID")
        self.row_of = {product_id: row for row, product_id in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def fingerprint(self) -> str:
        return catalog_fingerprint_from_texts(zip(self.ids, self.texts))

    def product_at(self, row: int) -> Product:
        """Materialize one product from metadata already retained in memory.

        Serving keeps the Product-schema columns once at startup and only creates
        Pydantic objects for the small result set.  This avoids a parquet scan for
        every request while avoiding 1.2M Pydantic model instances in memory.
        """
        if row < 0 or row >= len(self.ids):
            raise IndexError(f"Catalog row {row} is outside the catalog")
        def column(name: str, default: Any):
            values = self.columns.get(name)
            return values[row] if values is not None else default

        record = {
            "id": self.ids[row],
            "title": column("title", ""),
            "description": column("description", "") or "",
            "bullet_points": column("bullet_points", None),
            "brand": column("brand", None),
            "colour": column("colour", None),
            "locale": column("locale", "us") or "us",
        }
        return Product.model_validate(record)


def load_catalog(path: str | Path, keep_columns: tuple[str, ...] = ("title",)) -> EsciCatalog:
    """Load a prepared parquet catalog and derive retrieval text for every row.

    Only ``keep_columns`` are retained after text derivation, to bound memory on
    million-product catalogs; ``records_from_parquet`` recovers full rows on demand.
    """
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    ids: list[str] = []
    texts: list[str] = []
    kept: dict[str, list[Any]] = {column: [] for column in keep_columns}
    for batch in parquet.iter_batches(columns=list(CATALOG_COLUMNS), batch_size=50_000):
        columns = batch.to_pydict()
        for row in range(batch.num_rows):
            record = {column: columns[column][row] for column in CATALOG_COLUMNS}
            ids.append(str(record["id"]))
            texts.append(product_text(Product.model_validate(record)))
        for column in keep_columns:
            kept[column].extend(columns[column])
    return EsciCatalog(ids=ids, texts=texts, columns=kept)


def records_from_parquet(path: str | Path, product_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Full Product-schema records for selected IDs."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(CATALOG_COLUMNS), filters=[("id", "in", sorted(product_ids))])
    return {row["id"]: row for row in table.to_pylist()}


def load_queries(
    labels_path: str | Path, split: str
) -> list[dict[str, Any]]:
    """Return one record per query of a prepared split with its known ESCI labels."""
    import pyarrow.parquet as pq

    table = pq.read_table(labels_path, filters=[("prepared_split", "=", split)])
    groups: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    for row in table.to_pylist():
        query_id = row["query_id"]
        group = groups.setdefault(query_id, {"query_id": query_id, "query": row["query"], "split": split})
        if group["query"] != row["query"]:
            raise ValueError(f"Query {query_id} has inconsistent text")
        previous = labels[query_id].setdefault(row["product_id"], row["esci_label"])
        if previous != row["esci_label"]:
            raise ValueError(f"Query {query_id} has conflicting labels for {row['product_id']}")
    return [
        {**groups[query_id], "labels": labels[query_id]} for query_id in sorted(groups)
    ]


def stable_sample(queries: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """Deterministic query sample by sha256(seed:query_id), returned in query-ID order."""
    if limit is None or limit >= len(queries):
        return queries
    ordered = sorted(queries, key=lambda q: hashlib.sha256(f"{seed}:{q['query_id']}".encode()).hexdigest())
    return sorted(ordered[:limit], key=lambda q: q["query_id"])
