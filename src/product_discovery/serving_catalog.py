"""Memory-mapped product metadata for indexed search serving.

The retrieval indexes already hold catalog-scale data.  Keeping a second Python
list of 1.2M ``Product`` objects and their text fields would exceed the intended
memory budget on a laptop.  This module instead memory-maps a compact Arrow IPC
artifact and materializes only the rows returned by the retriever.
"""
from __future__ import annotations

import bisect
import hashlib
import os
from pathlib import Path
from typing import Iterable, Mapping

from .retrieval import product_text
from .schemas import Product


METADATA_COLUMNS = ("title", "description", "bullet_points", "brand", "colour", "locale")


def row_order_hash(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for product_id in ids:
        digest.update(str(product_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class ArrowServingCatalog:
    """Read product metadata by catalog row without materializing the catalog."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_product_count: int,
        expected_fingerprint: str,
        expected_row_order_hash: str,
    ):
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for the memory-mapped serving catalog") from exc
        self.path = Path(path)
        if not self.path.exists():
            raise RuntimeError(
                f"Serving metadata is missing at {self.path}. "
                "Build it with scripts/build_serving_metadata.py."
            )
        self._source = pa.memory_map(str(self.path), "r")
        reader = ipc.RecordBatchFileReader(self._source)
        metadata = {key.decode(): value.decode() for key, value in (reader.schema.metadata or {}).items()}
        actual_count = int(metadata.get("product_count", -1))
        if actual_count != expected_product_count:
            raise RuntimeError("Serving metadata product count does not match the dense index")
        if metadata.get("catalog_fingerprint") != expected_fingerprint:
            raise RuntimeError("Serving metadata catalog fingerprint does not match the dense index")
        if metadata.get("row_order_sha256") != expected_row_order_hash:
            raise RuntimeError("Serving metadata row order does not match the dense index")
        if tuple(reader.schema.names) != METADATA_COLUMNS:
            raise RuntimeError("Serving metadata schema does not match the Product fields used by the API")
        self._batches = [reader.get_batch(index) for index in range(reader.num_record_batches)]
        self._offsets: list[int] = []
        total = 0
        for batch in self._batches:
            self._offsets.append(total)
            total += batch.num_rows
        if total != expected_product_count:
            raise RuntimeError("Serving metadata batch rows do not match its declared product count")
        self._product_count = total

    def __len__(self) -> int:
        return self._product_count

    def product_at(self, row: int, product_id: str) -> Product:
        if row < 0 or row >= self._product_count:
            raise IndexError(f"Catalog row {row} is outside the catalog")
        batch_index = bisect.bisect_right(self._offsets, row) - 1
        batch = self._batches[batch_index]
        offset = row - self._offsets[batch_index]
        values = {name: batch.column(index)[offset].as_py() for index, name in enumerate(METADATA_COLUMNS)}
        return Product.model_validate({"id": product_id, **values})

    def lexical_fields_at(self, row: int, product_id: str) -> Mapping[str, str | None]:
        """Read only the identifier fields needed for bounded lexical re-scoring."""
        if row < 0 or row >= self._product_count:
            raise IndexError(f"Catalog row {row} is outside the catalog")
        batch_index = bisect.bisect_right(self._offsets, row) - 1
        batch = self._batches[batch_index]
        offset = row - self._offsets[batch_index]
        return {
            name: batch.column(METADATA_COLUMNS.index(name))[offset].as_py()
            for name in ("title", "brand", "colour")
        }


def build_serving_metadata(
    catalog_path: str | Path,
    output_path: str | Path,
    *,
    expected_product_count: int,
    expected_fingerprint: str,
    expected_row_order_hash: str,
) -> None:
    """Stream parquet product fields into a memory-mappable serving artifact.

    This is an offline build step.  It validates the source against the existing
    retrieval indexes while processing bounded parquet batches, so it never builds
    a full in-memory product catalog.
    """
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to build serving metadata") from exc
    source = pq.ParquetFile(catalog_path)
    if source.metadata.num_rows != expected_product_count:
        raise RuntimeError("Source catalog product count does not match the dense index")
    available = set(source.schema_arrow.names)
    required = {"id", *METADATA_COLUMNS}
    if required - available:
        raise RuntimeError(f"Source catalog is missing fields: {sorted(required - available)}")
    schema = pa.schema(
        [source.schema_arrow.field(name) for name in METADATA_COLUMNS],
        metadata={
            b"product_count": str(expected_product_count).encode(),
            b"catalog_fingerprint": expected_fingerprint.encode(),
            b"row_order_sha256": expected_row_order_hash.encode(),
        },
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    fingerprint = hashlib.sha256()
    row_order = hashlib.sha256()
    rows = 0
    try:
        with pa.OSFile(str(temporary), "wb") as sink, ipc.new_file(sink, schema) as writer:
            for batch in source.iter_batches(columns=["id", *METADATA_COLUMNS], batch_size=50_000):
                values = batch.to_pydict()
                for row in range(batch.num_rows):
                    product_id = str(values["id"][row])
                    row_order.update(product_id.encode("utf-8"))
                    row_order.update(b"\n")
                    product = Product.model_validate(
                        {
                            "id": product_id,
                            "title": values["title"][row],
                            "description": values["description"][row] or "",
                            "bullet_points": values["bullet_points"][row],
                            "brand": values["brand"][row],
                            "colour": values["colour"][row],
                            "locale": values["locale"][row] or "us",
                        }
                    )
                    fingerprint.update(product_id.encode("utf-8"))
                    fingerprint.update(b"\0")
                    fingerprint.update(product_text(product).encode("utf-8"))
                    fingerprint.update(b"\n")
                arrays = [batch.column(batch.schema.get_field_index(name)) for name in METADATA_COLUMNS]
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
                rows += batch.num_rows
        if rows != expected_product_count:
            raise RuntimeError("Serving metadata row count differs from the dense index")
        if fingerprint.hexdigest() != expected_fingerprint:
            raise RuntimeError("Source catalog fingerprint does not match the dense index")
        if row_order.hexdigest() != expected_row_order_hash:
            raise RuntimeError("Source catalog row order does not match the dense index")
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
