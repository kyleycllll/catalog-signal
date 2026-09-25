import pytest

pa = pytest.importorskip("pyarrow")
ipc = pytest.importorskip("pyarrow.ipc")
pq = pytest.importorskip("pyarrow.parquet")

from product_discovery.embeddings import catalog_fingerprint
from product_discovery.schemas import Product
from product_discovery.serving_catalog import ArrowServingCatalog, build_serving_metadata, row_order_hash


def test_memory_mapped_catalog_materializes_only_requested_row(tmp_path):
    path = tmp_path / "products.arrow"
    ids = ["a", "b"]
    schema = pa.schema(
        [
            pa.field("title", pa.string()),
            pa.field("description", pa.string()),
            pa.field("bullet_points", pa.string()),
            pa.field("brand", pa.string()),
            pa.field("colour", pa.string()),
            pa.field("locale", pa.string()),
        ],
        metadata={
            b"product_count": b"2",
            b"catalog_fingerprint": b"fixture-fingerprint",
            b"row_order_sha256": row_order_hash(ids).encode(),
        },
    )
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["First product", "Second product"]),
            pa.array(["first description", "second description"]),
            pa.array([None, "bullet point"]),
            pa.array(["Acme", "Bravo"]),
            pa.array(["red", "blue"]),
            pa.array(["us", "us"]),
        ],
        schema=schema,
    )
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, schema) as writer:
        writer.write_batch(batch)

    catalog = ArrowServingCatalog(
        path,
        expected_product_count=2,
        expected_fingerprint="fixture-fingerprint",
        expected_row_order_hash=row_order_hash(ids),
    )

    product = catalog.product_at(1, "b")
    assert product.id == "b"
    assert product.title == "Second product"
    assert product.brand == "Bravo"


def test_metadata_builder_streams_validated_parquet_fixture(tmp_path):
    source = tmp_path / "products.parquet"
    target = tmp_path / "products.arrow"
    ids = ["a", "b"]
    products = [
        Product(id="a", title="First product", description="first description", brand="Acme", colour="red"),
        Product(id="b", title="Second product", bullet_points="bullet point", brand="Bravo", colour="blue"),
    ]
    pq.write_table(
        pa.table(
            {
                "id": ids,
                "title": [product.title for product in products],
                "description": [product.description for product in products],
                "bullet_points": [product.bullet_points for product in products],
                "brand": [product.brand for product in products],
                "colour": [product.colour for product in products],
                "locale": ["us", "us"],
            }
        ),
        source,
    )

    build_serving_metadata(
        source,
        target,
        expected_product_count=2,
        expected_fingerprint=catalog_fingerprint(products),
        expected_row_order_hash=row_order_hash(ids),
    )
    catalog = ArrowServingCatalog(
        target,
        expected_product_count=2,
        expected_fingerprint=catalog_fingerprint(products),
        expected_row_order_hash=row_order_hash(ids),
    )

    assert catalog.product_at(0, "a").title == "First product"
