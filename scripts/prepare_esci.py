"""Prepare a provenance-rich Amazon Shopping Queries ESCI evaluation corpus.

Official test queries remain final-test-only. Train and validation are a
deterministic query-ID partition of official non-test rows. The output catalog is
persisted separately from labels, and its scope and cross-split product overlap
are recorded rather than hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ESCI_LABELS = {"E", "S", "C", "I"}


def find_parquet(root: Path, stem: str) -> Path:
    matches = sorted(path for path in root.rglob("*.parquet") if stem in path.name.lower())
    if not matches:
        raise FileNotFoundError(f"Could not find a parquet file containing '{stem}' below {root}")
    return matches[0]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(directory: str | Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepared_split(source_split: object, query_id: object, seed: int) -> str:
    """Keep official test queries isolated; hash non-test queries into train/validation."""
    if str(source_split).lower() == "test":
        return "test"
    digest = int(hashlib.sha1(f"{seed}:{query_id}".encode()).hexdigest()[:8], 16) % 10
    return "validation" if digest == 0 else "train"


def validate_split_integrity(rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    """Validate label vocabulary and query-level train/validation/test separation."""
    query_splits: dict[str, str] = {}
    split_queries: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for row in rows:
        split = str(row["prepared_split"])
        query_id = str(row["query_id"])
        label = str(row["esci_label"])
        if split not in split_queries:
            raise ValueError(f"Unknown prepared split: {split}")
        if label not in ESCI_LABELS:
            raise ValueError(f"Unknown ESCI label: {label}")
        previous = query_splits.setdefault(query_id, split)
        if previous != split:
            raise ValueError(f"Query {query_id} leaks between {previous} and {split}")
        split_queries[split].add(query_id)
    if split_queries["train"] & split_queries["validation"]:
        raise ValueError("Train and validation queries overlap")
    if split_queries["train"] & split_queries["test"]:
        raise ValueError("Train and test queries overlap")
    if split_queries["validation"] & split_queries["test"]:
        raise ValueError("Validation and test queries overlap")
    return split_queries


def select_complete_query_groups(rows: list[dict[str, Any]], max_rows: int, seed: int) -> list[dict[str, Any]]:
    """Deterministically select complete query groups up to an approximate row budget."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)
    query_ids = sorted(
        by_query,
        key=lambda query_id: hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    for query_id in query_ids:
        group = by_query[query_id]
        if selected and len(selected) + len(group) > max_rows:
            continue
        selected.extend(group)
        if len(selected) >= max_rows:
            break
    if not selected and query_ids:
        selected.extend(by_query[query_ids[0]])
    return selected


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def clean(value: object) -> str:
    return "" if value is None or str(value) == "nan" else str(value)


def product_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["product_id"]),
        "title": clean(row.get("product_title")),
        "description": clean(row.get("product_description")),
        "bullet_points": clean(row.get("product_bullet_point")),
        "brand": clean(row.get("product_brand")) or None,
        "colour": clean(row.get("product_color")) or None,
        "locale": str(row["product_locale"]),
        "category": "unknown",
        "attributes": [],
        "price": None,
    }


def load_products_by_id(
    products_path: Path, product_ids: set[str], locale: str
) -> dict[str, dict[str, Any]]:
    """Read only product columns and retain IDs referenced by prepared pairs."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit('Install data dependencies: pip install -e ".[data]"') from exc
    columns = [
        "product_id",
        "product_locale",
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ]
    products: dict[str, dict[str, Any]] = {}
    parquet = pq.ParquetFile(products_path)
    for batch in parquet.iter_batches(columns=columns, batch_size=32_768):
        for row in batch.to_pylist():
            product_id = str(row["product_id"])
            if row["product_locale"] == locale and product_id in product_ids:
                product = product_record(row)
                existing = products.setdefault(product_id, product)
                if existing != product:
                    raise ValueError(f"Product {product_id} has inconsistent metadata")
    missing = product_ids - products.keys()
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Missing {len(missing)} prepared product IDs in product corpus, e.g. {preview}")
    return products


CATALOG_COLUMNS = ("id", "title", "description", "bullet_points", "brand", "colour", "locale")


def load_full_locale_catalog(products_path: Path, locale: str) -> dict[str, dict[str, Any]]:
    """Load every official product row for one locale (the evaluation corpus)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit('Install data dependencies: pip install -e ".[data]"') from exc
    table = pq.read_table(products_path, filters=[("product_locale", "=", locale)])
    products: dict[str, dict[str, Any]] = {}
    for batch in table.to_batches(max_chunksize=65_536):
        for row in batch.to_pylist():
            product = product_record(row)
            if products.setdefault(product["id"], product) != product:
                raise ValueError(f"Product {product['id']} has inconsistent metadata")
    return products


def write_catalog_parquet(path: Path, catalog: list[dict[str, Any]]) -> None:
    """Persist the catalog in Product-schema columns, sorted by ID (row order == ID order)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = [row["id"] for row in catalog]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("Catalog must be sorted by unique product ID")
    table = pa.table({column: [row[column] for row in catalog] for column in CATALOG_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def write_labels_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist every labelled query-product pair (separate from catalog documents)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ordered = sorted(rows, key=lambda row: (str(row["query_id"]), str(row["product_id"])))
    table = pa.table(
        {
            "example_id": [str(row["example_id"]) for row in ordered],
            "query_id": [str(row["query_id"]) for row in ordered],
            "query": [str(row["query"]) for row in ordered],
            "product_id": [str(row["product_id"]) for row in ordered],
            "esci_label": [str(row["esci_label"]) for row in ordered],
            "source_split": [str(row["split"]) for row in ordered],
            "prepared_split": [str(row["prepared_split"]) for row in ordered],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def lfs_pointer_oid(dataset_dir: Path, path: Path) -> dict[str, Any] | None:
    """Read the Git-LFS pointer committed for a data file, when the source is a clone."""
    try:
        relative = path.resolve().relative_to(dataset_dir.resolve())
        text = subprocess.run(
            ["git", "-C", str(dataset_dir), "show", f"HEAD:{relative.as_posix()}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    fields = dict(line.split(" ", 1) for line in text.strip().splitlines() if " " in line)
    if "oid" not in fields:
        return None
    return {"oid": fields["oid"].removeprefix("sha256:"), "size": int(fields.get("size", 0))}


def verified_source_file(dataset_dir: Path, path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    pointer = lfs_pointer_oid(dataset_dir, path)
    if pointer is not None and (pointer["oid"] != digest or pointer["size"] != path.stat().st_size):
        raise ValueError(f"{path} does not match its official Git-LFS pointer {pointer}")
    return {
        "path": str(path),
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "git_lfs_pointer": pointer,
        "matches_git_lfs_pointer": None if pointer is None else True,
    }


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row["esci_label"]) for row in rows)
    positives: dict[str, int] = defaultdict(int)
    for row in rows:
        if str(row["esci_label"]) in {"E", "S"}:
            positives[str(row["query_id"])] += 1
    query_ids = {str(row["query_id"]) for row in rows}
    return {
        "rows": len(rows),
        "query_ids": len(query_ids),
        "product_ids": len({str(row["product_id"]) for row in rows}),
        "label_counts": dict(sorted(labels.items())),
        "queries_with_known_e_or_s": len(positives),
        "queries_without_known_e_or_s": len(query_ids) - len(positives),
    }


def output_pair(row: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": str(row["example_id"]),
        "query_id": str(row["query_id"]),
        "query": str(row["query"]),
        "label": str(row["esci_label"]),
        "product": product,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/esci"))
    parser.add_argument("--catalog-output", type=Path, default=Path("data/processed/esci_us_products.parquet"))
    parser.add_argument("--labels-output", type=Path, default=Path("data/esci/labels.parquet"))
    parser.add_argument(
        "--catalog-scope",
        choices=("full_locale", "labelled_pairs"),
        default="full_locale",
        help="full_locale: every official product for the locale (evaluation corpus); "
        "labelled_pairs: only products referenced by the sampled SFT pairs (legacy JSON).",
    )
    parser.add_argument("--manifest-output", type=Path, default=Path("data/manifests/esci_split_manifest.json"))
    parser.add_argument("--locale", default="us")
    parser.add_argument("--max-train", type=int, default=10_000)
    parser.add_argument("--max-eval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit('Install data dependencies: pip install -e ".[data]"') from exc

    examples_path = find_parquet(args.dataset_dir, "examples")
    products_path = find_parquet(args.dataset_dir, "products")
    examples_file = verified_source_file(args.dataset_dir, examples_path)
    products_file = verified_source_file(args.dataset_dir, products_path)
    source_rows = int(pd.read_parquet(examples_path, columns=["example_id"]).shape[0])
    examples = pd.read_parquet(
        examples_path,
        columns=[
            "example_id",
            "query",
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "split",
        ],
    )
    examples = examples[
        (examples.product_locale.eq(args.locale)) & (examples.small_version.eq(1))
    ].copy()
    if examples.empty:
        raise ValueError(f"No {args.locale} small_version ESCI examples found")
    examples["prepared_split"] = [
        prepared_split(row.split, row.query_id, args.seed) for row in examples.itertuples(index=False)
    ]
    all_rows = examples.to_dict(orient="records")
    all_query_sets = validate_split_integrity(all_rows)
    if {str(row["query_id"]) for row in all_rows if str(row["split"]).lower() == "test"} != all_query_sets["test"]:
        raise ValueError("Prepared test queries must equal official test queries exactly")
    rows_by_split = {
        split: [row for row in all_rows if row["prepared_split"] == split]
        for split in ("train", "validation", "test")
    }

    selected_rows: dict[str, list[dict[str, Any]]] = {}
    for split, max_rows in {
        "train": args.max_train,
        "validation": args.max_eval,
        "test": args.max_eval,
    }.items():
        selected_rows[split] = select_complete_query_groups(rows_by_split[split], max_rows, args.seed)
    selected_query_sets = validate_split_integrity(
        row for rows in selected_rows.values() for row in rows
    )
    product_ids = {
        str(row["product_id"]) for rows in selected_rows.values() for row in rows
    }
    if args.catalog_scope == "full_locale":
        products = load_full_locale_catalog(products_path, args.locale)
        labelled_ids = {str(row["product_id"]) for row in all_rows}
        missing = labelled_ids - products.keys()
        if missing:
            raise ValueError(f"{len(missing)} labelled product IDs are absent from the {args.locale} corpus")
    else:
        products = load_products_by_id(products_path, product_ids, args.locale)
    write_labels_parquet(args.labels_output, all_rows)

    output_rows: dict[str, list[dict[str, Any]]] = {}
    for split, rows in selected_rows.items():
        output_rows[split] = [output_pair(row, products[str(row["product_id"])]) for row in rows]
        write_jsonl(args.output_dir / f"{split}.jsonl", output_rows[split])
    catalog = [products[product_id] for product_id in sorted(products)]
    if args.catalog_output.suffix == ".parquet":
        write_catalog_parquet(args.catalog_output, catalog)
    else:
        args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
        args.catalog_output.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
    from product_discovery.embeddings import catalog_fingerprint_from_records

    split_product_ids = {
        split: {str(row["product_id"]) for row in rows} for split, rows in selected_rows.items()
    }
    full_split_product_ids = {
        split: {str(row["product_id"]) for row in rows} for split, rows in rows_by_split.items()
    }
    output_files = {
        split: args.output_dir / f"{split}.jsonl" for split in ("train", "validation", "test")
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "amazon-science/esci-data",
            "dataset_revision": git_revision(args.dataset_dir),
            "repository": "https://github.com/amazon-science/esci-data",
            "license": "Apache-2.0",
            "examples_path": str(examples_path),
            "examples_sha256": examples_file["sha256"],
            "products_path": str(products_path),
            "products_sha256": products_file["sha256"],
            "files": {"examples": examples_file, "products": products_file},
            "examples_rows_all_locales": source_rows,
            "locale": args.locale,
            "subset": "small_version=1",
        },
        "preparation": {
            "seed": args.seed,
            "code_revision": git_revision(Path(__file__).resolve().parents[1]),
            "script_sha256": sha256_file(Path(__file__)),
            "split_policy": "official test queries preserved; non-test query IDs hashed into train/validation",
            "sampling_policy": "deterministic complete query groups up to each split row budget",
            "label_vocabulary": sorted(ESCI_LABELS),
        },
        "labels": {
            "path": str(args.labels_output),
            "sha256": sha256_file(args.labels_output),
            "scope": f"every {args.locale} small_version=1 labelled pair, all prepared splits",
            "relevance_definition": "E and S are retrieval-relevant; graded gains E=3, S=2, C=1, I=0",
            "splits": {split: split_summary(rows) for split, rows in rows_by_split.items()},
            "cross_split_labelled_product_overlap": {
                "train_validation": len(full_split_product_ids["train"] & full_split_product_ids["validation"]),
                "train_test": len(full_split_product_ids["train"] & full_split_product_ids["test"]),
                "validation_test": len(full_split_product_ids["validation"] & full_split_product_ids["test"]),
            },
        },
        "sft_samples": {
            split: {
                "rows": len(rows),
                "query_ids": len(selected_query_sets[split]),
                "product_ids": len(split_product_ids[split]),
                "label_counts": dict(sorted(Counter(str(row["esci_label"]) for row in rows).items())),
                "path": str(output_files[split]),
                "sha256": sha256_file(output_files[split]),
            }
            for split, rows in selected_rows.items()
        },
        "query_split_integrity": {
            "all_source_queries": {split: len(query_ids) for split, query_ids in all_query_sets.items()},
            "prepared_queries": {split: len(query_ids) for split, query_ids in selected_query_sets.items()},
            "query_sets_disjoint": True,
            "official_test_queries_final_only": True,
            "labels_validated": True,
        },
        "catalog": {
            "path": str(args.catalog_output),
            "sha256": sha256_file(args.catalog_output),
            "content_fingerprint": catalog_fingerprint_from_records(catalog),
            "content_fingerprint_definition": (
                "sha256 over products sorted by ID of id + NUL + retrieval.product_text(Product) + newline; "
                "identical to embeddings.catalog_fingerprint and independent of file encoding"
            ),
            "products": len(catalog),
            "scope": args.catalog_scope,
            "selection": (
                f"Every official {args.locale}-locale product in the ESCI product corpus (all labelled "
                "products of both small and large versions). Built without reading any label."
                if args.catalog_scope == "full_locale"
                else "Products referenced by sampled labelled pairs only (legacy scoped catalog)."
            ),
            "sft_sample_cross_split_product_overlap": {
                "train_validation": len(split_product_ids["train"] & split_product_ids["validation"]),
                "train_test": len(split_product_ids["train"] & split_product_ids["test"]),
                "validation_test": len(split_product_ids["validation"] & split_product_ids["test"]),
            },
            "limitation": (
                "The ESCI product corpus is the union of products that were judged for some ESCI query; it "
                "is not the full Amazon catalog. Product overlap across query splits is retained and reported "
                "because catalog documents are shared by all queries; test-query labels are not used for "
                "training, tuning, or system selection."
            ),
        },
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
