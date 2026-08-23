"""Prepare the official Amazon Shopping Queries ESCI parquet files for SFT and search.

Example:
  python scripts/prepare_esci.py --dataset-dir C:/datasets/esci-data --max-train 25000
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def find_parquet(root: Path, stem: str) -> Path:
    matches = [path for path in root.rglob("*.parquet") if stem in path.name.lower()]
    if not matches:
        raise FileNotFoundError(f"Could not find a parquet file containing '{stem}' below {root}")
    return matches[0]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def product_record(row) -> dict:
    def clean(value) -> str:
        return "" if value is None or str(value) == "nan" else str(value)

    return {
        "id": str(row.product_id),
        "title": clean(row.product_title),
        "description": clean(row.product_description),
        "bullet_points": clean(row.product_bullet_point),
        "brand": clean(row.product_brand) or None,
        "colour": clean(row.product_color) or None,
        "locale": str(row.product_locale),
        "category": "unknown",
        "attributes": [],
        "price": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Cloned amazon-science/esci-data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/esci"))
    parser.add_argument("--catalog-output", type=Path, default=Path("data/processed/esci_products.json"))
    parser.add_argument("--locale", default="us")
    parser.add_argument("--max-train", type=int, default=25000)
    parser.add_argument("--max-eval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit('Install the data dependencies first: pip install -e ".[data]"') from exc

    examples_path = find_parquet(args.dataset_dir, "examples")
    products_path = find_parquet(args.dataset_dir, "products")
    examples = pd.read_parquet(examples_path)
    products = pd.read_parquet(products_path)
    examples = examples[examples.product_locale.eq(args.locale)]
    if "small_version" in examples:
        examples = examples[examples.small_version.eq(1)]
    products = products[products.product_locale.eq(args.locale)]
    merged = examples.merge(products, on=["product_locale", "product_id"], how="inner", validate="many_to_one")

    def split_name(row) -> str:
        if str(row.split).lower() == "test":
            return "test"
        digest = int(hashlib.sha1(f"{args.seed}:{row.query_id}".encode()).hexdigest()[:8], 16) % 10
        return "validation" if digest == 0 else "train"

    merged["prepared_split"] = merged.apply(split_name, axis=1)
    limits = {"train": args.max_train, "validation": args.max_eval, "test": args.max_eval}
    manifest = {
        "source": "amazon-science/esci-data",
        "locale": args.locale,
        "seed": args.seed,
        "labels": ["E", "S", "C", "I"],
        "split_policy": "official test split preserved; train/validation are deterministic query_id partitions",
        "splits": {},
    }
    sampled_frames = []
    for split, limit in limits.items():
        frame = merged[merged.prepared_split.eq(split)]
        # Balanced sampling makes minority Substitute/Complement examples visible during a small Colab run.
        per_label = max(1, limit // 4)
        parts = [group.sample(min(len(group), per_label), random_state=args.seed) for _, group in frame.groupby("esci_label")]
        sampled = pd.concat(parts).sample(frac=1, random_state=args.seed).head(limit)
        rows = [
            {
                "example_id": str(row.example_id),
                "query_id": str(row.query_id),
                "query": str(row.query),
                "label": str(row.esci_label),
                "product": product_record(row),
            }
            for row in sampled.itertuples(index=False)
        ]
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
        manifest["splits"][split] = {
            "rows": len(rows),
            "query_ids": int(sampled.query_id.nunique()),
            "label_counts": sampled.esci_label.value_counts().to_dict(),
        }
        sampled_frames.append(sampled)

    catalog_frame = pd.concat(sampled_frames).drop_duplicates("product_id")
    args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_output.write_text(json.dumps([product_record(row) for row in catalog_frame.itertuples(index=False)], ensure_ascii=False), encoding="utf-8")
    query_sets = {
        split: set(merged[merged.prepared_split.eq(split)].query_id.astype(str)) for split in limits
    }
    if query_sets["train"] & query_sets["validation"] or query_sets["train"] & query_sets["test"] or query_sets["validation"] & query_sets["test"]:
        raise RuntimeError("Query-level split leakage detected")
    manifest["query_leakage_checked"] = True
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
