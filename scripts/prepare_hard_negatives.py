"""Create matched ESCI baseline and hard-negative QLoRA training JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from product_discovery.embeddings import FaissTextEmbeddingIndex
from product_discovery.hard_negatives import build_training_variants


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/esci/train.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/indexes/esci_products.faiss"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/esci/training_variants"))
    parser.add_argument("--examples-per-label", type=int, default=6000)
    parser.add_argument("--max-per-query", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.train.read_text(encoding="utf-8").splitlines() if line.strip()]
    products = []
    seen = set()
    for row in rows:
        product = row["product"]
        if product["id"] not in seen:
            products.append(product)
            seen.add(product["id"])
    from product_discovery.schemas import Product

    index = FaissTextEmbeddingIndex.load(args.index, [Product.model_validate(row) for row in products])
    variants, manifest = build_training_variants(
        rows,
        index,
        seed=args.seed,
        examples_per_label=args.examples_per_label,
        max_per_query=args.max_per_query,
    )
    for name, values in variants.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", values)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
