"""Leakage-safe preparation for structured-intent LoRA examples."""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def product_group(row: dict) -> str:
    ids = row.get("product_ids", [])
    if not ids: raise ValueError("Every training row must declare at least one product_id")
    return "|".join(sorted(map(str, ids)))


def split_examples(rows: list[dict], seed: int = 42, train_fraction: float = .8, validation_fraction: float = .1) -> dict[str, list[dict]]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must be positive and leave a test partition")
    partitions = {"train": [], "validation": [], "test": []}
    # A product group deterministically enters exactly one split.
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{product_group(row)}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        target = "train" if bucket < train_fraction else "validation" if bucket < train_fraction + validation_fraction else "test"
        partitions[target].append(row)
    validate_no_leakage(partitions)
    return partitions


def validate_no_leakage(partitions: dict[str, list[dict]]) -> None:
    assigned: dict[str, str] = {}
    for split, rows in partitions.items():
        for row in rows:
            for product_id in row.get("product_ids", []):
                prior = assigned.setdefault(str(product_id), split)
                if prior != split: raise ValueError(f"Product {product_id} leaks between {prior} and {split}")


def prepare_file(source: str | Path, output_dir: str | Path, seed: int = 42) -> dict[str, int]:
    rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    parts = split_examples(rows, seed=seed); destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    for name, values in parts.items():
        (destination / f"{name}.jsonl").write_text("\n".join(json.dumps(value) for value in values) + ("\n" if values else ""), encoding="utf-8")
    manifest = {"source": str(source), "seed": seed, "counts": {key: len(value) for key, value in parts.items()}, "leakage_checked": True}
    (destination / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["counts"]
