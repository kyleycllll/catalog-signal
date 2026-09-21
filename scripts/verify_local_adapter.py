"""Check that local adapter inference reproduces the saved Colab classifier predictions.

Re-runs the base model and the saved QLoRA adapter on the exact 500 official-test
pairs in ``reports/evidence/esci-predictions.jsonl`` and reports agreement with the
saved predictions and the recomputed accuracy / macro-F1. It does no training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from product_discovery.evaluation import classification_metrics
from product_discovery.local_reranker import LocalAdapterReranker, classification_prompt
from product_discovery.provenance import run_provenance, sha256_file


def macro_f1(gold: list[str], predicted: list[str]) -> float:
    return classification_metrics(gold, predicted)["macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/evidence/esci-predictions.jsonl")
    parser.add_argument("--labels", default="data/esci/labels.parquet")
    parser.add_argument("--catalog", default="data/processed/esci_us_products.parquet")
    parser.add_argument("--adapter", default="models/esci-qwen-lora")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--base-weights", choices=("full", "nf4"), default="nf4")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    saved = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    example_ids = {row["example_id"] for row in saved}
    labels = {
        row["example_id"]: row
        for row in pq.read_table(args.labels).to_pylist()
        if row["example_id"] in example_ids
    }
    if len(labels) != len(saved):
        raise SystemExit("Some saved example IDs are absent from the prepared labels")
    if any(labels[row["example_id"]]["source_split"] != "test" for row in saved):
        raise SystemExit("Saved classifier evaluation rows are not all official-test rows")
    product_ids = {labels[row["example_id"]]["product_id"] for row in saved}
    products = {
        row["id"]: row
        for row in pq.read_table(args.catalog, filters=[("id", "in", sorted(product_ids))]).to_pylist()
    }
    prompts = []
    for row in saved:
        label_row = labels[row["example_id"]]
        if label_row["esci_label"] != row["gold"]:
            raise SystemExit(f"Gold label mismatch for example {row['example_id']}")
        product = products[label_row["product_id"]]
        prompts.append(
            classification_prompt(
                {
                    "query": label_row["query"],
                    "product_title": product["title"],
                    "product_brand": product["brand"],
                    "product_color": product["colour"],
                    "product_bullet_point": product["bullet_points"],
                    "product_description": product["description"],
                }
            )
        )

    gold = [row["gold"] for row in saved]
    report: dict = {
        "purpose": "Fidelity check of local adapter inference against saved Colab predictions",
        "provenance": run_provenance(),
        "inputs": {
            "predictions": args.predictions,
            "predictions_sha256": sha256_file(args.predictions),
            "adapter": args.adapter,
            "adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors"),
            "pairs": len(saved),
            "all_official_test": True,
        },
        "saved_colab": {
            "base": {"accuracy": classification_metrics(gold, [r["base_prediction"] for r in saved])["accuracy"],
                     "macro_f1": macro_f1(gold, [r["base_prediction"] for r in saved])},
            "adapter": {"accuracy": classification_metrics(gold, [r["prediction"] for r in saved])["accuracy"],
                        "macro_f1": macro_f1(gold, [r["prediction"] for r in saved])},
        },
        "local": {},
    }
    for name, adapter, saved_key in (("adapter", args.adapter, "prediction"), ("base", None, "base_prediction")):
        model = LocalAdapterReranker(
            adapter, dtype=args.dtype, batch_size=args.batch_size, base_weights=args.base_weights
        )
        started = time.perf_counter()
        results = model.generate(prompts)
        elapsed = time.perf_counter() - started
        predicted = [result.label for result in results]
        agreement = sum(p == row[saved_key] for p, row in zip(predicted, saved)) / len(saved)
        metrics = classification_metrics(gold, predicted)
        report["local"][name] = {
            "device": model.device,
            "dtype": args.dtype,
            "base_weights": args.base_weights,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics.get("confusion_matrix"),
            "agreement_with_saved_colab_predictions": agreement,
            "unparsed_generations": sum(not result.parsed for result in results),
            "seconds": round(elapsed, 2),
        }
        print(name, json.dumps(report["local"][name]))
        del model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
