"""Validation-only classification of a Qwen adapter checkpoint (for model selection).

Runs the same inference path as the test evaluation (``LocalAdapterReranker``: NF4
emulated base, fp16, batched greedy generation, notebook prompt/parser) on the
validation selection pairs, and writes validation_predictions.jsonl plus
validation_metrics.json next to the adapter (or to ``--output-dir``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from product_discovery.local_reranker import LocalAdapterReranker, classification_prompt
from product_discovery.provenance import run_provenance, sha256_file
from product_discovery.reranker_training import validation_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_reranker_ablation.json")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(config["output_root"]) / config["run_id"]
    manifest = json.loads((run_dir / "training_data_manifest.json").read_text(encoding="utf-8"))
    pairs = manifest["validation_pairs"]
    if sha256_file(pairs["path"]) != pairs["sha256"]:
        raise SystemExit("Validation pairs changed since the manifest was written")
    rows = [json.loads(line) for line in Path(pairs["path"]).read_text(encoding="utf-8").splitlines()]
    generation = config["qwen"]["generation"]
    model = LocalAdapterReranker(
        args.adapter, base_model=config["qwen"]["base_model"], dtype=generation["dtype"],
        batch_size=generation["batch_size"], max_length=generation["max_length"],
        max_new_tokens=generation["max_new_tokens"], base_weights="nf4",
    )
    # Sort by prompt length for batching efficiency; results are re-aligned by index.
    prompts = [classification_prompt(row) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    started = time.perf_counter()
    results = model.generate([prompts[i] for i in order])
    seconds = time.perf_counter() - started
    by_index = dict(zip(order, results))
    predictions = [by_index[i].label for i in range(len(rows))]
    summary = validation_summary(rows, predictions)
    out = args.output_dir or Path(args.adapter)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "validation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({"query_id": row["query_id"], "product_id": row["product_id"], "gold": row["label"],
                                     "prediction": predictions[i], "parsed": by_index[i].parsed,
                                     "raw": by_index[i].raw}) + "\n")
    metrics = {
        "adapter": args.adapter,
        "adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors"),
        "validation_pairs_sha256": pairs["sha256"],
        "pairs": len(rows),
        "unparsed_generations": sum(not r.parsed for r in results),
        "seconds": round(seconds, 1),
        "pairs_per_second": round(len(rows) / seconds, 3),
        "inference": f"{model.device}, nf4-emulated base, {generation['dtype']}, batch {generation['batch_size']}, greedy",
        "validation": summary,
        "provenance": run_provenance(),
    }
    (out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"adapter": args.adapter, "macro_f1": summary["macro_f1"], "accuracy": summary["accuracy"],
                      "e_to_c_or_i_rate": summary["e_to_c_or_i_rate"],
                      "unjudged_predicted_e_or_s_rate": summary["unjudged_predicted_e_or_s_rate"],
                      "seconds": metrics["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
