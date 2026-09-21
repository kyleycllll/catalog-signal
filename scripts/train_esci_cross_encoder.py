"""Train the single discriminative baseline: MiniLM cross-encoder with a 4-way ESCI head.

Trains only on a TRAIN variant from training_data_manifest.json, evaluates every epoch
on the VALIDATION selection pairs, and keeps the epoch with the highest validation
macro-F1. Official test data is never read.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from product_discovery.cross_encoder_reranker import CrossEncoderReranker, cross_encoder_product_text
from product_discovery.evaluation import LABELS
from product_discovery.provenance import run_provenance, sha256_file
from product_discovery.reranker_training import validation_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_reranker_ablation.json")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ce = config["cross_encoder"]
    run_dir = Path(config["output_root"]) / config["run_id"]
    manifest = json.loads((run_dir / "training_data_manifest.json").read_text(encoding="utf-8"))
    variant = manifest["variants"][args.variant]
    if sha256_file(variant["path"]) != variant["sha256"]:
        raise SystemExit("Training data changed since the manifest was written")
    if sha256_file(manifest["validation_pairs"]["path"]) != manifest["validation_pairs"]["sha256"]:
        raise SystemExit("Validation pairs changed since the manifest was written")
    rows = [json.loads(line) for line in Path(variant["path"]).read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(line) for line in Path(manifest["validation_pairs"]["path"]).read_text(encoding="utf-8").splitlines()]
    if {row["query_id"] for row in rows} & {row["query_id"] for row in val_rows}:
        raise SystemExit("Train/validation query overlap")

    seed = config["seed"]
    torch.manual_seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ce["model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        ce["model"], num_labels=len(LABELS), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
    ).to(device)
    parameters = sum(p.numel() for p in model.parameters())

    steps_per_epoch = math.ceil(len(rows) / ce["batch_size"])
    total_steps = steps_per_epoch * ce["epochs"]
    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": ce["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0},
    ], lr=ce["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * ce["warmup_ratio"]), total_steps)
    label_index = {label: i for i, label in enumerate(LABELS)}
    texts = [cross_encoder_product_text(row) for row in rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "variant": args.variant, "config": args.config, "config_sha256": sha256_file(args.config),
        "cross_encoder": ce, "seed": seed, "device": device, "parameters": parameters,
        "steps_per_epoch": steps_per_epoch, "provenance": run_provenance(),
    }, indent=2), encoding="utf-8")
    (args.output_dir / "data_manifest.json").write_text(json.dumps({
        "variant": args.variant, **variant, "validation_pairs": manifest["validation_pairs"],
        "leakage_check": "train and validation query IDs disjoint (asserted); no test data read",
    }, indent=2), encoding="utf-8")

    epochs, log = [], []
    best = None
    started = time.perf_counter()
    for epoch in range(1, ce["epochs"] + 1):
        model.train()
        order = list(range(len(rows)))
        random.Random(f"{seed}:{epoch}").shuffle(order)
        running = 0.0
        for step in range(steps_per_epoch):
            batch = order[step * ce["batch_size"] : (step + 1) * ce["batch_size"]]
            encoded = tokenizer([rows[i]["query"] for i in batch], [texts[i] for i in batch], truncation="only_second",
                                max_length=ce["max_length"], padding=True, return_tensors="pt").to(device)
            target = torch.tensor([label_index[rows[i]["label"]] for i in batch], device=device)
            loss = torch.nn.functional.cross_entropy(model(**encoded).logits.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())
            if (step + 1) % 50 == 0:
                log.append({"epoch": epoch, "step": step + 1, "loss": running / 50, "elapsed_s": round(time.perf_counter() - started, 1)})
                print(json.dumps(log[-1]), flush=True)
                running = 0.0
        checkpoint = args.output_dir / f"epoch-{epoch}"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        scorer = CrossEncoderReranker(str(checkpoint), device=device, max_length=ce["max_length"])
        probs = scorer.probabilities([row["query"] for row in val_rows], val_rows)
        predictions = [LABELS[max(range(4), key=p.__getitem__)] for p in probs]
        summary = validation_summary(val_rows, predictions)
        with (checkpoint / "validation_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row, pred, p in zip(val_rows, predictions, probs):
                handle.write(json.dumps({"query_id": row["query_id"], "product_id": row["product_id"], "gold": row["label"],
                                         "prediction": pred, "probabilities_ESCI": [round(x, 5) for x in p]}) + "\n")
        epochs.append({"epoch": epoch, "path": str(checkpoint), "validation_macro_f1": summary["macro_f1"],
                       "validation_accuracy": summary["accuracy"],
                       "model_sha256": sha256_file(checkpoint / "model.safetensors"), "validation": summary})
        print(json.dumps({"epoch": epoch, "val_macro_f1": summary["macro_f1"], "val_acc": summary["accuracy"]}), flush=True)
        if best is None or summary["macro_f1"] > best["validation_macro_f1"]:
            best = epochs[-1]
        del scorer

    metrics = {
        "variant": args.variant, "training_seconds": round(time.perf_counter() - started, 1),
        "loss_log": log, "epochs": epochs, "selected": {"epoch": best["epoch"], "path": best["path"],
        "rule": ce["selection"], "validation_macro_f1": best["validation_macro_f1"]},
        "provenance": run_provenance(),
    }
    (args.output_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["selected"], indent=2))


if __name__ == "__main__":
    main()
