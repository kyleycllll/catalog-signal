"""Freeze validation-only model selection for the reranker ablation (validation_selection.json).

Qwen: per arm, the checkpoint with the higher validation macro-F1 (ties -> later); both arms go to
test. Cross-encoder: the (variant, epoch) with the highest validation macro-F1. Also copies
validation metrics/predictions and training-run records into the tracked report directory.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from product_discovery.provenance import run_provenance, sha256_file

R = Path("reports/experiments/esci-reranker-ablation-v1")
KEEP = ["examples", "accuracy", "macro_f1", "per_label", "confusion_matrix", "confusion_matrix_labels", "e_to_c", "e_to_i",
        "e_to_c_or_i_rate", "unjudged_prediction_counts", "unjudged_predicted_e_or_s_rate", "unjudged_predicted_e_rate"]


def main() -> None:
    qwen = {"historical_qwen": "models/ablation/historical_validation"}
    for arm in ("A_balanced_random", "B_hard_negative"):
        for ckpt in ("checkpoint-312", "checkpoint-625"):
            qwen[f"qwen_{arm}/{ckpt}"] = f"models/ablation/qwen_{arm}/{ckpt}"
    table = {}
    for name, directory in qwen.items():
        metrics = json.loads(Path(directory, "validation_metrics.json").read_text())
        out = R / "validation" / name.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        for f in ("validation_metrics.json", "validation_predictions.jsonl"):
            shutil.copy(Path(directory, f), out / f)
        table[name] = {"model_sha256": metrics["adapter_sha256"], "pairs_per_second": metrics["pairs_per_second"],
                       "unparsed": metrics["unparsed_generations"], **{k: metrics["validation"][k] for k in KEEP}}
    for arm in ("A_balanced_random", "B_hard_negative"):
        run = json.loads(Path(f"models/ablation/ce_{arm}/training_metrics.json").read_text())
        for epoch in run["epochs"]:
            name = f"ce_{arm}/epoch-{epoch['epoch']}"
            out = R / "validation" / name.replace("/", "__")
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(epoch["path"], "validation_predictions.jsonl"), out / "validation_predictions.jsonl")
            table[name] = {"model_sha256": epoch["model_sha256"], **{k: epoch["validation"][k] for k in KEEP}}
        for kind in ("ce", "qwen"):
            target = R / "training_runs" / f"{kind}_{arm}"
            target.mkdir(parents=True, exist_ok=True)
            for f in ("run_config.json", "data_manifest.json", "training_metrics.json"):
                shutil.copy(Path(f"models/ablation/{kind}_{arm}/{f}"), target / f)

    def best_checkpoint(prefix: str) -> str:
        chosen = None
        for key, value in table.items():
            if key.startswith(prefix) and (chosen is None or value["macro_f1"] >= table[chosen]["macro_f1"]):
                chosen = key
        return chosen

    selected = {
        "qwen_A_balanced_random": best_checkpoint("qwen_A_balanced_random/"),
        "qwen_B_hard_negative": best_checkpoint("qwen_B_hard_negative/"),
        "cross_encoder": max((k for k in table if k.startswith("ce_")), key=lambda k: table[k]["macro_f1"]),
    }
    paths = {k: f"models/ablation/{v}" for k, v in selected.items()}
    hashes = {k: sha256_file(Path(p) / ("model.safetensors" if k == "cross_encoder" else "adapter_model.safetensors"))
              for k, p in paths.items()}
    output = {
        "provenance": run_provenance(), "frozen_before_test": True,
        "rules": {"qwen": "per arm, checkpoint with the higher validation macro-F1 (ties -> later); BOTH arms go to test (pre-registered: A = retrained historical data approach, B = hard-negative arm)",
                  "cross_encoder": "(variant, epoch) with the highest validation macro-F1"},
        "selected": selected, "selected_paths": paths, "selected_model_sha256": hashes,
        "validation_pairs_sha256": json.loads((R / "training_data_manifest.json").read_text())["validation_pairs"]["sha256"],
        "validation_table": table,
    }
    (R / "validation_selection.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
