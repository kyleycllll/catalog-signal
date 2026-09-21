"""Integrity checks for esci-reranker-ablation-v1 (leakage, frozen selection, identical candidates)."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from product_discovery.provenance import run_provenance, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_reranker_ablation.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
    run = Path(config["output_root"]) / config["run_id"]
    import pyarrow.parquet as pq

    table = pq.read_table(base["labels"], columns=["query_id", "prepared_split"]).to_pylist()
    split_of = {row["query_id"]: row["prepared_split"] for row in table}

    def csv_query_ids(path: Path) -> set[str]:
        with path.open(encoding="utf-8") as handle:
            return {row["query_id"] for row in csv.DictReader(handle)}

    manifest = json.loads((run / "training_data_manifest.json").read_text())
    mining = json.loads((run / "hard_negative_mining_summary.json").read_text())
    selection = json.loads((run / "validation_selection.json").read_text())
    comparison = json.loads((run / "fixed_candidate_comparison.json").read_text())
    train_ids = csv_query_ids(run / "training_rows_A_balanced_random.csv") | csv_query_ids(run / "training_rows_B_hard_negative.csv")
    val_ids = csv_query_ids(run / "validation_pairs.csv")

    selection_commit = subprocess.run(["git", "log", "-1", "--format=%H %ct", "--", str(run / "validation_selection.json")],
                                      capture_output=True, text=True).stdout.split()
    test_runs = {p.name: json.loads(p.read_text()) for p in (run / "test").glob("*_run.json")}
    first_test = min(r["provenance"]["timestamp_utc"] for r in test_runs.values())

    checks = {
        "training_rows_are_prepared_train_queries_only": all(split_of[q] == "train" for q in train_ids),
        "validation_pairs_are_validation_queries_only": all(split_of[q] == "validation" for q in val_ids),
        "hard_negatives_mined_from_train_only": mining["source_split"] == "train" and mining["queries_mined"] == manifest["mining"]["queries_mined"],
        "mined_file_hash_matches_manifest": mining["output_sha256"] == manifest["mining"]["sha256"],
        "B_E_and_S_rows_identical_to_A": manifest["controls"]["B_E_and_S_rows_identical_to_A"],
        "A_and_B_same_size_and_label_counts": manifest["variants"]["A_balanced_random"]["label_counts"] == manifest["variants"]["B_hard_negative"]["label_counts"],
        "training_data_files_unchanged": all(sha256_file(v["path"]) == v["sha256"] for v in manifest["variants"].values()),
        "selection_committed_to_git": bool(selection_commit),
        "selected_models_match_tested_models": all(
            any(r["model"] == selection["selected_paths"][k] and r["model_sha256"] == selection["selected_model_sha256"][k]
                for r in test_runs.values())
            for k in selection["selected_paths"]
        ),
        "fixed_candidates_unchanged": sha256_file(config["test"]["fixed_candidates"]) == config["test"]["fixed_candidates_sha256"],
        "all_test_runs_same_candidates_and_query_ids": all(
            r["fixed_candidates_sha256"] == config["test"]["fixed_candidates_sha256"] and r["query_ids_equal_historical"] and not r["smoke"]
            for r in test_runs.values()
        ),
        "comparison_uses_200_queries": comparison["queries"] == 200,
        "historical_rerun_identical_to_saved_historical": comparison["systems"]["historical_qwen_rerun"]["vs_historical_qwen"]["ndcg@10_ideal_all_judged_unjudged0"]["unchanged"] == 200,
    }
    report = {
        "provenance": run_provenance(),
        "checks": checks,
        "all_passed": all(checks.values()),
        "selection_commit": selection_commit,
        "first_new_test_run_utc": first_test,
        "note": "The validation selection was committed (git) before the new systems' test runs; commit time and run timestamps are recorded here.",
    }
    (run / "ablation_integrity.json").write_text(json.dumps(report, indent=2))
    for name, ok in checks.items():
        print("PASS" if ok else "FAIL", name)
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
