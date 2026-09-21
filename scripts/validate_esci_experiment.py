"""Post-hoc integrity checks for a completed held-out ESCI retrieval + reranking run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from product_discovery.provenance import run_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_heldout.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config["output_root"]) / config["run_id"]
    retrieval = json.loads((out / "retrieval_results.json").read_text())
    reranking = json.loads((out / "reranking_results.json").read_text())
    manifest = json.loads(Path(config["manifest"]).read_text())
    labels = pq.read_table(config["labels"], columns=["query_id", "prepared_split", "source_split"]).to_pandas()
    test = set(labels[labels.source_split == "test"].query_id)
    validation = set(labels[labels.prepared_split == "validation"].query_id)
    train = set(labels[labels.prepared_split == "train"].query_id)
    fixed = {}
    for line in (out / "fixed_candidates_test.jsonl").read_text().splitlines():
        row = json.loads(line)
        fixed[row["query_id"]] = row
    selected = retrieval["selection"]["selected_system"]
    ranked = {}
    with (out / "retrieval_per_query.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "test" and row["system"] == selected:
                ranked[row["query_id"]] = row["ranked_product_ids"]
    rows = [json.loads(line) for line in (out / "reranking_per_query.jsonl").read_text().splitlines()]
    saved_classifier = [json.loads(line) for line in Path("reports/evidence/esci-predictions.jsonl").read_text().splitlines()]
    checks = {
        "prepared_test_equals_official_test": set(labels[labels.prepared_split == "test"].query_id) == test,
        "train_validation_test_query_disjoint": not (test & validation or test & train or validation & train),
        "system_selected_on_validation": retrieval["selection"]["split"] == "validation",
        "rrf_k_is_untuned_default_60": retrieval["indexes"]["rrf"]["k"] == 60,
        "fixed_candidates_equal_selected_system_test_rankings": len(fixed) == len(ranked)
        and all(fixed[q]["product_ids"] == ranked[q] for q in fixed),
        "fixed_candidate_queries_official_test_only": set(fixed) <= test,
        "reranking_queries_official_test_only": all(r["query_id"] in test for r in rows),
        "reranking_used_exact_fixed_candidate_ids": all(
            r["original_candidate_ids"] == fixed[r["query_id"]]["product_ids"][: config["reranking"]["candidate_k"]]
            for r in rows
        ),
        "reranking_is_permutation_of_candidates": all(
            r["error"] is None and sorted(r["reranked_ids"]) == sorted(r["original_candidate_ids"]) for r in rows
        ),
        "dense_index_fingerprint_matches_catalog_and_manifest": retrieval["indexes"]["dense"]["catalog_fingerprint"]
        == retrieval["data"]["catalog_content_fingerprint"]
        == manifest["catalog"]["content_fingerprint"],
        "dense_index_smoke_tests_pass": all(retrieval["indexes"]["smoke_test"].values()),
        "source_files_match_git_lfs_pointers": all(
            f["matches_git_lfs_pointer"] for f in manifest["source"]["files"].values()
        ),
        "results_carry_provenance": all(
            "provenance" in r and r["provenance"].get("git_commit") and "config_sha256" in r
            for r in (retrieval, reranking)
        ),
        "saved_classifier_eval_rows_official_test_only": all(p["query_id"] in test for p in saved_classifier),
    }
    report = {
        "provenance": run_provenance(),
        "checks": checks,
        "all_passed": all(checks.values()),
        "not_verifiable_from_artifacts": [
            "The exact training rows of the historical adapter run were not saved. The notebook builds training "
            "and validation data only from official non-test query IDs, and the saved held-out predictions are all "
            "official-test rows, but the historical run's training set cannot be re-derived byte-for-byte.",
        ],
        "not_used_on_test_queries": [
            "training", "prompt tuning", "hard-negative mining", "RRF tuning", "checkpoint selection",
            "retrieval system selection",
        ],
    }
    (out / "experiment_integrity.json").write_text(json.dumps(report, indent=2))
    for name, ok in checks.items():
        print("PASS" if ok else "FAIL", name)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
