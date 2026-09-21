"""Record large run artifacts that are kept out of Git, plus compact tracked summaries.

For each large file this writes its SHA-256, size, row count, schema, and the command
that regenerates it to ``<run>/large_artifacts.json``. For ``retrieval_per_query.jsonl``
it also writes ``retrieval_per_query_metrics.csv`` (per-query metrics without ranked
ID lists), which is small enough to track and is enough to recompute every aggregate
and bootstrap interval in ``retrieval_results.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from product_discovery.provenance import run_provenance, sha256_file

LARGE = {
    "retrieval_per_query.jsonl": {
        "generated_by": "python scripts/run_esci_retrieval_benchmark.py --config configs/esci_heldout.json",
        "schema": {
            "split": "validation | test",
            "system": "bm25 | dense | hybrid",
            "query_id": "ESCI query_id (str)",
            "query": "query text",
            "known_labels": "number of judged products for the query",
            "known_e_or_s": "number of judged E/S products for the query",
            "ranked_product_ids": "top-40 product IDs in rank order",
            "retrieved_known_labels": "ESCI label or null (unjudged) per ranked position",
            "scores": "retrieval score per ranked position",
            "metrics": "reciprocal_rank, recall/hit_rate/precision/judged at 10 and 40",
            "latency_ms": "measured end-to-end retrieval wall time",
        },
        "why_not_in_git": "61 MB; regenerable from pinned data, indexes and code. The per-query metrics are tracked in retrieval_per_query_metrics.csv and the hybrid test top-40 IDs in fixed_candidates_test.jsonl.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("reports/experiments/esci-us-heldout-v1"))
    args = parser.parse_args()
    records = {}
    for name, info in LARGE.items():
        path = args.run_dir / name
        rows = 0
        metric_names: list[str] = []
        summary_path = args.run_dir / "retrieval_per_query_metrics.csv"
        with path.open(encoding="utf-8") as handle, summary_path.open("w", newline="", encoding="utf-8") as out:
            writer = None
            for line in handle:
                row = json.loads(line)
                rows += 1
                if writer is None:
                    metric_names = list(row["metrics"])
                    writer = csv.writer(out)
                    writer.writerow(["split", "system", "query_id", "known_labels", "known_e_or_s", *metric_names, "latency_ms"])
                writer.writerow([
                    row["split"], row["system"], row["query_id"], row["known_labels"], row["known_e_or_s"],
                    *(row["metrics"][m] for m in metric_names), row["latency_ms"],
                ])
        records[name] = {
            "expected_path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": rows,
            **info,
            "tracked_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        }
    output = {"provenance": run_provenance(), "untracked_large_artifacts": records}
    (args.run_dir / "large_artifacts.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
