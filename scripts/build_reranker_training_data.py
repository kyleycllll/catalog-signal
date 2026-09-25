"""Build the matched reranker training variants and the validation selection set.

* ``A_balanced_random``: a class-balanced random baseline, ``rows_per_label`` random
  prepared-TRAIN pairs per ESCI label.
* ``B_hard_negative``: the *same* E and S rows as A; the I and C rows are replaced by
  retrieval-mined hard negatives (judged I/C products the hybrid retriever ranks in
  the top ``mining_k`` for that train query). Same size and label counts as A. If the
  mined supply for a label is short, the remainder is filled from A's random rows of
  that label and the fill count is recorded.
* ``validation_pairs``: judged products in the hybrid top-40 of a fixed sample of
  VALIDATION queries (the reranking distribution), plus the first
  ``unjudged_per_query`` unjudged candidates per query to measure how often a model
  labels unjudged candidates E/S. Used only for model selection.

No official-test query is read. Every leakage condition is asserted.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from product_discovery.esci_catalog import load_queries, records_from_parquet, stable_sample
from product_discovery.provenance import run_provenance, sha256_file

LABELS = ("E", "S", "C", "I")


def product_fields(record: dict) -> dict:
    return {
        "product_id": record["id"],
        "product_title": record.get("title") or "",
        "product_brand": record.get("brand") or "",
        "product_color": record.get("colour") or "",
        "product_bullet_point": record.get("bullet_points") or "",
        "product_description": record.get("description") or "",
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_identity_csv(path: Path, rows: list[dict], extra: tuple[str, ...] = ()) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "product_id", "label", *extra])
        for row in rows:
            writer.writerow([row["query_id"], row["product_id"], row["label"], *(row.get(name, "") for name in extra)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_reranker_ablation.json")
    parser.add_argument("--data-dir", type=Path, default=Path("data/esci/ablation"))
    parser.add_argument("--unjudged-per-query", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
    output = Path(config["output_root"]) / config["run_id"]
    seed = config["seed"]
    per_label = config["training_data"]["rows_per_label"]
    mining = config["mining"]

    splits = {split: load_queries(base["labels"], split) for split in ("train", "validation", "test")}
    ids = {split: {q["query_id"] for q in rows} for split, rows in splits.items()}
    assert not (ids["train"] & ids["validation"]) and not (ids["train"] & ids["test"]) and not (ids["validation"] & ids["test"])
    train_by_id = {q["query_id"]: q for q in splits["train"]}

    # ---- A: class-balanced random train pairs ----
    train_pairs = [
        {"query_id": q["query_id"], "query": q["query"], "product_id": pid, "label": label}
        for q in splits["train"] for pid, label in sorted(q["labels"].items())
    ]
    by_label = defaultdict(list)
    for row in train_pairs:
        by_label[row["label"]].append(row)
    variant_a = {label: random.Random(f"{seed}:{label}:A").sample(by_label[label], per_label) for label in LABELS}

    # ---- B: same E/S rows; I/C replaced by hybrid-mined hard negatives ----
    mined_rows = [json.loads(line) for line in (output / "hard_negative_mining.jsonl").read_text(encoding="utf-8").split("\n") if line]
    assert all(row["query_id"] in ids["train"] for row in mined_rows), "mined a non-train query"
    variant_b = {"E": variant_a["E"], "S": variant_a["S"]}
    supply: dict[str, int] = {}
    filled: dict[str, int] = {}
    rank_stats: dict[str, dict] = {}
    for label in mining["hard_negative_labels"]:
        pool = [
            {"query_id": row["query_id"], "query": row["query"], "product_id": item["product_id"],
             "label": label, "hybrid_rank": item["hybrid_rank"]}
            for row in mined_rows for item in row["judged_in_top_k"] if item["label"] == label
        ]
        for item in pool:
            assert train_by_id[item["query_id"]]["labels"][item["product_id"]] == label, "label changed"
        supply[label] = len(pool)
        pool.sort(key=lambda row: (row["hybrid_rank"], row["query_id"], row["product_id"]))
        chosen, per_query = [], Counter()
        for row in pool:
            if per_query[row["query_id"]] < mining["max_per_query_per_label"]:
                chosen.append(row)
                per_query[row["query_id"]] += 1
            if len(chosen) == per_label:
                break
        chosen_keys = {(row["query_id"], row["product_id"]) for row in chosen}
        fill = [row for row in variant_a[label] if (row["query_id"], row["product_id"]) not in chosen_keys][: per_label - len(chosen)]
        filled[label] = len(fill)
        ranks = sorted(row["hybrid_rank"] for row in chosen)
        rank_stats[label] = {"hard_rows": len(chosen), "median_hybrid_rank": ranks[len(ranks) // 2] if ranks else None,
                             "max_hybrid_rank": ranks[-1] if ranks else None,
                             "distinct_queries": len({row["query_id"] for row in chosen})}
        variant_b[label] = chosen + fill

    # ---- validation pairs from the baseline run's validation hybrid top-40 ----
    baseline_dir = Path(base["output_root"]) / base["run_id"]
    val_cfg = config["training_data"]["validation_pairs"]
    val_queries = stable_sample(splits["validation"], val_cfg["max_queries"], seed)
    wanted = {q["query_id"]: q for q in val_queries}
    val_rows = []
    with (baseline_dir / "retrieval_per_query.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "validation" or row["system"] != "hybrid" or row["query_id"] not in wanted:
                continue
            labels = wanted[row["query_id"]]["labels"]
            unjudged = 0
            for rank, pid in enumerate(row["ranked_product_ids"][:40], start=1):
                label = labels.get(pid)
                if label is None:
                    if unjudged >= args.unjudged_per_query:
                        continue
                    unjudged += 1
                val_rows.append({"query_id": row["query_id"], "query": row["query"], "product_id": pid,
                                 "label": label or "UNJUDGED", "hybrid_rank": rank})
    assert {row["query_id"] for row in val_rows} <= ids["validation"]
    assert len({row["query_id"] for row in val_rows}) == len(val_queries)

    # ---- attach product text and write ----
    variants = {
        "A_balanced_random": [row for label in LABELS for row in variant_a[label]],
        "B_hard_negative": [row for label in LABELS for row in variant_b[label]],
    }
    needed = {row["product_id"] for rows in variants.values() for row in rows} | {row["product_id"] for row in val_rows}
    records = records_from_parquet(base["catalog"], needed)
    assert needed <= records.keys()
    manifest_variants = {}
    for name, rows in variants.items():
        random.Random(f"{seed}:{name}:shuffle").shuffle(rows)
        full = [{**row, **product_fields(records[row["product_id"]]), "product_id": row["product_id"]} for row in rows]
        assert all(row["query_id"] in ids["train"] for row in full)
        assert len({(row["query_id"], row["product_id"]) for row in full}) == len(full), "duplicate pair"
        path = args.data_dir / f"{name}.jsonl"
        write_rows(path, full)
        write_identity_csv(output / f"training_rows_{name}.csv", full, ("hybrid_rank",))
        manifest_variants[name] = {
            "path": str(path), "sha256": sha256_file(path), "rows": len(full),
            "label_counts": dict(Counter(row["label"] for row in full)),
            "distinct_queries": len({row["query_id"] for row in full}),
            "identity_csv": str(output / f"training_rows_{name}.csv"),
            "identity_csv_sha256": sha256_file(output / f"training_rows_{name}.csv"),
        }
    val_full = [{**row, **product_fields(records[row["product_id"]]), "product_id": row["product_id"]} for row in val_rows]
    val_path = args.data_dir / "validation_pairs.jsonl"
    write_rows(val_path, val_full)
    write_identity_csv(output / "validation_pairs.csv", val_full, ("hybrid_rank",))

    same_es = all(
        [(r["query_id"], r["product_id"]) for r in variant_a[l]] == [(r["query_id"], r["product_id"]) for r in variant_b[l]]
        for l in ("E", "S")
    )
    a_keys = {(r["query_id"], r["product_id"]) for r in variants["A_balanced_random"]}
    b_keys = {(r["query_id"], r["product_id"]) for r in variants["B_hard_negative"]}
    manifest = {
        "provenance": run_provenance(),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "labels_sha256": sha256_file(base["labels"]),
        "catalog_sha256_manifest": json.loads(Path(base["manifest"]).read_text())["catalog"]["sha256"],
        "seed": seed,
        "source_split": "train (prepared, non-official-test query IDs)",
        "train_queries": len(splits["train"]),
        "train_pairs_available": len(train_pairs),
        "train_label_counts_available": dict(Counter(row["label"] for row in train_pairs)),
        "mining": {
            "file": str(output / "hard_negative_mining.jsonl"),
            "sha256": sha256_file(output / "hard_negative_mining.jsonl"),
            "system": mining["system"], "mining_k": mining["mining_k"],
            "queries_mined": len(mined_rows),
            "judged_hard_negative_supply": supply,
            "max_per_query_per_label": mining["max_per_query_per_label"],
            "filtering_rules": mining["policy"],
            "selected": rank_stats,
            "random_fill_rows_when_supply_short": filled,
        },
        "variants": manifest_variants,
        "controls": {
            "B_E_and_S_rows_identical_to_A": same_es,
            "rows_shared_between_A_and_B": len(a_keys & b_keys),
            "rows_in_B_not_in_A (hard negatives added in place of random negatives)": len(b_keys - a_keys),
            "rows_in_A_not_in_B (random negatives removed)": len(a_keys - b_keys),
        },
        "validation_pairs": {
            "path": str(val_path), "sha256": sha256_file(val_path), "rows": len(val_full),
            "queries": len(val_queries),
            "label_counts": dict(Counter(row["label"] for row in val_full)),
            "rule": f"all judged products in the validation hybrid top-40 plus the first {args.unjudged_per_query} unjudged per query (label UNJUDGED; excluded from classification metrics)",
            "source": str(baseline_dir / "retrieval_per_query.jsonl"),
        },
        "leakage_checks": {
            "train_validation_test_query_ids_disjoint": True,
            "training_rows_train_queries_only": True,
            "mined_queries_train_only": True,
            "validation_rows_validation_queries_only": True,
            "official_test_labels_read": "only to assert query-ID disjointness; no test pair is used",
            "hard_negative_labels_unchanged": True,
        },
    }
    (output / "training_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("mining", "variants", "controls", "validation_pairs")}, indent=2))


if __name__ == "__main__":
    main()
