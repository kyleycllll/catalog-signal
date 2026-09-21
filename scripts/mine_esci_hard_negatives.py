"""Mine retrieval-informed hard negatives from TRAIN queries only.

For a deterministic sample of prepared-train queries, run the exact hybrid retriever
that produced the fixed test candidates over the full 1.2M-product corpus and record
the rank of every *judged* product for that query within the top ``mining_k``. The
output keeps observed ESCI labels as-is; unjudged retrieved products are counted but
never labelled. Validation and test queries are never retrieved here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

from product_discovery.esci_catalog import load_catalog, load_queries, stable_sample
from product_discovery.indexed_retrieval import top_rows_by_score
from product_discovery.provenance import run_provenance, sha256_file


def _benchmark_module():
    path = Path(__file__).with_name("run_esci_retrieval_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_esci_retrieval_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def truncated_hybrid(retriever, query: str, k: int) -> list[int]:
    """RRF (k=60) over the BM25 top-``depth`` and dense top-``depth`` lists.

    Identical inputs to ``IndexedRetriever.hybrid_search``; the only difference is that a
    product missing from one list gets rank ``depth + 1`` for it instead of its exact
    full-catalog rank (which costs ~360 ms/query). Agreement with the exact hybrid is
    measured on a sample and recorded in the summary.
    """
    depth = retriever.fusion_depth
    bm25_top = top_rows_by_score(retriever.bm25.scores(retriever.tokenize(query)), depth)
    dense_top, _ = retriever._dense_top(retriever.encode_query(query), depth)
    bm25_rank = {int(row): rank for rank, row in enumerate(bm25_top, start=1)}
    dense_rank = {int(row): rank for rank, row in enumerate(dense_top, start=1)}
    fused = {
        row: 1.0 / (retriever.rrf_k + bm25_rank.get(row, depth + 1)) + 1.0 / (retriever.rrf_k + dense_rank.get(row, depth + 1))
        for row in set(bm25_rank) | set(dense_rank)
    }
    return sorted(fused, key=lambda row: (-fused[row], row))[:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_reranker_ablation.json")
    parser.add_argument("--max-queries", type=int, default=None, help="Override for smoke runs only")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="Merge shard outputs and write the summary")
    parser.add_argument("--agreement-queries", type=int, default=50,
                        help="Train queries on which truncated vs exact hybrid top-k agreement is measured")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
    mining = config["mining"]
    if mining["source_split"] != "train":
        raise SystemExit("Hard negatives may only be mined from TRAIN queries")
    output = Path(config["output_root"]) / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)

    train = load_queries(base["labels"], "train")
    held_out = {q["query_id"] for split in ("validation", "test") for q in load_queries(base["labels"], split)}
    if {q["query_id"] for q in train} & held_out:
        raise SystemExit("Train queries overlap validation/test queries")
    limit = args.max_queries if args.max_queries is not None else mining["max_queries"]
    queries = stable_sample(train, limit, config["seed"])
    if args.merge:
        merge(config, args, output, len(train), queries)
        return
    queries = queries[args.shard_index :: args.num_shards]

    manifest = json.loads(Path(base["manifest"]).read_text(encoding="utf-8"))
    catalog = load_catalog(base["catalog"])
    retrieval_config = {**base["retrieval"], "seed": base["seed"], "dense_index": base["dense_index"],
                        "bm25_index": base["bm25_index"]}
    retriever, _, encoder_device = _benchmark_module().load_retriever(retrieval_config, catalog, manifest)

    catalog.texts = []  # only needed for the fingerprint check; frees several GB
    k = mining["mining_k"]
    work = output / "work"
    work.mkdir(exist_ok=True)
    path = work / f"hard_negative_mining.shard{args.shard_index}of{args.num_shards}.jsonl"

    agreement = []
    if args.shard_index == 0:
        for query in queries[: args.agreement_queries]:
            exact = retriever.search(mining["system"], query["query"], k).rows
            approx = truncated_hybrid(retriever, query["query"], k)
            ranks = {row: rank for rank, row in enumerate(exact, start=1)}
            agreement.append({
                "overlap_at_k": len(set(exact) & set(approx)) / k,
                "overlap_at_40": len(set(exact[:40]) & set(approx[:40])) / 40,
                "identical_order_at_k": exact == approx,
                "judged_rank_equal": all(
                    ranks.get(row) == rank for rank, row in enumerate(approx, start=1)
                    if catalog.ids[row] in query["labels"] and row in ranks
                ),
            })
        (work / "truncated_vs_exact_agreement.json").write_text(json.dumps({
            "queries": len(agreement),
            "mean_overlap_at_k": sum(a["overlap_at_k"] for a in agreement) / len(agreement),
            "mean_overlap_at_40": sum(a["overlap_at_40"] for a in agreement) / len(agreement),
            "identical_order_at_k_fraction": sum(a["identical_order_at_k"] for a in agreement) / len(agreement),
            "judged_items_same_rank_fraction": sum(a["judged_rank_equal"] for a in agreement) / len(agreement),
            "per_query": agreement,
        }, indent=2), encoding="utf-8")
        print(f"agreement measured on {len(agreement)} queries", flush=True)

    totals: Counter = Counter()
    started = time.perf_counter()
    with path.open("w", encoding="utf-8") as handle:
        for number, query in enumerate(queries, start=1):
            ids = [catalog.ids[row] for row in truncated_hybrid(retriever, query["query"], k)]
            labels = query["labels"]
            judged = [
                {"product_id": pid, "label": labels[pid], "hybrid_rank": rank}
                for rank, pid in enumerate(ids, start=1)
                if pid in labels
            ]
            totals.update(item["label"] for item in judged)
            totals["unjudged"] += len(ids) - len(judged)
            handle.write(json.dumps({
                "query_id": query["query_id"],
                "query": query["query"],
                "split": "train",
                "retrieved": len(ids),
                "judged_in_top_k": judged,
                "known_labels_for_query": dict(Counter(labels.values())),
            }) + "\n")
            if number % 500 == 0:
                rate = number / (time.perf_counter() - started)
                print(f"{number}/{len(queries)} queries ({rate:.2f} q/s)", flush=True)

    (path.with_suffix(".done.json")).write_text(json.dumps({
        "queries": len(queries), "seconds": round(time.perf_counter() - started, 1),
        "encoder_device": encoder_device, "provenance": run_provenance(),
    }, indent=2), encoding="utf-8")
    print(f"shard {args.shard_index}/{args.num_shards} done: {len(queries)} queries", flush=True)


def merge(config, args, output: Path, train_available: int, queries) -> None:
    mining = config["mining"]
    work = output / "work"
    rows, shard_info = [], []
    for index in range(args.num_shards):
        shard = work / f"hard_negative_mining.shard{index}of{args.num_shards}.jsonl"
        rows.extend(json.loads(line) for line in shard.read_text(encoding="utf-8").split("\n") if line)
        shard_info.append(json.loads(shard.with_suffix(".done.json").read_text(encoding="utf-8")))
    expected = sorted(q["query_id"] for q in queries)
    rows.sort(key=lambda row: row["query_id"])
    if [row["query_id"] for row in rows] != expected:
        raise SystemExit("Merged shards do not cover the sampled train queries exactly once")
    path = output / "hard_negative_mining.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    totals: Counter = Counter()
    for row in rows:
        totals.update(item["label"] for item in row["judged_in_top_k"])
        totals["unjudged"] += row["retrieved"] - len(row["judged_in_top_k"])
    summary = {
        "provenance": run_provenance(),
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "source_split": "train",
        "train_queries_available": train_available,
        "queries_mined": len(rows),
        "leakage_check": "mined query IDs are prepared-train only and disjoint from all validation and test query IDs (asserted)",
        "system": mining["system"],
        "fusion": "truncated RRF k=60 over BM25 top-1000 and dense top-1000 (missing component rank = 1001); see truncated_vs_exact_agreement",
        "truncated_vs_exact_agreement": {key: value for key, value in json.loads((work / "truncated_vs_exact_agreement.json").read_text()).items() if key != "per_query"},
        "mining_k": mining["mining_k"],
        "judged_labels_in_top_k": {label: totals[label] for label in ("E", "S", "C", "I")},
        "unjudged_in_top_k": totals["unjudged"],
        "shards": shard_info,
        "wall_seconds_max_shard": max(info["seconds"] for info in shard_info),
        "output": str(path),
        "output_sha256": sha256_file(path),
    }
    (output / "hard_negative_mining_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("provenance", "shards")}, indent=2))


if __name__ == "__main__":
    main()
