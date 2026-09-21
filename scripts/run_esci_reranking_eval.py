"""Fixed-candidate reranking evaluation: retrieval order vs the same IDs after Qwen.

The candidate IDs come from ``fixed_candidates_test.jsonl``, written by the
retrieval benchmark for the system selected on validation queries. Qwen only
labels those candidates (E/S/C/I, pointwise generative classification) and
``retrieval.rerank_with_sft`` reorders them. No product is added or removed, which
is asserted for every query.

Metric families (ESCI judgments are partial):
* full-rank (``*_unjudged0``): all 40 positions kept, unjudged gain 0.
  - ``ndcg@k_ideal_all_judged``: ideal DCG from all known judgments for the query.
  - ``ndcg@k_ideal_candidates``: ideal DCG from the fixed candidate set's labels
    (how well the fixed set is ordered; identical denominator before/after).
  - ``mrr``: reciprocal original rank of the first known E/S.
  - ``precision@5``: known E/S in top 5 / 5 (unjudged non-relevant).
* judged-pool diagnostics (``judged_pool_*``): unjudged candidates removed after
  ranking. Reported separately and never called full NDCG.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from product_discovery.esci_catalog import load_queries, records_from_parquet, stable_sample
from product_discovery.evaluation import (
    ACCEPTABLE_LABELS,
    GAIN,
    LABELS,
    classification_metrics,
    judged_pool_labels,
    judged_pool_mrr,
    judged_pool_ndcg_at_k,
    judged_precision_at_k,
    ndcg_unjudged_as_zero,
    precision_at_k,
    reciprocal_rank,
)
from product_discovery.local_reranker import LocalAdapterReranker
from product_discovery.provenance import run_provenance, sha256_file
from product_discovery.retrieval import rerank_with_sft
from product_discovery.schemas import Product, SessionState


def ranking_metrics(ranked: list[str], candidates: list[str], labels: dict[str, str], ks: list[int]) -> dict[str, float]:
    relevant = {pid for pid, label in labels.items() if label in ACCEPTABLE_LABELS}
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"ndcg@{k}_ideal_all_judged_unjudged0"] = ndcg_unjudged_as_zero(ranked, labels, k)
        metrics[f"ndcg@{k}_ideal_candidates_unjudged0"] = ndcg_unjudged_as_zero(ranked, labels, k, candidates)
    metrics["mrr_unjudged0"] = reciprocal_rank(ranked, relevant)
    metrics["precision@5_unjudged0"] = precision_at_k(ranked, relevant, 5)
    pool = judged_pool_labels(ranked, labels)
    for k in ks:
        metrics[f"judged_pool_ndcg@{k}"] = judged_pool_ndcg_at_k(pool, k)
    metrics["judged_pool_mrr"] = judged_pool_mrr(pool)
    metrics["judged_precision@5"] = judged_precision_at_k(pool, 5)
    return metrics


def paired(pre: list[float], post: list[float], samples: int, seed: int) -> dict[str, float]:
    differences = np.asarray(post) - np.asarray(pre)
    rng = np.random.default_rng(seed)
    means = [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(samples)]
    return {
        "pre_mean": float(np.mean(pre)),
        "post_mean": float(np.mean(post)),
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "queries_improved": int((differences > 1e-12).sum()),
        "queries_worsened": int((differences < -1e-12).sum()),
        "queries_unchanged": int((np.abs(differences) <= 1e-12).sum()),
    }


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 2) if values else None,
        "p50_ms": round(float(np.percentile(values, 50)), 2) if values else None,
        "p95_ms": round(float(np.percentile(values, 95)), 2) if values else None,
        "max_ms": round(max(values), 2) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_heldout.json")
    parser.add_argument("--max-queries", type=int, default=None, help="Override for smoke runs only")
    parser.add_argument("--output-suffix", default="", help="Write to reranking*_<suffix> files (smoke runs)")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rerank_config = config["reranking"]
    output = Path(config["output_root"]) / config["run_id"]
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""

    retrieval = json.loads((output / "retrieval_results.json").read_text(encoding="utf-8"))
    candidates_info = retrieval["fixed_candidates"]
    if sha256_file(candidates_info["path"]) != candidates_info["sha256"]:
        raise SystemExit("Fixed candidate file changed since the retrieval benchmark wrote it")
    if retrieval["selection"]["split"] != "validation":
        raise SystemExit("Candidate system must be selected on validation queries")
    fixed = {}
    for line in Path(candidates_info["path"]).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        fixed[row["query_id"]] = row

    test_queries = [q for q in load_queries(config["labels"], "test") if q["query_id"] in fixed]
    limit = args.max_queries if args.max_queries is not None else rerank_config["max_queries"]
    queries = stable_sample(test_queries, limit, config["seed"])
    k = rerank_config["candidate_k"]
    needed = {pid for q in queries for pid in fixed[q["query_id"]]["product_ids"][:k]}
    records = records_from_parquet(config["catalog"], needed)
    if needed - records.keys():
        raise SystemExit("Some candidate IDs are absent from the evaluation catalog")

    model = LocalAdapterReranker(
        rerank_config["adapter"],
        base_model=rerank_config["base_model"],
        dtype=rerank_config["dtype"],
        batch_size=rerank_config["batch_size"],
        base_weights=rerank_config.get("base_weights", "full"),
    )
    state = SessionState(id="offline-evaluation")
    ks = rerank_config["ndcg_ks"]

    # Warm up MPS kernels on one query outside the timed/evaluated set.
    warm = queries[0]
    warm_products = [Product.model_validate(records[pid]) for pid in fixed[warm["query_id"]]["product_ids"][:8]]
    asyncio.run(model.rerank(warm["query"], warm_products))

    per_query_rows: list[dict[str, Any]] = []
    gold_judged: list[str] = []
    predicted_judged: list[str] = []
    unjudged_predictions: Counter = Counter()
    with (output / f"reranking_per_query{suffix}.jsonl").open("w", encoding="utf-8") as handle, (
        output / f"reranking_latency{suffix}.csv"
    ).open("w", newline="", encoding="utf-8") as latency_file:
        latency_writer = csv.writer(latency_file)
        latency_writer.writerow(["query_id", "candidates", "rerank_latency_ms", "ms_per_candidate"])
        for number, query in enumerate(queries, start=1):
            candidate_row = fixed[query["query_id"]]
            candidate_ids = candidate_row["product_ids"][:k]
            retrieval_scores = {
                pid: {"retrieval": float(score)} for pid, score in zip(candidate_ids, candidate_row["scores"][:k])
            }
            products = [Product.model_validate(records[pid]) for pid in candidate_ids]
            labels = query["labels"]
            record: dict[str, Any] = {
                "query_id": query["query_id"],
                "query": query["query"],
                "candidate_system": candidate_row["system"],
                "original_candidate_ids": candidate_ids,
                "known_labels": [labels.get(pid) for pid in candidate_ids],
                "known_e_or_s_total_for_query": sum(label in ACCEPTABLE_LABELS for label in labels.values()),
            }
            try:
                started = time.perf_counter()
                ranked = asyncio.run(rerank_with_sft(query["query"], products, retrieval_scores, state, model))
                elapsed_ms = (time.perf_counter() - started) * 1000
                reranked_ids = [row.product.id for row in ranked]
                if sorted(reranked_ids) != sorted(candidate_ids) or len(reranked_ids) != len(candidate_ids):
                    raise RuntimeError("Reranking changed the candidate set")
                predictions = {row.product.id: row.relevance_label.value for row in ranked}
                pre = ranking_metrics(candidate_ids, candidate_ids, labels, ks)
                post = ranking_metrics(reranked_ids, candidate_ids, labels, ks)
                for pid in candidate_ids:
                    if pid in labels:
                        gold_judged.append(labels[pid])
                        predicted_judged.append(predictions[pid])
                    else:
                        unjudged_predictions[predictions[pid]] += 1
                record.update(
                    {
                        "qwen_predictions": [predictions[pid] for pid in candidate_ids],
                        "mapped_gains": [GAIN[predictions[pid]] for pid in candidate_ids],
                        "reranked_ids": reranked_ids,
                        "reranked_known_labels": [labels.get(pid) for pid in reranked_ids],
                        "pre_rerank_metrics": pre,
                        "post_rerank_metrics": post,
                        "rerank_latency_ms": round(elapsed_ms, 2),
                        "error": None,
                    }
                )
                latency_writer.writerow(
                    [query["query_id"], len(candidate_ids), f"{elapsed_ms:.2f}", f"{elapsed_ms / len(candidate_ids):.2f}"]
                )
            except Exception as exc:  # recorded, never silently dropped
                record.update({"error": f"{type(exc).__name__}: {exc}"})
            per_query_rows.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if number % 10 == 0:
                print(f"{number}/{len(queries)} queries", flush=True)

    completed = [row for row in per_query_rows if row["error"] is None]
    metric_names = list(completed[0]["pre_rerank_metrics"].keys())
    comparisons = {
        name: paired(
            [row["pre_rerank_metrics"][name] for row in completed],
            [row["post_rerank_metrics"][name] for row in completed],
            rerank_config["bootstrap_samples"],
            config["seed"],
        )
        for name in metric_names
    }
    labels_by_query = {q["query_id"]: q["labels"] for q in queries}

    def oracle_order(row: dict[str, Any]) -> list[str]:
        labels = labels_by_query[row["query_id"]]
        ids = row["original_candidate_ids"]
        return sorted(ids, key=lambda pid: (-(GAIN[labels[pid]] if pid in labels else 0), ids.index(pid)))

    oracle = {
        name: float(np.mean([
            ranking_metrics(oracle_order(row), row["original_candidate_ids"], labels_by_query[row["query_id"]], ks)[name]
            for row in completed
        ]))
        for name in metric_names
    }
    classification = classification_metrics(gold_judged, predicted_judged) if gold_judged else None
    latencies = [row["rerank_latency_ms"] for row in completed]
    results = {
        "run_id": config["run_id"],
        "provenance": run_provenance(),
        "config": rerank_config,
        "config_sha256": sha256_file(args.config),
        "data_manifest": config["manifest"],
        "data_manifest_sha256": sha256_file(config["manifest"]),
        "fixed_candidates": candidates_info,
        "candidate_system": retrieval["selection"]["selected_system"],
        "candidate_system_selected_on": "validation",
        "model": {
            "base_model": rerank_config["base_model"],
            "adapter": rerank_config["adapter"],
            "adapter_sha256": sha256_file(Path(rerank_config["adapter"]) / "adapter_model.safetensors"),
            "task": "pointwise generative ESCI classification (E/S/C/I); not pairwise/listwise/cross-encoder",
            "gain_mapping": GAIN,
            "inference": f"local {model.device}, {rerank_config.get('base_weights')} base weights, {rerank_config['dtype']} compute + LoRA",
            "inference_fidelity": rerank_config.get("inference_fidelity"),
            "fidelity_evidence": "classifier_reproduction.json",
            "score_formula": rerank_config["score_formula"],
        },
        "queries": {
            "requested": len(queries),
            "completed": len(completed),
            "failed": len(per_query_rows) - len(completed),
            "sample_rule": rerank_config["query_sample"],
        },
        "candidate_set_invariant": "asserted per query: reranked IDs are a permutation of the fixed candidate IDs",
        "comparisons_pre_vs_post": comparisons,
        "oracle_reorder_of_fixed_candidates": {
            "description": "upper bound: fixed candidates sorted by known gain (unjudged=0); uses test labels, reference only",
            "metrics": oracle,
        },
        "qwen_on_judged_candidates": classification,
        "qwen_on_unjudged_candidates": dict(unjudged_predictions),
        "judged_candidate_fraction": len(gold_judged) / max(1, len(gold_judged) + sum(unjudged_predictions.values())),
        "latency_ms": {
            "rerank_per_query": latency_summary(latencies),
            "note": "wall time for Qwen labelling of all candidates (batched greedy generation) plus score/sort, one query at a time",
        },
    }
    (output / f"reranking_results{suffix}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({name: comparisons[name] for name in metric_names[:6]}, indent=2))



if __name__ == "__main__":
    main()
