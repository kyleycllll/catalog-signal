"""Run configured ESCI retrieval/reranking comparisons and write evidence artifacts.

This runner does not invent unavailable QLoRA results. If the remote adapter is not
configured, only retrieval systems are scored and requested reranker systems are
recorded as unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from product_discovery.data_pipeline import validate_catalog
from product_discovery.embeddings import FaissTextEmbeddingIndex
from product_discovery.evaluation import ranking_metrics_from_labels, retrieval_metrics
from product_discovery.model_client import FineTunedModelClient
from product_discovery.retrieval import rank_without_reranker, rerank_with_sft, retrieve
from product_discovery.schemas import Constraints, SessionState


def read_cases(path: Path) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = str(row["query_id"])
        group = groups.setdefault(query_id, {"query": row["query"], "labels": {}})
        if group["query"] != row["query"]:
            raise ValueError(f"Query {query_id} has inconsistent text")
        product_id, label = str(row["product"]["id"]), str(row["label"])
        previous = group["labels"].setdefault(product_id, label)
        if previous != label:
            raise ValueError(f"Query {query_id} has conflicting labels for product {product_id}")
    return [{"query_id": query_id, **value} for query_id, value in sorted(groups.items())]


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 2),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
    }


async def evaluate_system(
    name: str,
    strategy: str,
    rerank: bool,
    cases: list[dict[str, Any]],
    products,
    dense_index,
    candidate_k: int,
    client: FineTunedModelClient | None,
) -> dict[str, Any]:
    if rerank and client is None:
        raise RuntimeError("QLoRA reranking was requested but no FINETUNED_MODEL_URL is configured")
    if rerank:
        await client.health()
    query_local_labels: list[list[str]] = []
    global_ranked_ids: list[list[str]] = []
    global_relevant_ids: list[set[str]] = []
    retrieval_latencies: list[float] = []
    rerank_latencies: list[float] = []
    for case in cases:
        started = time.perf_counter()
        candidates = retrieve(
            case["query"],
            products,
            Constraints(),
            strategy=strategy,
            limit=candidate_k,
            dense_index=dense_index,
        )
        retrieval_latencies.append((time.perf_counter() - started) * 1000)
        if rerank:
            started = time.perf_counter()
            ranked = await rerank_with_sft(
                case["query"], candidates.products, candidates.scores, SessionState(id="offline-evaluation"), client
            )
            rerank_latencies.append((time.perf_counter() - started) * 1000)
        else:
            ranked = rank_without_reranker(candidates)
        ranked_ids = [row.product.id for row in ranked]
        labels = case["labels"]
        local = [labels[product_id] for product_id in ranked_ids if product_id in labels]
        if local:
            query_local_labels.append(local)
        global_ranked_ids.append(ranked_ids)
        global_relevant_ids.append({product_id for product_id, label in labels.items() if label in {"E", "S"}})
    if not query_local_labels:
        raise RuntimeError("No judged candidates were retrieved; cannot calculate ranking metrics")
    return {
        "name": name,
        "status": "completed",
        "strategy": strategy,
        "reranking": rerank,
        "sample": {"queries": len(cases), "queries_with_judged_retrieval": len(query_local_labels)},
        "metrics": {
            "query_local_judged_ranking": ranking_metrics_from_labels(query_local_labels),
            "global_candidate_coverage": retrieval_metrics(global_ranked_ids, global_relevant_ids),
        },
        "latency_ms": {
            "candidate_generation": latency_summary(retrieval_latencies),
            "qlora_reranking": latency_summary(rerank_latencies),
        },
    }


def summary_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| System | Status | Recall@5 | MRR | nDCG@10 | Candidate p50 | Rerank p50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        if row["status"] != "completed":
            lines.append(f"| {row['name']} | unavailable | — | — | — | — | — |")
            continue
        metrics = row["metrics"]["query_local_judged_ranking"]
        latency = row["latency_ms"]
        lines.append(
            "| {name} | completed | {recall:.4f} | {mrr:.4f} | {ndcg:.4f} | {candidate} ms | {rerank} ms |".format(
                name=row["name"],
                recall=metrics.get("recall_at_5", 0.0),
                mrr=metrics.get("mrr", 0.0),
                ndcg=metrics.get("ndcg_at_10", 0.0),
                candidate=latency["candidate_generation"]["p50_ms"] or "—",
                rerank=latency["qlora_reranking"]["p50_ms"] or "—",
            )
        )
    return "\n".join(lines) + "\n"


async def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    catalog_path = Path(config["catalog_path"])
    cases_path = Path(config["evaluation_jsonl"])
    products = validate_catalog(catalog_path)
    cases = read_cases(cases_path)
    systems = config.get("systems", [])
    if not systems:
        raise ValueError("Experiment config needs at least one system")
    needs_dense = any(system["strategy"] in {"dense", "hybrid"} for system in systems)
    dense_index = None
    if needs_dense:
        dense_index = FaissTextEmbeddingIndex.load(Path(config["dense_index_path"]), products)
    client = None
    if any(bool(system.get("rerank")) for system in systems):
        client = FineTunedModelClient(
            base_url=config.get("reranker_url"), api_key=config.get("reranker_api_key")
        )
    results = []
    for system in systems:
        try:
            results.append(
                await evaluate_system(
                    system["name"],
                    system["strategy"],
                    bool(system.get("rerank")),
                    cases,
                    products,
                    dense_index,
                    int(config.get("candidate_k", 40)),
                    client,
                )
            )
        except Exception as exc:  # Preserve the failed experiment as evidence rather than substituting a result.
            results.append(
                {
                    "name": system["name"],
                    "status": "unavailable",
                    "strategy": system["strategy"],
                    "reranking": bool(system.get("rerank")),
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
    index_metadata = getattr(dense_index, "metadata", None)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "dataset": {
            "catalog_path": str(catalog_path),
            "evaluation_jsonl": str(cases_path),
            "catalog_products": len(products),
            "queries": len(cases),
            "judgment_note": "Global coverage treats unjudged products as non-relevant; local ranking uses only retrieved judged products.",
        },
        "dense_index": None if index_metadata is None else {
            "model_name": index_metadata.model_name,
            "version": index_metadata.version,
            "dimensions": index_metadata.dimensions,
            "product_count": index_metadata.product_count,
            "path": str(config["dense_index_path"]),
            "bytes": Path(config["dense_index_path"]).stat().st_size,
        },
        "results": results,
    }
    training_metadata = config.get("training_metadata_path")
    if training_metadata and Path(training_metadata).exists():
        report["training_metadata"] = json.loads(Path(training_metadata).read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "comparison.md").write_text(summary_table(results), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/search_experiment.example.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config.get("output_dir", "reports/experiments/latest"))
    report = asyncio.run(run(config, output_dir))
    print((output_dir / "comparison.md").read_text(encoding="utf-8"))
    print(f"Evidence written to {output_dir}; completed={sum(row['status'] == 'completed' for row in report['results'])}")


if __name__ == "__main__":
    main()
