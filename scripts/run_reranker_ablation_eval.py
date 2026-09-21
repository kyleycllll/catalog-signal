"""Fixed-candidate TEST evaluation of a frozen reranker on the historical candidate sets.

Uses exactly the saved hybrid top-40 candidates (SHA-256 checked) and exactly the 200
test queries of the historical Qwen evaluation (asserted against its per-query file).
No retrieval is run. Metrics come from ``run_esci_reranking_eval.ranking_metrics`` so
every system is scored by identical code.

* ``--kind qwen``: ``retrieval.rerank_with_sft`` with ``LocalAdapterReranker`` (the
  historical inference path, argmax label -> gain).
* ``--kind cross_encoder``: one forward pass per candidate; ranked two ways from the
  same probabilities: expected gain (pre-registered primary) and argmax label via
  ``rerank_with_sft`` (same interface as Qwen). Latency covers model + scoring.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from product_discovery.esci_catalog import load_queries, records_from_parquet, stable_sample
from product_discovery.evaluation import GAIN
from product_discovery.provenance import run_provenance, sha256_file
from product_discovery.reranker_training import expected_gain_order
from product_discovery.retrieval import rerank_with_sft
from product_discovery.schemas import Product, SessionState


def _historical_module():
    path = Path(__file__).with_name("run_esci_reranking_eval.py")
    spec = importlib.util.spec_from_file_location("run_esci_reranking_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_reranker_ablation.json")
    parser.add_argument("--kind", choices=["qwen", "cross_encoder"], required=True)
    parser.add_argument("--model", required=True, help="Adapter dir (qwen) or model dir (cross_encoder)")
    parser.add_argument("--name", required=True, help="System name used in output file names")
    parser.add_argument("--max-queries", type=int, default=None, help="Smoke runs only")
    args = parser.parse_args()
    historical = _historical_module()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
    test_cfg = config["test"]
    out_dir = Path(config["output_root"]) / config["run_id"] / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke = args.max_queries is not None
    suffix = "_smoke" if smoke else ""

    if sha256_file(test_cfg["fixed_candidates"]) != test_cfg["fixed_candidates_sha256"]:
        raise SystemExit("Fixed candidate file changed")
    fixed = {}
    for line in filter(None, Path(test_cfg["fixed_candidates"]).read_text(encoding="utf-8").split("\n")):
        row = json.loads(line)
        fixed[row["query_id"]] = row
    test_queries = [q for q in load_queries(base["labels"], "test") if q["query_id"] in fixed]
    queries = stable_sample(test_queries, test_cfg["max_queries"], config["seed"])
    baseline_dir = Path(base["output_root"]) / base["run_id"]
    historical_ids = [json.loads(line)["query_id"] for line in (baseline_dir / "reranking_per_query.jsonl").read_text(encoding="utf-8").split("\n") if line]
    if [q["query_id"] for q in queries] != historical_ids:
        raise SystemExit("Query sample differs from the historical reranking evaluation")
    if smoke:
        queries = queries[: args.max_queries]
    k, ks = test_cfg["candidate_k"], test_cfg["ndcg_ks"]
    needed = {pid for q in queries for pid in fixed[q["query_id"]]["product_ids"][:k]}
    records = records_from_parquet(base["catalog"], needed)
    if needed - records.keys():
        raise SystemExit("Candidate IDs missing from catalog")

    if args.kind == "qwen":
        from product_discovery.local_reranker import LocalAdapterReranker

        generation = config["qwen"]["generation"]
        model = LocalAdapterReranker(args.model, base_model=config["qwen"]["base_model"], dtype=generation["dtype"],
                                     batch_size=generation["batch_size"], max_length=generation["max_length"],
                                     max_new_tokens=generation["max_new_tokens"], base_weights="nf4")
        model_hash = sha256_file(Path(args.model) / "adapter_model.safetensors")
    else:
        from product_discovery.cross_encoder_reranker import CrossEncoderReranker

        model = CrossEncoderReranker(args.model, max_length=config["cross_encoder"]["max_length"])
        model_hash = sha256_file(Path(args.model) / "model.safetensors")
    state = SessionState(id="offline-evaluation")

    warm = queries[0]
    warm_products = [Product.model_validate(records[pid]) for pid in fixed[warm["query_id"]]["product_ids"][:8]]
    asyncio.run(model.rerank(warm["query"], warm_products))

    per_query_path = out_dir / f"{args.name}_per_query{suffix}.jsonl"
    latency_path = out_dir / f"{args.name}_latency{suffix}.csv"
    with per_query_path.open("w", encoding="utf-8") as handle, latency_path.open("w", newline="", encoding="utf-8") as lat:
        writer = csv.writer(lat)
        writer.writerow(["query_id", "candidates", "rerank_latency_ms"])
        for number, query in enumerate(queries, start=1):
            candidate_row = fixed[query["query_id"]]
            ids = candidate_row["product_ids"][:k]
            retrieval_scores = {pid: {"retrieval": float(s)} for pid, s in zip(ids, candidate_row["scores"][:k])}
            products = [Product.model_validate(records[pid]) for pid in ids]
            labels = query["labels"]
            record: dict[str, Any] = {"query_id": query["query_id"], "query": query["query"],
                                      "original_candidate_ids": ids, "known_labels": [labels.get(pid) for pid in ids]}
            started = time.perf_counter()
            if args.kind == "qwen":
                ranked = asyncio.run(rerank_with_sft(query["query"], products, retrieval_scores, state, model))
                elapsed = (time.perf_counter() - started) * 1000
                reranked = [row.product.id for row in ranked]
                predictions = {row.product.id: row.relevance_label.value for row in ranked}
                record["predictions"] = [predictions[pid] for pid in ids]
                variants = {"primary": reranked}
            else:
                label_map, gains, probs = model.score_products(query["query"], products)
                primary = expected_gain_order(products, retrieval_scores, gains)
                elapsed = (time.perf_counter() - started) * 1000

                class _Fixed:  # argmax-label variant through the production function, no second forward pass
                    async def rerank(self, q, ps):
                        from product_discovery.schemas import ModelPrediction, RelevanceLabel
                        return [ModelPrediction(product_id=p.id, label=RelevanceLabel(label_map[p.id])) for p in ps]

                argmax = [row.product.id for row in asyncio.run(rerank_with_sft(query["query"], products, retrieval_scores, state, _Fixed()))]
                record["predictions"] = [label_map[pid] for pid in ids]
                record["expected_gain"] = [round(gains[pid], 4) for pid in ids]
                record["probabilities_ESCI"] = [[round(x, 4) for x in probs[pid]] for pid in ids]
                variants = {"primary": primary, "argmax_label": argmax}
            for name, order in variants.items():
                if sorted(order) != sorted(ids):
                    raise RuntimeError("Reranking changed the candidate set")
            record["mapped_gains"] = [GAIN[p] for p in record["predictions"]]
            record["reranked_ids"] = variants["primary"]
            record["pre_rerank_metrics"] = historical.ranking_metrics(ids, ids, labels, ks)
            record["post_rerank_metrics"] = historical.ranking_metrics(variants["primary"], ids, labels, ks)
            if "argmax_label" in variants:
                record["reranked_ids_argmax_label"] = variants["argmax_label"]
                record["post_rerank_metrics_argmax_label"] = historical.ranking_metrics(variants["argmax_label"], ids, labels, ks)
            record["rerank_latency_ms"] = round(elapsed, 2)
            writer.writerow([query["query_id"], len(ids), f"{elapsed:.2f}"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if number % 10 == 0:
                print(f"{number}/{len(queries)} queries", flush=True)

    meta = {
        "system": args.name, "kind": args.kind, "model": args.model, "model_sha256": model_hash,
        "config_sha256": sha256_file(args.config), "fixed_candidates_sha256": test_cfg["fixed_candidates_sha256"],
        "queries": len(queries), "query_ids_equal_historical": True, "smoke": smoke,
        "per_query": str(per_query_path), "per_query_sha256": sha256_file(per_query_path),
        "latency_csv": str(latency_path), "device": str(model.device), "provenance": run_provenance(),
    }
    (out_dir / f"{args.name}_run{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
