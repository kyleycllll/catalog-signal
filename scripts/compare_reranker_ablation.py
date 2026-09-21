"""Combine all rerankers evaluated on the identical fixed hybrid top-40 test candidates.

Systems: hybrid order without reranking, the historical Qwen adapter (reused unchanged
from esci-us-heldout-v1), the locally retrained Qwen arms, and the cross-encoder.
Writes ranking/classification/latency comparisons with paired bootstrap intervals,
per-query NDCG@10 for every system, and a failure analysis.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from product_discovery.esci_catalog import records_from_parquet
from product_discovery.evaluation import ACCEPTABLE_LABELS, classification_metrics
from product_discovery.provenance import run_provenance, sha256_file

METRICS = [
    "ndcg@5_ideal_all_judged_unjudged0", "ndcg@10_ideal_all_judged_unjudged0", "mrr_unjudged0",
    "precision@5_unjudged0", "judged_pool_ndcg@5", "judged_pool_ndcg@10", "judged_pool_mrr",
]
PARAMETERS = {
    "qwen": "494M base (Qwen2.5-0.5B, 4-bit NF4 in serving) + 2.16M LoRA",
    "cross_encoder": "22.7M (MiniLM-L6 cross-encoder, fp32)",
}


def paired(a, b, samples: int, seed: int) -> dict:
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(samples)]
    return {"mean_difference": float(d.mean()), "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)), "improved": int((d > 1e-12).sum()),
            "worsened": int((d < -1e-12).sum()), "unchanged": int((np.abs(d) <= 1e-12).sum())}


def pushed_out(pre_ids, post_ids, labels_by_id, wanted=ACCEPTABLE_LABELS, k=5):
    before = [pid for pid in pre_ids[:k] if labels_by_id.get(pid) in wanted]
    return len(before), sum(pid not in post_ids[:k] for pid in before)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_reranker_ablation.json")
    parser.add_argument("--systems", nargs="*", default=[], help="name=kind pairs, e.g. qwen_B_hard_negative=qwen")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = json.loads(Path(config["base_config"]).read_text(encoding="utf-8"))
    run_dir = Path(config["output_root"]) / config["run_id"]
    test_dir = run_dir / "test"
    seed, samples = config["seed"], config["test"]["bootstrap_samples"]

    historical_path = Path(base["output_root"]) / base["run_id"] / "reranking_per_query.jsonl"
    hist = [json.loads(line) for line in historical_path.read_text().splitlines()]
    order = [row["query_id"] for row in hist]
    rows: dict[str, dict[str, dict]] = {"no_rerank": {}, "historical_qwen": {}}
    kinds = {"no_rerank": "none", "historical_qwen": "qwen"}
    for row in hist:
        labels = dict(zip(row["original_candidate_ids"], row["known_labels"]))
        rows["no_rerank"][row["query_id"]] = {"ids": row["original_candidate_ids"], "metrics": row["pre_rerank_metrics"],
                                              "labels": labels, "query": row["query"], "predictions": None, "latency": None}
        rows["historical_qwen"][row["query_id"]] = {"ids": row["reranked_ids"], "metrics": row["post_rerank_metrics"],
                                                    "predictions": row["qwen_predictions"], "latency": row["rerank_latency_ms"]}
    sources = {"historical_qwen": {"path": str(historical_path), "sha256": sha256_file(historical_path)}}
    for spec in args.systems:
        name, kind = spec.split("=")
        path = test_dir / f"{name}_per_query.jsonl"
        sources[name] = {"path": str(path), "sha256": sha256_file(path)}
        data = [json.loads(line) for line in path.read_text().splitlines()]
        if [r["query_id"] for r in data] != order:
            raise SystemExit(f"{name}: query IDs differ from the historical evaluation")
        for r in data:
            base_row = rows["no_rerank"][r["query_id"]]
            if r["original_candidate_ids"] != base_row["ids"] or r["pre_rerank_metrics"] != base_row["metrics"]:
                raise SystemExit(f"{name}: candidate set or pre-rerank metrics differ for {r['query_id']}")
        kinds[name] = kind
        rows[name] = {r["query_id"]: {"ids": r["reranked_ids"], "metrics": r["post_rerank_metrics"],
                                      "predictions": r["predictions"], "latency": r["rerank_latency_ms"]} for r in data}
        if kind == "cross_encoder":
            kinds[f"{name}_argmax_label"] = kind
            rows[f"{name}_argmax_label"] = {r["query_id"]: {"ids": r["reranked_ids_argmax_label"],
                                                            "metrics": r["post_rerank_metrics_argmax_label"],
                                                            "predictions": r["predictions"], "latency": r["rerank_latency_ms"]}
                                            for r in data}

    ref = rows["no_rerank"]
    summary: dict[str, dict] = {}
    for name, per in rows.items():
        entry: dict = {"kind": kinds[name]}
        entry["means"] = {m: float(np.mean([per[q]["metrics"][m] for q in order])) for m in METRICS}
        if name != "no_rerank":
            entry["vs_no_rerank"] = {m: paired([ref[q]["metrics"][m] for q in order], [per[q]["metrics"][m] for q in order], samples, seed) for m in METRICS}
        if name not in ("no_rerank", "historical_qwen"):
            h = rows["historical_qwen"]
            entry["vs_historical_qwen"] = {m: paired([h[q]["metrics"][m] for q in order], [per[q]["metrics"][m] for q in order], samples, seed) for m in METRICS}
        es_before = es_out = e_before = e_out = 0
        for q in order:
            labels = ref[q]["labels"]
            b, o = pushed_out(ref[q]["ids"], per[q]["ids"], labels)
            es_before += b; es_out += o
            b, o = pushed_out(ref[q]["ids"], per[q]["ids"], labels, {"E"})
            e_before += b; e_out += o
        entry["known_e_or_s_in_pre_top5"] = es_before
        entry["known_e_or_s_pushed_out_of_top5"] = es_out
        entry["known_e_in_pre_top5"] = e_before
        entry["known_e_pushed_out_of_top5"] = e_out
        if per[order[0]]["predictions"] is not None:
            gold, pred, unjudged = [], [], Counter()
            for q in order:
                for pid, p in zip(ref[q]["ids"], per[q]["predictions"]):
                    g = ref[q]["labels"].get(pid)
                    if g is None:
                        unjudged[p] += 1
                    else:
                        gold.append(g); pred.append(p)
            cls = classification_metrics(gold, pred)
            e_row = dict(zip("ESCI", cls["confusion_matrix"]["E"]))
            cls["e_to_c"], cls["e_to_i"] = e_row["C"], e_row["I"]
            total = sum(unjudged.values())
            cls["unjudged_prediction_share"] = {k: unjudged[k] / total for k in "ESCI"}
            entry["classification_on_judged_candidates"] = cls
            median_es = statistics.median(
                sum(p in ("E", "S") for p in per[q]["predictions"]) / len(per[q]["predictions"]) for q in order)
            entry["median_share_of_candidates_predicted_e_or_s"] = median_es
        if per[order[0]]["latency"] is not None:
            lat = [per[q]["latency"] for q in order]
            entry["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                                   "mean": float(np.mean(lat)), "per_candidate_p50": float(np.percentile(lat, 50)) / 40}
        entry["parameters"] = PARAMETERS.get(kinds[name], "n/a")
        summary[name] = entry

    # ---- failure analysis ----
    ids_needed = {pid for q in order for pid in ref[q]["ids"]}
    records = records_from_parquet(base["catalog"], ids_needed)
    per_query = []
    for q in order:
        query = ref[q]["query"]
        titles = " ".join((records[pid].get("title") or "").lower() for pid in ref[q]["ids"])
        tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 3]
        row = {
            "query_id": q, "query": query,
            "slices": {
                "has_digit": any(c.isdigit() for c in query),
                "short_query_le_2_tokens": len(query.split()) <= 2,
                "lexical_mismatch_token (typo/rare-entity proxy)": any(t not in titles for t in tokens),
                "pre_top1_is_known_E": ref[q]["labels"].get(ref[q]["ids"][0]) == "E",
            },
            "ndcg@10": {name: rows[name][q]["metrics"]["ndcg@10_ideal_all_judged_unjudged0"] for name in rows},
        }
        per_query.append(row)
    slice_names = list(per_query[0]["slices"])
    slices = {}
    for s in slice_names:
        members = [r for r in per_query if r["slices"][s]]
        slices[s] = {"queries": len(members), "mean_ndcg@10": {name: float(np.mean([r["ndcg@10"][name] for r in members])) if members else None for name in rows}}
    (run_dir / "per_query_ndcg10_all_systems.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_query))

    def extremes(a: str, b: str, n: int = 8):
        diffs = sorted(per_query, key=lambda r: r["ndcg@10"][b] - r["ndcg@10"][a])
        pick = lambda r: {"query": r["query"], a: round(r["ndcg@10"][a], 3), b: round(r["ndcg@10"][b], 3)}
        return {"largest_losses": [pick(r) for r in diffs[:n]], "largest_gains": [pick(r) for r in diffs[::-1][:n]]}

    names = list(rows)
    representative = ["habanero hot sauce cholula organic", "vsco collage poster", "xyliwhite mouthwash", "rue"]
    failure = {
        "slices": slices,
        "representative_queries": [{"query": r["query"], "ndcg@10": {k: round(v, 3) for k, v in r["ndcg@10"].items()}}
                                   for r in per_query if r["query"] in representative],
        "pairwise_extremes": {f"{b}_vs_{a}": extremes(a, b) for a, b in
                              [("no_rerank", n) for n in names if n != "no_rerank"] + [("historical_qwen", n) for n in names if n not in ("no_rerank", "historical_qwen")]},
    }
    output = {"provenance": run_provenance(), "config_sha256": sha256_file(args.config), "queries": len(order),
              "candidate_file_sha256": config["test"]["fixed_candidates_sha256"], "sources": sources,
              "systems": summary}
    (run_dir / "fixed_candidate_comparison.json").write_text(json.dumps(output, indent=2))
    (run_dir / "failure_analysis.json").write_text(json.dumps(failure, indent=2))
    for name, entry in summary.items():
        m = entry["means"]
        cls = entry.get("classification_on_judged_candidates")
        print(f"{name:40s} nDCG5 {m['ndcg@5_ideal_all_judged_unjudged0']:.4f} nDCG10 {m['ndcg@10_ideal_all_judged_unjudged0']:.4f} "
              f"MRR {m['mrr_unjudged0']:.4f} F1 {cls['macro_f1'] if cls else float('nan'):.3f} "
              f"p50 {entry.get('latency_ms', {}).get('p50', float('nan')):.0f}")


if __name__ == "__main__":
    main()
