"""Held-out ESCI retrieval benchmark: BM25 vs dense MiniLM/FAISS vs hybrid RRF.

* Catalog: the prepared full-locale ESCI product corpus (see the data manifest).
* Selection: the system whose fixed candidates are passed to the reranker is chosen
  on *validation* queries using the pre-registered rule in the config. Test
  queries are only scored, never used to choose anything.
* Metrics use original ranking positions. ESCI judgments are partial, so Recall@K
  is recall of *known judged E/S positives*, and Precision@K treats unjudged
  retrieved products as non-relevant (a lower bound). Judged@K reports how much of
  the top K carries any ESCI judgment for the query.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from product_discovery.embeddings import FaissTextEmbeddingIndex
from product_discovery.esci_catalog import load_catalog, load_queries, stable_sample
from product_discovery.evaluation import (
    ACCEPTABLE_LABELS,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from product_discovery.indexed_retrieval import BM25Index, IndexedRetriever
from product_discovery.provenance import run_provenance, sha256_file
from product_discovery.retrieval import tokens


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "max_ms": round(max(values), 3),
    }


def query_metrics(ranked_ids: list[str], labels: dict[str, str], ks: list[int]) -> dict[str, float]:
    relevant = {product_id for product_id, label in labels.items() if label in ACCEPTABLE_LABELS}
    metrics: dict[str, float] = {"reciprocal_rank": reciprocal_rank(ranked_ids, relevant)}
    for k in ks:
        metrics[f"recall_at_{k}"] = recall_at_k(ranked_ids, relevant, k)
        metrics[f"hit_rate_at_{k}"] = hit_rate_at_k(ranked_ids, relevant, k)
        metrics[f"precision_at_{k}"] = precision_at_k(ranked_ids, relevant, k)
        metrics[f"judged_at_{k}"] = sum(pid in labels for pid in ranked_ids[:k]) / max(1, min(k, len(ranked_ids)))
    return metrics


def paired_bootstrap(a: list[float], b: list[float], samples: int, seed: int) -> dict[str, float]:
    """Mean of (b - a) with a paired 95% bootstrap interval over queries."""
    differences = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    means = np.array(
        [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(samples)]
    )
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "queries_better": int((differences > 0).sum()),
        "queries_worse": int((differences < 0).sum()),
        "queries_tied": int((differences == 0).sum()),
    }


def load_retriever(config: dict[str, Any], catalog, manifest: dict[str, Any]):
    fingerprint = catalog.fingerprint()
    if fingerprint != manifest["catalog"]["content_fingerprint"]:
        raise SystemExit("Catalog content fingerprint does not match the data manifest")
    dense = FaissTextEmbeddingIndex.load(config["dense_index"])
    dense.validate_catalog_fingerprint(fingerprint)
    if dense.ids != catalog.ids:
        raise SystemExit("Dense index ID order differs from catalog row order")
    if dense.metadata.model_name != config["embedding_model"]:
        raise SystemExit("Dense index was built with a different embedding model")

    bm25_dir = Path(config["bm25_index"])
    if (bm25_dir / "metadata.json").exists():
        bm25 = BM25Index.load(bm25_dir, fingerprint)
    else:
        started = time.perf_counter()
        bm25 = BM25Index.build(catalog.texts)
        bm25.save(bm25_dir, fingerprint)
        print(f"Built BM25 index in {time.perf_counter() - started:.1f}s", flush=True)
    if bm25.document_count != len(catalog):
        raise SystemExit("BM25 index document count differs from catalog")

    # Tokenizer integrity: CountVectorizer rows must equal retrieval.tokens() counts.
    rng = np.random.default_rng(config["seed"])
    for row in rng.choice(len(catalog), size=min(2000, len(catalog)), replace=False):
        expected: dict[str, int] = {}
        for token in tokens(catalog.texts[int(row)]):
            expected[token] = expected.get(token, 0) + 1
        observed = bm25.matrix[int(row)].tocoo()
        inverse = {column: term for term, column in ((t, bm25.vocabulary[t]) for t in expected)}
        got = {inverse.get(int(c), f"<col {c}>"): int(v) for c, v in zip(observed.col, observed.data)}
        if got != expected:
            raise SystemExit(f"BM25 tokenization differs from retrieval.tokens for row {row}")

    import faiss

    vectors = faiss.rev_swig_ptr(dense.index.get_xb(), dense.index.ntotal * dense.index.d)
    vectors = np.asarray(vectors).reshape(dense.index.ntotal, dense.index.d)
    encoder = FaissTextEmbeddingIndex._embedder_for(dense.metadata.model_name)

    def encode_query(query: str) -> np.ndarray:
        vector = np.asarray(encoder.encode([query], normalize_embeddings=True), dtype=np.float32)
        if vector.shape != (1, dense.metadata.dimensions):
            raise RuntimeError("Query embedding dimension differs from the index")
        return vector[0]

    retriever = IndexedRetriever(
        tokenize=tokens,
        bm25=bm25,
        faiss_index=dense.index,
        encode_query=encode_query,
        vectors=vectors,
        rrf_k=config["rrf_k"],
        fusion_depth=config["fusion_depth"],
    )
    return retriever, dense, str(encoder.device)


def smoke_test(retriever: IndexedRetriever, catalog, dense) -> dict[str, Any]:
    checks = {
        "index_vector_count_equals_catalog": int(dense.index.ntotal) == len(catalog),
        "index_ids_equal_catalog_order": dense.ids == catalog.ids,
        "metadata_fingerprint_matches_catalog": dense.metadata.catalog_fingerprint == catalog.fingerprint(),
        "normalized": bool(dense.metadata.normalized),
    }
    sample = retriever.dense_search("wireless bluetooth headphones", 40)
    checks["dense_returns_k"] = len(sample.rows) == 40
    checks["returned_rows_in_catalog"] = all(0 <= row < len(catalog) for row in sample.rows)
    norms = np.linalg.norm(retriever.vectors[:: max(1, len(catalog) // 5000)], axis=1)
    checks["sampled_vector_norms_unit"] = bool(np.allclose(norms, 1.0, atol=1e-3))
    if not all(checks.values()):
        raise SystemExit(f"Dense index smoke test failed: {checks}")
    return checks


def evaluate_split(
    retriever: IndexedRetriever,
    catalog,
    queries: list[dict[str, Any]],
    systems: list[str],
    ks: list[int],
    retrieve_k: int,
    per_query_handle,
    latency_writer,
    split: str,
) -> dict[str, dict[str, Any]]:
    per_system: dict[str, dict[str, list]] = {s: {"metrics": [], "latency": [], "candidates": []} for s in systems}
    for warmup in queries[:10]:  # exclude model/allocator warm-up from latency statistics
        for system in systems:
            retriever.search(system, warmup["query"], retrieve_k)
    for number, query in enumerate(queries, start=1):
        for system in systems:
            started = time.perf_counter()
            result = retriever.search(system, query["query"], retrieve_k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            ranked_ids = [catalog.ids[row] for row in result.rows]
            metrics = query_metrics(ranked_ids, query["labels"], ks)
            per_system[system]["metrics"].append(metrics)
            per_system[system]["latency"].append(elapsed_ms)
            per_system[system]["candidates"].append(
                {"query_id": query["query_id"], "product_ids": ranked_ids, "scores": result.scores}
            )
            record = {
                "split": split,
                "system": system,
                "query_id": query["query_id"],
                "query": query["query"],
                "known_labels": len(query["labels"]),
                "known_e_or_s": sum(label in ACCEPTABLE_LABELS for label in query["labels"].values()),
                "ranked_product_ids": ranked_ids,
                "retrieved_known_labels": [query["labels"].get(pid) for pid in ranked_ids],
                "scores": [round(float(score), 6) for score in result.scores],
                "metrics": metrics,
                "latency_ms": round(elapsed_ms, 3),
            }
            if result.component_ranks is not None:
                record["component_ranks"] = result.component_ranks
            if result.timings_ms is not None:
                record["stage_latency_ms"] = {k: round(v, 3) for k, v in result.timings_ms.items()}
                per_system[system].setdefault("stages", []).append(result.timings_ms)
            per_query_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            latency_writer.writerow([split, system, query["query_id"], f"{elapsed_ms:.3f}"])
        if number % 500 == 0:
            print(f"{split}: {number}/{len(queries)} queries", flush=True)
    summary: dict[str, dict[str, Any]] = {}
    for system, rows in per_system.items():
        keys = rows["metrics"][0].keys()
        summary[system] = {
            "queries": len(rows["metrics"]),
            "metrics": {key: float(np.mean([m[key] for m in rows["metrics"]])) for key in keys},
            "latency_ms": latency_summary(rows["latency"]),
            "stage_latency_ms": {
                stage: latency_summary([t[stage] for t in rows["stages"]]) for stage in rows["stages"][0]
            } if rows.get("stages") else None,
            "_per_query": rows["metrics"],
            "_candidates": rows["candidates"],
        }
        summary[system]["metrics"]["mrr"] = summary[system]["metrics"].pop("reciprocal_rank")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/esci_heldout.json")
    parser.add_argument("--max-test-queries", type=int, default=None, help="Override for smoke runs only")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    retrieval_config = config["retrieval"]
    output = Path(config["output_root"]) / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    started = time.perf_counter()
    catalog = load_catalog(config["catalog"])
    print(f"Loaded catalog ({len(catalog):,}) in {time.perf_counter() - started:.1f}s", flush=True)
    retriever, dense, encoder_device = load_retriever({**retrieval_config, "seed": config["seed"],
                                                        "dense_index": config["dense_index"],
                                                        "bm25_index": config["bm25_index"]}, catalog, manifest)
    smoke = smoke_test(retriever, catalog, dense)

    systems = retrieval_config["systems"]
    ks = retrieval_config["candidate_ks"]
    retrieve_k = max(ks)
    selection = retrieval_config["selection"]
    validation = stable_sample(load_queries(config["labels"], "validation"), selection["max_queries"], config["seed"])
    test_limit = args.max_test_queries if args.max_test_queries is not None else retrieval_config["test_max_queries"]
    test = stable_sample(load_queries(config["labels"], "test"), test_limit, config["seed"])
    test_ids = {q["query_id"] for q in test}
    if test_ids & {q["query_id"] for q in validation}:
        raise SystemExit("Validation and test query IDs overlap")

    with (output / "retrieval_per_query.jsonl").open("w", encoding="utf-8") as per_query, (
        output / "retrieval_latency.csv"
    ).open("w", newline="", encoding="utf-8") as latency_file:
        latency_writer = csv.writer(latency_file)
        latency_writer.writerow(["split", "system", "query_id", "latency_ms"])
        validation_summary = evaluate_split(
            retriever, catalog, validation, systems, ks, retrieve_k, per_query, latency_writer, "validation"
        )
        metric = selection["metric"]
        selected = max(systems, key=lambda s: (validation_summary[s]["metrics"][metric], -systems.index(s)))
        print(f"Selected on validation by {metric}: {selected}", flush=True)
        test_summary = evaluate_split(
            retriever, catalog, test, systems, ks, retrieve_k, per_query, latency_writer, "test"
        )

    comparisons = {}
    for baseline, challenger in (("bm25", "hybrid"), ("bm25", "dense"), ("dense", "hybrid")):
        comparisons[f"{challenger}_minus_{baseline}"] = {
            key: paired_bootstrap(
                [m[key] for m in test_summary[baseline]["_per_query"]],
                [m[key] for m in test_summary[challenger]["_per_query"]],
                retrieval_config["bootstrap_samples"],
                config["seed"],
            )
            for key in ("recall_at_10", "recall_at_40", "hit_rate_at_10", "hit_rate_at_40", "reciprocal_rank")
        }

    candidates_path = output / "fixed_candidates_test.jsonl"
    with candidates_path.open("w", encoding="utf-8") as handle:
        for row in test_summary[selected]["_candidates"]:
            handle.write(json.dumps({"system": selected, **row}) + "\n")

    def public(summary: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {s: {k: v for k, v in row.items() if not k.startswith("_")} for s, row in summary.items()}

    results = {
        "run_id": config["run_id"],
        "provenance": run_provenance(),
        "config": config,
        "config_sha256": sha256_file(args.config),
        "data": {
            "manifest": config["manifest"],
            "manifest_sha256": sha256_file(config["manifest"]),
            "catalog": config["catalog"],
            "catalog_products": len(catalog),
            "catalog_content_fingerprint": catalog.fingerprint(),
            "labels_sha256": sha256_file(config["labels"]),
            "validation_queries": len(validation),
            "test_queries": len(test),
            "test_query_ids_sha256": hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest(),
        },
        "indexes": {
            "dense": {**dense.metadata.__dict__, "path": config["dense_index"], "query_encoder_device": encoder_device,
                      "faiss_file_sha256": sha256_file(config["dense_index"])},
            "bm25": {"path": config["bm25_index"], "k1": retriever.bm25.k1, "b": retriever.bm25.b,
                     "vocabulary_terms": len(retriever.bm25.vocabulary), "documents": retriever.bm25.document_count},
            "rrf": {"k": retriever.rrf_k, "fusion_depth": retriever.fusion_depth,
                    "formula": "score(d) = sum_i 1 / (k + rank_i(d)) over full BM25 and dense rankings"},
            "smoke_test": smoke,
        },
        "metric_definitions": {
            "relevant": "ESCI E or S judged for the query",
            "recall_at_k": "known judged E/S positives in original top K / all known judged E/S positives (coverage of known positives, not catalog-wide recall)",
            "hit_rate_at_k": "1 if any known E/S positive is in the original top K",
            "precision_at_k": "known E/S in top K / min(K, returned); unjudged counted non-relevant (lower bound)",
            "judged_at_k": "fraction of top K with any ESCI judgment for the query",
            "mrr": "reciprocal original rank of the first known E/S positive within the top 40 (0 if none)",
            "latency_ms": "wall time per query, single query at a time, including query embedding for dense/hybrid",
        },
        "selection": {
            "rule": selection,
            "split": "validation",
            "validation_metrics": {s: validation_summary[s]["metrics"] for s in systems},
            "selected_system": selected,
        },
        "validation": public(validation_summary),
        "test": public(test_summary),
        "test_paired_comparisons": comparisons,
        "fixed_candidates": {
            "path": str(candidates_path),
            "system": selected,
            "k": retrieve_k,
            "sha256": sha256_file(candidates_path),
        },
    }
    (output / "retrieval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({s: public(test_summary)[s]["metrics"] for s in systems}, indent=2))


if __name__ == "__main__":
    main()
