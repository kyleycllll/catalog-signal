# Retrieval + reranking evaluation: completion report

Date: 2026-09-21. Run ID: `esci-us-heldout-v1`. Code base: `7424dbb` plus uncommitted changes
(provenance records `git_tracked_changes_uncommitted: true`). All numbers below come from
`reports/experiments/esci-us-heldout-v1/*.json`. The integrity checks are in
`experiment_integrity.json`, and all 14 pass.

## 1. Dataset

- **Source:** Amazon Shopping Queries (ESCI), `github.com/amazon-science/esci-data` at commit
  `7916cdf6ab75`. Both parquet files match their official Git-LFS SHA-256 oids. The products file
  inherited from the previous agent was corrupt and was re-downloaded.
- **Manifest:** `data/manifests/esci_split_manifest.json`. It records file hashes, row, label, query
  and product counts, the query-split integrity checks, cross-split product overlap, and the catalog
  fingerprint.
- **Scope:** US locale. Labels are `small_version=1`. Official test queries are **final-test only**.
  Train and validation are a deterministic query-ID hash partition of non-test queries, and all three
  query sets are disjoint and asserted.
- **Evaluation catalog:** all **1,215,854** US products in the official ESCI product corpus. That
  corpus is the union of products judged for any ESCI query (small or large version); it is not the
  whole Amazon catalog. The catalog was built without reading labels. Products are shared across
  query splits, and the overlap is reported (for example, 32,499 labelled products appear in both
  train and test labels).
- **Test queries:** 8,956 for retrieval, all of which have at least one known E/S. Reranking used a
  fixed deterministic sample of 200 of them. Validation used 2,132 queries, only for selecting the
  retrieval system.
- **Partial-judgment caveat:** ESCI judges about 20 products per query. Only about 14% of retrieved
  top-40 items carry a judgment for their query. Recall means recall of *known* judged E/S positives,
  and unjudged items count as non-relevant, so metrics are lower bounds.

## 2. Retrieval results (8,956 official test queries)

| System | Recall@10 | Recall@40 | HitRate@10 | HitRate@40 | MRR | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.1541 | 0.2880 | 0.6567 | 0.7921 | 0.4543 | 14.9 ms | 51.4 ms |
| Dense MiniLM + FAISS | 0.1292 | 0.2549 | 0.6112 | 0.7644 | 0.4059 | 38.3 ms | 97.4 ms |
| Hybrid RRF (k = 60) | **0.1649** | **0.3224** | **0.6967** | **0.8342** | **0.4844** | 430.3 ms* | 833.5 ms* |

Hybrid minus BM25 on Recall@40 is **+0.0345, 95% paired bootstrap CI [0.0317, 0.0369]**. Every
hybrid-vs-BM25 metric interval excludes 0. Dense alone is significantly worse than BM25.

\*Measured hybrid wall time includes about 360 ms (p50) of exact full-catalog rank lookup, used only
to reproduce `retrieve()` exactly. Its measured component stages sum to about 55 ms (p50), which is an
estimate for a top-1000 fusion, not a measured end-to-end figure.

Details: `reports/experiments/esci-us-heldout-v1/retrieval_comparison.md`.

## 3. Reranking results (200 test queries, fixed hybrid top-40 candidates)

The metrics below are full-rank, with unjudged items given gain 0 and the ideal computed over all
known judgments.

| System | NDCG@5 | NDCG@10 | MRR | p50 rerank latency |
| --- | ---: | ---: | ---: | ---: |
| Hybrid RRF (pre-rerank) | 0.2545 | 0.2410 | 0.4278 | — |
| Same candidates after Qwen QLoRA | 0.1982 | 0.1926 | 0.3928 | 8,838 ms / query (40 candidates) |
| Δ [95% CI] | −0.0562 [−0.0848, −0.0311] | −0.0484 [−0.0695, −0.0292] | −0.0351 [−0.0747, +0.0032] | |

- **NDCG@10:** 48 queries improved, 73 worsened and 79 were unchanged.
- **Judged-pool diagnostics** (unjudged items removed after ranking; not full NDCG) also decline:
  NDCG@10 falls from 0.7297 to 0.7159, CI [−0.0256, −0.0025].
- **Oracle reorder** of the same candidates (reference only, uses test labels): NDCG@10 is 0.5021.

Details: `reports/experiments/esci-us-heldout-v1/reranking_comparison.md`.

## 4. Classifier evidence

- **Task:** Qwen2.5-0.5B-Instruct with a 4-bit NF4 QLoRA adapter (r = 16, q/k/v/o). It performs
  pointwise generative E/S/C/I classification, with E/S/C/I mapped to gains 3/2/1/0 at serving time.
- **Historical saved evidence:** `reports/evidence/esci-predictions.jsonl`, 500 official-test pairs,
  class-balanced. Macro-F1 is **0.1338 for the base and 0.2417 for QLoRA**; accuracy is 0.272 and
  0.284. I recomputed this from the file, and it matches `esci-comparison.json`. All 500 example IDs
  are official-test rows, and their gold labels match the prepared labels.
- **Local reproduction:** `classifier_reproduction.json`. With emulated NF4 weights and fp16 compute,
  the adapter reaches macro-F1 **0.2419** with 97.4% agreement with the Colab predictions. The base
  reaches 0.1338 with 99.8% agreement.
- **Artifact:** adapter `models/esci-qwen-lora/adapter_model.safetensors`, SHA-256 `72fca854…`.

## 5. Error analysis

- **BM25 failure:** `post-partum depression workbook` has Recall@40 of 0.00 for BM25 and 1.00 for
  dense. The hyphen tokenization `post`/`partum` does not match "Postpartum". `airmax red` matches
  Ubiquiti "airMAX" radios instead of Nike "Air Max".
- **Dense failure:** `king dedede chrisymas` has Recall@40 of 0.00 for dense and 0.64 for BM25. The
  rare entity plus misspelling sends MiniLM to unrelated media. `name tag clip` returns video titles
  beginning "Clip: …".
- **Hybrid:** it is best in every query slice (length, digits, negation). It is still worse than BM25
  on 1,634 of 8,956 queries by Recall@40, typically when dense injects topically similar but
  wrong-type items (`makita impact drill` → drill bits).
- **Qwen reranking failure pattern:** little discrimination and frequent E→C/I errors on true exact
  matches.
  - Qwen labels a median of 69% of candidates E or S.
  - 34% of judged exact matches are labelled C or I.
  - Because the label dominates the score, those errors demote true matches. 116 of the 260 known
    E/S products in the pre-rerank top 5 were pushed out.
  - Example: `habanero hot sauce cholula organic` goes from NDCG@10 0.877 to 0.221, because the
    Cholula exact matches were labelled C.

## 6. Interview claim status

| Claim | Status | Evidence / required wording |
| --- | --- | --- |
| "Hybrid retrieval improved retrieval quality" | **YES** (with scope) | Over BM25, on 8,956 held-out ESCI test queries against the 1.2M-product US corpus: Recall@40 of known E/S positives went from 0.288 to 0.322 (+0.0345, CI [0.032, 0.037]), and MRR from 0.454 to 0.484. Dense alone was *worse* than BM25. |
| "Reranking improved NDCG" | **NO** | The QLoRA reranker **lowered** NDCG@10 from 0.241 to 0.193 (−0.048, CI [−0.070, −0.029]) and NDCG@5 from 0.255 to 0.198 on fixed hybrid candidates (200 test queries). |
| "QLoRA improved ESCI relevance classification" | **YES** (narrow) | Macro-F1 went from 0.134 to 0.242 on 500 class-balanced official-test pairs (saved artifact, reproduced at 0.2419 locally). Accuracy only moved from 0.272 to 0.284. This did **not** translate into better ranking. |
| "I can claim Recall@K" | **YES**, scoped | "Recall@K of known judged E/S positives on the ESCI US product corpus (1.2M products), official test queries." It is not catalog-wide recall; judgments are partial. |
| "NDCG is full-catalog" | **NO** | NDCG is computed over the fixed top-40 candidate list with unjudged items at gain 0 (a lower bound), plus separately labelled judged-pool diagnostics. It is not a fully judged, full-catalog NDCG. |

## 7. Remaining limitations

- Judgments are partial. All absolute metrics are lower bounds, and ESCI's pools may favour lexical
  systems.
- The reranking sample is 200 queries (reduced from 300 for throughput, decided before any output).
  Intervals are wide, but the NDCG losses are significant.
- Reranking inference is a faithful local NF4 emulation, not the bitsandbytes CUDA kernels. Latency
  is measured on an M3 under memory pressure and is not the Colab/HTTP serving latency.
- The exact training rows of the historical adapter were not saved. By the notebook's construction
  they come from non-test queries only, but that cannot be re-verified byte-for-byte.
- The application's `retrieve()` cannot serve a 1.2M catalog interactively (it tokenizes per query).
  The benchmark uses an equivalence-tested indexed implementation.
- Versioning: these reports are tracked through explicit `.gitignore` exceptions (the repo still
  ignores `*.md` by default). Raw data, indexes and the adapter remain outside Git by design. The 61 MB
  `retrieval_per_query.jsonl` is also untracked; its SHA-256, size, schema and regeneration command are
  in `large_artifacts.json`, and its per-query metrics are tracked in `retrieval_per_query_metrics.csv`.

## Reproduce

```bash
python scripts/prepare_esci.py --dataset-dir /path/to/esci-data
python scripts/build_indexes.py --catalog data/processed/esci_us_products.parquet --text-output data/indexes/esci_us_minilm.faiss --manifest data/manifests/esci_split_manifest.json --dtype float16
python scripts/run_esci_retrieval_benchmark.py --config configs/esci_heldout.json
python scripts/verify_local_adapter.py --output reports/experiments/esci-us-heldout-v1/classifier_reproduction.json
python scripts/run_esci_reranking_eval.py --config configs/esci_heldout.json
python scripts/validate_esci_experiment.py --config configs/esci_heldout.json
```
