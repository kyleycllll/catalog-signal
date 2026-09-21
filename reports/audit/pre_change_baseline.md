# Pre-change baseline audit

Audit date: 2026-09-21  
Git HEAD: `7424dbb` (`changed`)

This report records the repository before any behavior-changing remediation. It is
intentionally a factual baseline, not a result claim.

## Repository and reproducibility state

- `git status --short --ignored` reports `reports/` as untracked.
- The broad `*.md` rule in `.gitignore` excludes the root `README.md`, project
  documentation, and this audit report. `models/`, `data/indexes/`,
  `data/processed/`, and `data/raw/` are also ignored.
- Consequently, Git HEAD tracks the application code and a small sample catalog,
  but not the current README, local QLoRA adapter, processed catalog, dense-index
  artifacts, raw data, or evidence reports.
- The workspace contains `models/esci-qwen-lora/`, including an adapter config and
  safetensors file; `reports/evidence/` contains classifier predictions, a
  comparison JSON, an artifact audit, and one live integration smoke trace. These
  are local evidence, not reproducible clone-level artifacts.
- The workspace has a 141-product processed Farfetch catalog and a test-only
  `data/indexes/text_embeddings.npz`, but neither is the default ESCI catalog or
  FAISS index used by the application.

## Current runtime paths and runnable modes

The default API catalog lookup prefers `data/processed/esci_products.json` and
falls back to `data/sample_esci_catalog.json`. The first file is absent.

Observed local health state without contacting the remote model service:

```json
{
  "catalog_products": 6,
  "catalog_path": "data/sample_esci_catalog.json",
  "sample_catalog": true,
  "dense_index_available": false,
  "dense_index_path": "data/indexes/text_embeddings.faiss",
  "dense_index_version": null,
  "dense_index_error": "Dense index or metadata is missing at data/indexes/text_embeddings.faiss"
}
```

- BM25-only candidate retrieval runs against the six-product sample catalog. For
  `black university backpack`, the top three IDs were `demo-backpack-1`,
  `demo-backpack-2`, and `demo-bottle-1`.
- Dense and hybrid API requests fail explicitly because the configured FAISS index
  and metadata are absent; they do not fall back to BM25.
- The default API/UI request is hybrid retrieval with reranking, so it cannot
  complete locally before the dense-index check.
- BM25 plus reranking additionally requires a reachable configured remote adapter.
  Configuration was not treated as proof that the external service is live.
- `scripts/run_search_experiments.py` cannot run its example configuration because
  the referenced ESCI catalog, labels, and FAISS index are absent.

## Code and evaluation baseline

- BM25 over title, description, bullet points, category, colour, brand, material,
  and attributes is implemented in `src/product_discovery/retrieval.py`.
- Dense retrieval is designed as normalized `all-MiniLM-L6-v2` vectors in FAISS
  `IndexFlatIP`; a production index is absent.
- Hybrid retrieval uses deterministic reciprocal-rank fusion with `k=60`.
- The remote QLoRA reranker maps E/S/C/I labels to gains 3/2/1/0 and combines 80%
  label relevance with 20% normalized retrieval score.
- The active API route does not invoke the separately implemented intent extraction,
  constraint-update, result-reference, or generated-answer helpers.
- `retrieval_metrics()` labels a boolean any-relevant-in-top-K calculation as
  `recall_at_k`; this is HitRate@K/Success@K, not Recall@K.
- The experiment runner removes unjudged candidates before its query-local ranking
  metrics, so those values are not end-to-end NDCG/MRR over original ranks.

## Existing measured evidence

The saved 500-row prediction artifact supports a pair-classification comparison:

| System | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Frozen base Qwen | 0.2720 | 0.1338 |
| QLoRA adapter | 0.2840 | 0.2417 |

This does not establish retrieval quality or reranking-quality improvement. There
is no completed BM25-vs-dense-vs-hybrid report and no before/after reranking NDCG
artifact. The sole live integration trace is a six-product BM25 smoke test whose
recorded total latency is 5,955.88 ms; it is not a latency benchmark.

## Test baseline

The repository-required Python version is 3.11+. The checked-in `.venv` and
`.venv-esci` environments were created on Windows and are not usable on this macOS
host. System Python is 3.9 and cannot import the project's `X | None` annotations.

An isolated Python 3.12 environment was created outside the repository. The
documented editable install succeeded, then the suite was run:

```text
26 passed, 1 skipped, 2 warnings
```

The skipped test is the optional FAISS persistence test because FAISS is not in the
baseline test environment. Warnings come from third-party FastAPI/Starlette test
client deprecations.

## Current claim status

Defensible today:

- BM25, MiniLM/FAISS, RRF, and QLoRA reranking code paths are implemented.
- A local QLoRA adapter and saved pair-classification predictions exist.
- The adapter's stored prediction artifact improves macro-F1 relative to its frozen
  base on those 500 saved held-out pairs.

Not defensible today:

- Dense/hybrid retrieval runs locally against the intended ESCI catalog.
- Hybrid beats BM25.
- Recall@K, end-to-end NDCG, or reranking MRR improved.
- The default demo is runnable from a fresh clone.
- The current notebook exactly reproduces the historical 10k/one-epoch evidence.

## Risks to resolve in later phases

1. Correct metric definitions and preserve original ranking positions.
2. Create a provenance-rich, query-disjoint ESCI data manifest.
3. Build a matching MiniLM/FAISS index and run held-out candidate benchmarks.
4. Evaluate reranking on fixed candidate sets before claiming ranking improvement.
5. Version lightweight configs, manifests, results, and documentation.
