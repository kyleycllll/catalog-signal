# Adaptive Product Search — Task Handover

## Current status

The functional serving milestone is implemented and fixture-tested. The
production loader no longer materializes the 1.2M-product catalog in Python
memory. Before starting the API against the full data, build the required
memory-mapped serving metadata artifact with:

```bash
PYTHONPATH=src python scripts/build_serving_metadata.py
```

This is an offline, streaming build; it has deliberately **not** been run during
this milestone after the full-catalog smoke test exceeded the local memory budget.

## Completed

- Replaced the legacy session/chat/Qwen serving surface with one `POST /search`
  endpoint accepting only `{ "query": "..." }`.
- Added deterministic Unicode, punctuation, hyphen, compact-name, brand, colour,
  numeric/model-token, and negation analysis.
- Wired the production pipeline to `IndexedRetriever.hybrid_search`, then the
  local MiniLM cross-encoder's expected ESCI gain, then deterministic exclusion
  adjustment.
- Removed remote Qwen, Colab, ngrok, session-memory, strategy-selection, and
  reranker-selection dependencies from active serving and the frontend.
- Added `/health`, `/ready`, structured request logs, and per-stage latency
  instrumentation.
- Added memory-mapped Arrow serving metadata and a streaming builder so result
  metadata is read only for retrieved rows rather than holding 1.2M products.
- Added deterministic adaptive RRF weighting: model/numeric/brand/rare-token
  queries favour BM25; long natural-language requests favour dense retrieval;
  negated requests retain a lexical majority while remaining hybrid.
- Added bounded field-aware lexical re-scoring of only the top 120 BM25 rows,
  rewarding title, exact brand, model-token, and colour matches without scanning
  the catalog or rewarding incidental description text.
- Added a collapsed frontend search-quality panel with query attributes, adaptive
  weights, and actual per-result ranking signals. It exposes no serving knobs.
- Added a streaming, reproducible 100-query fixed-candidate reranking bundle
  builder for low-memory cloud evaluation.
- Kept reference BM25/dense/hybrid retrieval baselines separate from the serving
  pipeline for offline comparison only.
- Removed the obsolete agent/session/Qwen serving implementation, its notebooks,
  synthetic-intent assets, Qwen-only evaluation scripts, and stale Qwen result
  artifacts from Git tracking. The active package now has one MiniLM
  cross-encoder reranker path.

## Files changed

- `src/product_discovery/api.py`
- `src/product_discovery/adaptive_retrieval.py`
- `src/product_discovery/search_pipeline.py`
- `src/product_discovery/query_understanding.py`
- `src/product_discovery/serving_catalog.py`
- `src/product_discovery/embeddings.py`
- `src/product_discovery/esci_catalog.py`
- `src/product_discovery/cross_encoder_reranker.py`
- `src/product_discovery/schemas.py`
- `scripts/build_minimal_reranker_eval_bundle.py`
- `scripts/build_serving_metadata.py`
- `pyproject.toml`
- `Dockerfile`
- `.env.example`
- `frontend/src/main.jsx`
- `frontend/src/style.css`
- `tests/test_api.py`
- `tests/test_adaptive_retrieval.py`
- `tests/test_query_understanding.py`
- `tests/test_serving_catalog.py`
- Removed legacy Qwen/agent/session sources, associated tests and notebooks,
  synthetic-intent data, and Qwen-only reports (see the staged deletions).

## Tests run

- `env PYTHONPATH=src /private/tmp/catalog-signal-eval-venv/bin/python -m pytest -q`
  - 54 passed after the legacy-file cleanup (two third-party deprecation
    warnings from FastAPI/Starlette's test client).
- `npm run build` in `frontend/`
  - passed.
- `git diff --check`
  - passed.
- `python -m py_compile scripts/build_minimal_reranker_eval_bundle.py scripts/build_serving_metadata.py`
  - passed.

## Experimental results

- Cloud fixed-candidate MiniLM reranking evaluation (100 historical test
  queries, 40 fixed hybrid candidates/query, seed 42):
  - NDCG@5: 0.3044 → 0.3532 (mean delta +0.0488; 35 improved, 22 harmed;
    bootstrap 95% CI [-0.0055, +0.1069]).
  - NDCG@10: 0.3349 → 0.3858 (mean delta +0.0509; 46 improved, 23 harmed;
    bootstrap 95% CI [+0.0028, +0.1035]).
  - MRR: 0.4233 → 0.4592 (mean delta +0.0359; 31 improved, 19 harmed;
    bootstrap 95% CI [-0.0260, +0.1040]).
  - Cloud CPU cross-encoder-only latency for 40 candidates/query: p50 3268.37
    ms, p95 3772.76 ms, p99 3817.48 ms (mean 3358.53 ms).
- This is a fixed-candidate cross-encoder check using raw historical query text:
  it does not measure retrieval quality, query normalization, API latency, or
  full end-to-end serving performance.
- Existing benchmark code and `IndexedRetriever` ranking behavior remain intact;
  the indexed-reference equality test continues to pass.
- No full-catalog serving latency is claimed or measured for this milestone.

## Decisions

- The serving architecture is one fixed path: query understanding → indexed RRF
  hybrid retrieval with deterministic adaptive weights and bounded field-aware
  lexical boosts → MiniLM cross-encoder expected-gain ranking → exclusion
  adjustment.
- RRF remains at the existing frozen value (`k=60`, depth `1000`) until the
  validation-only tuning phase.
- Field-aware lexical re-scoring reads title, brand, and colour for at most 120
  already-retrieved BM25 rows. It does not create another index, duplicate the
  catalog, or scan product metadata per request.
- Product metadata is an Arrow IPC file opened through memory mapping; the dense
  index ID mapping supplies the row-to-product-ID relationship and is checked by
  product count, catalog fingerprint, and row-order hash.
- The minimal evaluation uses a generated 1.48 MB bundle rather than transferring
  or loading the catalog in Colab; all cloud outputs were written as new files.
- Do not load a full Parquet catalog or 1.2M Pydantic products during application
  startup. The failed smoke-test approach is explicitly retired.
- The repository cleanup intentionally excludes raw catalog data, indexes, model
  files, local environment files, and retained offline/preparation components.

## Issues

- `data/indexes/esci_us_product_metadata.arrow` does not yet exist, so `/ready`
  will correctly report an actionable metadata-build error until the streaming
  builder is run.
- Full end-to-end startup and latency measurement remain pending. Run them only
  after the metadata artifact exists, using controlled memory observation; do not
  combine a full Python catalog materialization with index loading.

## Next steps

1. Build the serving metadata artifact with the new streaming script, preferably
   while monitoring memory, then verify `/ready` and a small `/search` smoke test.
2. If desired, run a small controlled end-to-end smoke test after metadata exists;
   never start the API by loading the parquet catalog or 1.2M Pydantic products.
3. Keep evaluation, tuning, and further ranking experiments optional rather than
   expanding the production architecture.
