# Claude takeover state

Recorded: 2026-09-21, at handoff from the previous agent. This report records only what
was inspected and verified. It does not restate the handoff notes.

## Git state at takeover

- Branch `main`, HEAD `7424dbb` (`changed`), up to date with `origin/main`.
- Modified (uncommitted): `scripts/prepare_esci.py`, `scripts/run_search_experiments.py`,
  `src/product_discovery/evaluation.py`, `tests/test_evaluation.py`,
  `tests/test_experiment_runner.py`.
- Untracked: `reports/`. Note that `.gitignore` contains `*.md`, so Markdown reports are
  ignored unless force-added.
- No uncommitted work was reset or discarded.

## Verified completed work

| Item | Status | Evidence |
| --- | --- | --- |
| Phase 0 baseline report | Present | `reports/audit/pre_change_baseline.md` |
| Phase 1 metric fixes | Verified by code review | `evaluation.py` implements Recall@K (fraction of known positives), HitRate@K, Precision@K (denominator `min(k, len)`), original-rank reciprocal rank, and explicitly named judged-pool metrics. The runner now reports `observed_candidate_coverage` separately from `retrieved_judged_pool_ranking`. |
| Phase 1 tests | Pass | Fresh Python 3.12 environment at `/private/tmp/catalog-signal-eval-venv` with the `data,retrieval,dev` extras plus torch/transformers/peft: **29 passed, 0 skipped**. The FAISS persistence test that was skipped in Phase 0 now runs because `faiss-cpu 1.15.1` is installed. |

## Partial Phase 2 work found

- `scripts/prepare_esci.py` was rewritten, but its output has never been generated. It
  has a query-level split, split-integrity assertions, and a provenance manifest.
  Its catalog is **only the products referenced by the sampled labelled pairs**
  (10k train / 1k validation / 1k test rows). That is too small, and too biased toward
  judged products, for a valid retrieval benchmark.
- There is no `data/manifests/`, `data/esci/`, `configs/esci_heldout.json`, FAISS index, or
  `reports/experiments/` output.

## Data-download state

- The previous agent cloned `https://github.com/amazon-science/esci-data.git` into
  `/private/tmp/esci-data` at commit `7916cdf6ab75a462e77f20ab40428a10923998d5`
  (2024-10-07). `git-lfs` is not installed. The parquet files were fetched manually.
- `shopping_queries_dataset_examples.parquet`: 51,286,808 bytes. The SHA-256 is
  `4a735b69…263a`, which **matches** the LFS pointer oid.
- `shopping_queries_dataset_products.parquet` was **corrupt**: 1,109,774,969 bytes against
  the expected 1,108,857,465, with SHA-256 `46febad4…` against the expected `25124442…`.
  The download had "finished" with extra bytes, probably from a resumed or appended
  transfer. No download process was still running.
- Remediation: the file was re-downloaded from the official GitHub LFS media endpoint
  pinned to the commit above:
  `https://media.githubusercontent.com/media/amazon-science/esci-data/7916cdf…/shopping_queries_dataset/shopping_queries_dataset_products.parquet`.
  The new SHA-256 is `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265`,
  which **matches** the LFS pointer oid exactly. The corrupt copy was deleted.
- Raw data stays outside the repository in `/private/tmp/esci-data`. Only manifests,
  configs and compact results are intended for Git.

## Next actions (executed after this report)

1. Extend `prepare_esci.py` to write the full US product corpus as the evaluation catalog,
   the complete query-level label files, and a provenance manifest.
2. Build the MiniLM + FAISS `IndexFlatIP` index over the full US corpus, with fingerprint
   metadata.
3. Run BM25, dense and hybrid RRF on official test queries. Select the reranking input
   system on validation queries, not on test queries.
4. Run the fixed-candidate Qwen reranking evaluation locally with the saved adapter.
5. Validate, test and document.

## Outcome (end of this handoff)

All next actions above were completed. See `reports/retrieval_reranking_evaluation_complete.md`.
There was one additional finding during execution. Running the adapter on a full-precision bf16 base
does **not** reproduce the Colab model: 63.4% agreement and macro-F1 0.146, logged in
`reports/experiments/esci-us-heldout-v1/classifier_reproduction_bf16_full.log`. Emulating the 4-bit
NF4 base does reproduce it (97.4%, macro-F1 0.2419), so the reranking evaluation used the NF4
emulation.
