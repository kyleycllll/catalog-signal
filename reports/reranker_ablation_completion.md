# Reranker ablation: completion report

Run `esci-reranker-ablation-v1`. Dates: 2026-09-21 to 2026-09-22. Config: `configs/esci_reranker_ablation.json`.

- **Sources:** every number is from `reports/experiments/esci-reranker-ablation-v1/`:
  - `fixed_candidate_comparison.json` (test)
  - `validation_selection.json` (validation)
  - `training_data_manifest.json`
  - `hard_negative_mining_summary.json`
  - `failure_analysis.json`
- **Integrity:** `ablation_integrity.json`, 13/13 checks pass.
- **Tests:** 61 pass.

**Bottom line**

- Retrieval-mined hard negatives did **not** fix the Qwen failure; they made it worse.
- A 22.7M-parameter MiniLM cross-encoder is the only reranker that beats the hybrid order on the
  fixed test candidates, and it is about 64× faster than Qwen.
- Its gain is modest (NDCG@10 +0.027, CI [0.015, 0.039]).
- The more conservative judged-pool NDCG@10 diagnostic is positive but not significant.

## 1. Baseline state

- Baseline commit **`11ada8e`**: the full retrieval and fixed-candidate reranking benchmark
  (`reports/retrieval_reranking_evaluation_complete.md`). Historical results are unchanged.
- The 61 MB `retrieval_per_query.jsonl` is not in Git. Its SHA-256, size, schema and regeneration
  command are in `esci-us-heldout-v1/large_artifacts.json`, and its per-query metrics are tracked in
  `retrieval_per_query_metrics.csv`. That CSV reproduces the reported Recall@40 values
  (0.2880 / 0.2549 / 0.3224).
- Further commits:
  - `3d49f03`: the mining, data and training pipeline.
  - `b54dd1e`: validation selections frozen before any new test run.
  - The final commit holds the test results and this report.
- **Historical adapter, same-conditions rerun.** The saved adapter was re-run on the same 200 test
  queries. It reproduces the recorded predictions exactly: 8,000/8,000 labels and 200/200 orderings.
  The recorded historical results are therefore used unchanged. The rerun is used only for a
  same-machine latency figure.
- **Diagnosed probable cause of the historical adapter's weakness** (not verifiable
  byte-for-byte):
  - The historical protocol records `max_sequence_length: 256`, and the label is the last thing in
    the training text.
  - On the prepared train sample, **57% of training texts exceed 256 tokens**. With TRL's default
    keep-start truncation those rows lose their label.
  - Training was also class-balanced (C is 25% of training but 3.6% of reranking candidates), with
    loss on the full prompt.
  - The historical training rows and TRL version were not saved, so this is a strong hypothesis,
    not a proven cause.

## 2. Hard-negative dataset

- **Source:** prepared-TRAIN queries only; all 18,756 were mined. Official test queries, test
  candidates and test labels were not used. Validation was used only for selection.
- **Retriever:** hybrid RRF (k = 60) over the same BM25 top-1000 and MiniLM/FAISS top-1000 lists
  as the fixed-candidate system.
  - Fusion was truncated: a product missing from one list gets rank 1001.
  - Exact full-catalog rank fusion ran at about 1.4 queries/s under memory pressure, so it was
    replaced before any training data existed.
  - On 50 train queries, the truncated and exact versions overlap **97.8% in the top 40** and
    97.6% in the top 100 (`truncated_vs_exact_agreement.json`).
- **Mining K:** 100.
- **Judged labels found in the top 100:** E 79,069; S 47,943; C 7,560; I 14,017. There were
  1,727,011 unjudged products, never used as negatives.
- **Rule:** a hard negative is a product with an **observed** ESCI I or C label for that train
  query, ranked in the hybrid top 100.
  - Labels are never changed.
  - Hardest (lowest rank) first, at most 2 per query per label.
- **Selected hard negatives:**
  - I: 2,500, median hybrid rank 3 (max 8), from 1,907 queries.
  - C: 2,500, median rank 5 (max 17), from 1,730 queries.
  - No random fill was needed.
- **Variants** (seed 42, 10,000 rows each, 2,500 per label):
  - **A, `A_balanced_random`:** the historical data approach, with random class-balanced train
    pairs from 7,193 queries.
  - **B, `B_hard_negative`:** the **identical** 5,000 E and S rows (asserted), with the I and C
    rows replaced by the hard negatives. 4,547 rows differ from A, and 453 of A's random I/C rows
    happened to be hard ones.
- **Validation selection set:** all judged products in the hybrid top 40 of 300 validation
  queries, plus the first 2 unjudged per query.
  - Judged products: E 863, S 535, C 142, I 139.
  - Unjudged products: 600, used only to measure how often a model predicts E or S on them.
- Row identities, with hybrid rank, are tracked in `training_rows_*.csv` and
  `validation_pairs.csv`. The 12 MB mining file is untracked, and its hash is in the manifest.

## 3. Training config

- **Qwen** (`scripts/train_esci_reranker.py`, one config for A and B):
  - Model: Qwen2.5-0.5B-Instruct with an emulated 4-bit NF4 double-quantized base (the
    representation that reproduced the historical adapter), frozen in fp16.
  - LoRA: r = 16, α = 32, dropout 0.05, on q/k/v/o. That is 2,162,688 trainable parameters, the
    same as the historical adapter.
  - Optimisation: AdamW, lr 2e-4, cosine schedule, 5% warmup, effective batch 16 (4 × 4
    accumulation), 1 epoch (625 steps), fp16 with loss scaling, gradient clipping 1.0, seed 42.
  - Prompt: the serving prompt, byte-identical.
  - Two changes from the historical recipe:
    - Loss is on the `{"label":"X"}` target plus EOS only.
    - The label is never truncated: max length 1024, and the product text is shortened if needed
      (6 rows in A, 11 in B).
  - Checkpoints at 50% and 100%; per arm, the one with the higher validation macro-F1 was kept.
  - Per-run artifacts are in `training_runs/qwen_*/`, adapters in `models/ablation/` (untracked).
- **Cross-encoder** (`scripts/train_esci_cross_encoder.py`):
  - Model: `cross-encoder/ms-marco-MiniLM-L6-v2` (22,714,756 parameters) with a new 4-way ESCI
    head.
  - Training: lr 2e-5, batch 32, 3 epochs, 10% warmup, weight decay 0.01, max 256 tokens.
  - Product text uses the same fields and 900-character caps as the Qwen prompt.
  - Trained on A and on B separately. The (variant, epoch) with the best validation macro-F1 was
    selected.
- **Preregistered test ranking score for the cross-encoder:** expected gain Σ p(c)·gain(c) in place
  of the argmax gain in the production formula `0.8·g/3 + 0.2·norm RRF`. The argmax-label variant
  is also reported.
- **Compute:** local Apple M3 (16 GB) with MPS.
  - Qwen A took 3.2 h. That run was slowed by swap and an early overlapping job; B took 1.3 h.
    These wall times are not comparable.
  - The cross-encoder took about 10 minutes per variant.

## 4. Classification results

**Validation** (300 validation queries, 1,679 judged pairs). This set was used for selection:

| Model | Macro-F1 | Acc. | E recall | E→C/I rate | Unjudged predicted E/S |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical Qwen adapter | 0.222 | 0.351 | 0.585 | 0.319 | 0.668 |
| Qwen A @ 312 steps | 0.236 | 0.259 | 0.270 | 0.684 | 0.312 |
| **Qwen A @ 625 (selected)** | **0.395** | 0.440 | 0.467 | 0.254 | 0.660 |
| Qwen B @ 312 steps | 0.201 | 0.229 | 0.256 | 0.707 | 0.257 |
| **Qwen B @ 625 (selected)** | **0.286** | 0.298 | 0.174 | 0.461 | 0.513 |
| **Cross-encoder A, epoch 3 (selected)** | **0.424** | 0.528 | 0.623 | 0.211 | 0.692 |
| Cross-encoder B, best epoch (2) | 0.269 | 0.268 | 0.200 | 0.706 | 0.445 |

Hard negatives reduced validation macro-F1 and raised E→C/I for **both** architectures.

**Test** (the 1,070 judged candidates in the 200 fixed test queries; identical items for every
system). Cells show precision / recall:

| System | Macro-F1 | Acc. | E P / R | S P / R | C P / R | I P / R | E→C | E→I | Unjudged predicted E |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| Historical Qwen | 0.247 | 0.358 | 0.50 / 0.56 | 0.42 / 0.12 | 0.08 / 0.49 | 0.14 / 0.16 | 127 | 56 | 58.7% |
| Qwen A (fixed recipe) | 0.386 | 0.456 | 0.58 / 0.53 | 0.47 / 0.35 | 0.17 / 0.69 | 0.27 / 0.37 | 76 | 47 | 30.0% |
| Qwen B (hard negatives) | 0.263 | 0.293 | 0.44 / **0.17** | 0.40 / 0.41 | 0.12 / 0.54 | 0.14 / 0.43 | 98 | **140** | 21.1% |
| Cross-encoder | **0.410** | **0.530** | 0.65 / **0.65** | 0.59 / 0.42 | 0.15 / 0.74 | 0.31 / 0.20 | 84 | **13** | 30.8% |

- **Exact matches labelled C or I** (of 544 true E): historical 183 (33.6%), A 123 (22.6%),
  **B 238 (43.8%)**, cross-encoder 97 (17.8%).
- **Confusion matrices** are in `fixed_candidate_comparison.json`, with rows gold E/S/C/I and
  columns predicted.

## 5. Fixed-candidate ranking results

- **Candidates:** the same 200 official test queries and the same hybrid top 40 as the historical
  evaluation (`fixed_candidates_test.jsonl`, SHA-256 `44201cf3…`, checked on every run).
- **Query IDs:** asserted identical to the historical evaluation.
- **No re-retrieval.** Every reranker only permutes the fixed IDs.
- **Metrics:**
  - Full-rank metrics give unjudged items gain 0.
  - Judged-pool metrics are diagnostics.
- Paired bootstrap with 2,000 resamples. Δ is measured against no reranker.

| System | NDCG@5 | NDCG@10 | MRR | Δ NDCG@10 [95% CI] | Queries improved / worsened / unchanged (NDCG@10) | Known E/S pushed out of top 5 (of 260) |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| Hybrid, no reranker | 0.2545 | 0.2410 | 0.4278 | — | — | — |
| Historical Qwen | 0.1982 | 0.1926 | 0.3928 | **−0.0484 [−0.0695, −0.0292]** | 48 / 73 / 79 | 116 |
| Qwen A (fixed recipe) | 0.2372 | 0.2280 | 0.4211 | −0.0130 [−0.0305, +0.0048] | 64 / 59 / 77 | 83 |
| **Qwen B (hard negatives)** | **0.1686** | **0.1695** | **0.3395** | **−0.0715 [−0.0968, −0.0470]** | 39 / 82 / 79 | **143** |
| **Cross-encoder (expected gain, preregistered)** | **0.2810** | **0.2682** | **0.4641** | **+0.0272 [+0.0152, +0.0390]** | 88 / 41 / 71 | **49** |
| Cross-encoder (argmax label) | 0.2675 | 0.2525 | 0.4519 | +0.0115 [−0.0033, +0.0251] | 72 / 38 / 90 | 59 |

**Other intervals against no reranker:**

- NDCG@5:
  - Cross-encoder: +0.0266 [0.0103, 0.0432].
  - Qwen A: −0.0173 [−0.0378, +0.0032].
  - Qwen B: −0.0858 [−0.1188, −0.0535].
- MRR:
  - Cross-encoder: +0.0362 [0.0057, 0.0674].
  - Qwen A: −0.0068 [−0.0388, +0.0248].
  - Qwen B: −0.0883 [−0.1371, −0.0384].
- Judged-pool diagnostics:
  - NDCG@10: cross-encoder +0.0080 [−0.0008, +0.0175]; A −0.0056 [−0.0154, +0.0025];
    B −0.0143 [−0.0255, −0.0033].
  - NDCG@5: cross-encoder +0.0140 [+0.0018, +0.0268].

**Pairwise NDCG@10 comparisons:**

- B − A: **−0.0585 [−0.0830, −0.0338]**. Hard negatives significantly hurt ranking.
- A − historical: +0.0354 [0.0152, 0.0566].
- B − historical: −0.0231 [−0.0453, +0.0012]; on MRR, −0.0533 [−0.1031, −0.0023].
- Cross-encoder − A: +0.0402 [0.0209, 0.0590].
- Cross-encoder − historical: +0.0756 [0.0551, 0.0976].

Reference: an oracle ordering of the same candidates reaches NDCG@10 0.5021. It uses test labels,
so it is not a system.

## 6. Simple baseline results

The MiniLM cross-encoder, selected on validation (variant A, epoch 3), is:

- **the best classifier:** test macro-F1 0.410, E recall 0.65, and only 13 E→I errors;
- **the only system with a significant full-rank NDCG gain** over the hybrid order;
- **the reranker that demotes the fewest known relevant products:** 49 known E/S products pushed
  out of the top 5, against 116 for the historical adapter.

Its preregistered expected-gain score matters. Ranking by the argmax label from the same model
gives a gain whose CI includes 0. The probability-weighted score is what makes it work.

## 7. Quality / latency comparison

| System | Macro-F1 (test judged) | NDCG@5 | NDCG@10 | MRR | p50 (ms / query) | p95 (ms / query) |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid, no reranker | — | 0.2545 | 0.2410 | 0.4278 | 0 (retrieval only) | 0 |
| Historical Qwen QLoRA | 0.247 | 0.1982 | 0.1926 | 0.3928 | 8,838 recorded / 8,609 rerun | 15,423 recorded / 10,722 rerun |
| Qwen A QLoRA (fixed recipe) | 0.386 | 0.2372 | 0.2280 | 0.4211 | 8,613 | 10,730 |
| Qwen B QLoRA + hard negatives | 0.263 | 0.1686 | 0.1695 | 0.3395 | 8,595 | 10,706 |
| MiniLM cross-encoder (expected gain) | 0.410 | 0.2810 | 0.2682 | 0.4641 | 134 | 136 |

**How these latencies were measured:**

- Latency is the reranking wall time for 40 candidates on one query, on an Apple M3 with MPS.
- It is measured sequentially with no other job running.
- It excludes retrieval: hybrid takes 15–430 ms, and the 430 ms includes the exact-lookup cost
  explained in the baseline report.

**Model and batching:**

- **Qwen:**
  - 494M base parameters (NF4 at serving; fp16 emulation here) plus an 8.7 MB fp32 LoRA adapter.
  - Batched greedy generation, 8 prompts per batch, up to 16 new tokens, 5 generate calls per
    query.
  - Latency is dominated by prefill of about 300-token prompts: about 215 ms per candidate.
- **Cross-encoder:**
  - 22.7M parameters, 91 MB fp32.
  - One batched forward pass over the 40 pairs, max 256 tokens: about 3.3 ms per candidate.
  - About 64× faster than Qwen.
  - Simpler to implement: there is no generation or parser, and no fallback-to-I path. The unparsed
    rate for Qwen was 0 in these runs.
- These are local measurements, not a production serving SLO.

## 8. Failure analysis

Per-query NDCG@10 for every system is in `per_query_ndcg10_all_systems.jsonl`. Slices, extremes
and representative queries are in `failure_analysis.json`.

- **Did hard negatives fix the Qwen failure? No, they made it worse.**
  - B's test E recall fell to 0.17, from 0.53 for A and 0.56 historically.
  - E→I errors rose to 140, from 47 (A) and 56 (historical). 238 of 544 exact matches were
    labelled C or I.
  - 143 of 260 known relevant top-5 products were pushed out, against 116 historically.
  - **Mechanism** (measured on the training data):
    - In B, the I and C rows have *higher* query–title token overlap (0.64 and 0.72) than the E
      rows (0.51).
    - In A, I rows have 0.28 and C rows 0.50.
    - Hard negatives without matched hard positives teach "strong lexical match → irrelevant or
      accessory", which inverts a genuine signal.
    - Examples of mined hard negatives: `the long goodbye blu ray` → "A Long Goodbye [Blu-ray]"
      (I, rank 2); `zxi 900` → a Kawasaki ZXI 1100 key switch (I, rank 1); `wok stove` → a wok
      ring for a gas stove (C, rank 1).
    - Several are near-duplicates of exact-match text, and some ESCI I/C judgements at rank 1–3
      are arguably noisy.
  - The same degradation appears in the cross-encoder trained on B: validation F1 0.269 against
    0.424, with E→C/I at 0.71. This is a data effect, not a Qwen quirk.
- **Exact matches demoted:**
  - Queries whose hybrid top-1 is a known E (54 queries), mean NDCG@10: no reranker 0.590,
    historical 0.411, A 0.520, B 0.341, cross-encoder 0.600.
  - Known E products pushed out of the top 5 (of 187): historical 86, A 50, B 98,
    cross-encoder 26.
- **Representative queries** (NDCG@10; no reranker / historical / A / B / cross-encoder):
  - `habanero hot sauce cholula organic`: 0.877 / 0.221 / 0.407 / 0.449 / **0.825**. Every Qwen
    variant still demotes the Cholula exact matches; the cross-encoder mostly keeps them.
  - `xyliwhite mouthwash`: 0.779 / 0.258 / 0.801 / 0.600 / **0.922**.
  - `vsco collage poster`: 0.599 / 0.000 / 0.000 / 0.599 / 0.599. A still loses the single known
    exact match; B and the cross-encoder keep it.
  - `rue`: 0.221 / 0.580 / **1.000** / 0.494 / 0.376. A Qwen win: retrieval put a gift card first.
- **Where hard negatives helped** (B vs historical):
  - `king dedede chrisymas` (0.108 → 0.732), `zeirbah` (0.287 → 0.645), `vsco collage poster`
    (0.000 → 0.599), `nfl white autograph football`.
  - These are mostly rare-entity queries where historical over-promoted unrelated items.
- **Where hard negatives hurt** (B vs historical):
  - `apres gel x kit` (0.761 → 0.188), `tradesmart shooting earmuffs` (0.595 → 0.092),
    `bts earrings` (0.538 → 0.064), `sigma 105mm f1.4 art nikon` (0.711 → 0.297),
    `emerson glasses` (0.410 → 0.000).
  - These are exact brand/model queries whose exact matches B labelled I.
- **Where the cross-encoder wins and loses:**
  - Largest gains over no reranker: `king dedede chrisymas` (0.146 → 0.486), `redi whip cream`
    (0.169 → 0.506), `30 oz coffee mug` (0.041 → 0.302), `computer sound systems`.
  - Largest losses: `10oz tervis cups without lid` (0.676 → 0.449; negation),
    `rachael ray marine blue pot` (0.920 → 0.723; colour attribute),
    `wireless charging power bank magetic magetic` (typo), `nfl white autograph football`.
- **Slices** (mean NDCG@10; no reranker / historical / A / B / cross-encoder):
  - Queries with digits (36): 0.244 / 0.182 / 0.222 / 0.165 / 0.279.
  - Queries of 2 tokens or fewer (39): 0.241 / 0.191 / 0.264 / 0.141 / 0.262.
  - A lexical-mismatch proxy for typos or rare entities, where a query token appears in no candidate
    title (36 queries): 0.130 / 0.126 / 0.135 / 0.125 / 0.152. Retrieval dominates here and every
    reranker is near the floor.
- **Irrelevant items over-promoted / unjudged candidates predicted E:**
  - Historical Qwen labels 58.7% of unjudged candidates E. For A, B and the cross-encoder the
    shares are 30.0%, 21.1% and 30.8%.
  - Predicted E or S on unjudged candidates stays high for every model (57–66%). Many unjudged
    top-40 hybrid candidates are plausibly relevant, so this cannot be scored.
  - The cross-encoder's full-rank gain is only partly confirmed by the judged-pool diagnostic.
    NDCG@10 is not significant, while NDCG@5 is +0.014 [0.002, 0.027]. Part of the full-rank gain
    comes from placing known-relevant items above unjudged ones, which is correct only if those
    unjudged items are less relevant.
- **Complement vs Exact:** every model over-predicts C. C precision is ≤ 0.17 for every model, and
  E→C remains the largest cross-encoder error (84). C is 3.6% of judged candidates but 25% of
  class-balanced training data. That prior mismatch was not tested in this chunk.

## 9. Currently defensible claims

1. "On 200 held-out ESCI test queries with fixed hybrid top-40 candidates, a fine-tuned 22.7M
   MiniLM cross-encoder (expected-gain scoring, selected on validation) improved NDCG@10 from 0.241
   to 0.268 (+0.027, 95% paired-bootstrap CI [0.015, 0.039]), NDCG@5 from 0.255 to 0.281, and MRR
   from 0.428 to 0.464, adding about 134 ms per 40 candidates on an M3."
   - Caveats that must accompany it: unjudged candidates count as gain 0; the judged-pool NDCG@10
     gain (+0.008) is not significant; and it is one run with one seed.
2. "Retrieval-mined hard negatives (judged I/C in the hybrid top 100 of train queries) hurt: the
   hard-negative Qwen QLoRA lowered NDCG@10 versus no reranking by 0.072 [0.047, 0.097] and versus
   the matched class-balanced run by 0.059 [0.034, 0.083]. The same data also degraded a
   cross-encoder on validation."
3. "Fixing the QLoRA recipe (completion-only loss, label never truncated) raised test ESCI
   macro-F1 on reranking candidates from 0.247 to 0.386 and removed the significant NDCG loss, but
   did not beat no-reranking (NDCG@10 −0.013 [−0.031, +0.005])."
4. "The generative Qwen reranker costs about 8.6 s per 40 candidates locally, about 64× the
   cross-encoder, while ranking worse."
5. The retrieval claims and the narrow historical classifier claim from the baseline report are
   unchanged.

## 10. Claims still unsupported

- That any Qwen or generative reranker improves ranking. No Qwen variant beat the hybrid order.
- That hard negatives help, in any form tested.
- That the truncation bug *caused* the historical failure. Qwen A differs in several ways at once:
  platform, training rows, loss masking and EOS.
- Full-catalog or fully judged NDCG, or any claim that treats unjudged candidates as known
  irrelevant.
- Production latency: only local M3/MPS wall time was measured, with no serving stack or
  concurrency. Hybrid retrieval latency is additional.
- Generalisation beyond one seed, one 200-query test sample and one candidate generator (hybrid
  top 40); multi-seed variance is unmeasured.
- That the cross-encoder is the best discriminative option. Only one untuned configuration was
  tried: no model search, and no tuning of the 0.8/0.2 blend weight.
- That class balance, C-prior correction, or hard negatives with matched hard positives would help.
  None was tested.

## Decision

- Hard negatives did not fix the Qwen failure (question 1 of the task): they made it significantly
  worse.
- The simpler reranker is better (question 2). The MiniLM cross-encoder beats every Qwen variant
  on classification, ranking and latency, and it is the only reranker whose full-rank NDCG gain
  over the hybrid order is significant.
- Generative classification is not justified by quality or cost here.
- Because the cross-encoder's judged-pool gain is not significant and this is a single run, the
  defensible engineering choice is:
  - ship **hybrid retrieval with the cross-encoder only as a measured, optional stage**;
  - keep **no reranker** as the safe default until a second seed or larger test sample confirms
    the gain;
  - **do not use the Qwen reranker**.

## Reproduce

```bash
export PYTHONPATH=src
python scripts/mine_esci_hard_negatives.py --shard-index 0 --num-shards 1
python scripts/mine_esci_hard_negatives.py --merge --num-shards 1
python scripts/build_reranker_training_data.py
scripts/run_ablation_qwen_chain.sh          # train A, B; validate checkpoints + historical adapter
scripts/run_ablation_ce_chain.sh            # cross-encoder on A and B
python scripts/select_reranker_models.py      # validation-only selection (committed before test)
scripts/run_ablation_test_chain.sh          # fixed-candidate test runs
python scripts/compare_reranker_ablation.py --systems qwen_A_balanced_random=qwen qwen_B_hard_negative=qwen cross_encoder=cross_encoder historical_qwen_rerun=qwen
python scripts/validate_reranker_ablation.py
```
