# Fixed-candidate reranking: hybrid order vs the same candidates after Qwen QLoRA

Run `esci-us-heldout-v1`. The source for every number here is `reranking_results.json`, with
per-query rows (original IDs, known labels, Qwen predictions, gains, reranked order, and pre/post
metrics) in `reranking_per_query.jsonl` and latencies in `reranking_latency.csv`.

## Setup

- **Candidates:** the top 40 of **hybrid RRF**, the system selected on *validation* queries, for each
  test query. They were frozen in `fixed_candidates_test.jsonl` (SHA-256 `44201cf3…`), and the hash is
  checked before reranking.
- **Queries:** 200 official test queries, chosen by `sha256(42:query_id)` order. The sample was fixed
  before any reranking output existed. It was reduced from the planned 300 to 200 because local
  throughput measured 1.94 pairs/s, below a threshold set in advance. 200 completed and 0 failed.
- **Model:** Qwen2.5-0.5B-Instruct plus the saved QLoRA adapter (`models/esci-qwen-lora`, SHA-256
  `72fca854…`). It performs **pointwise generative E/S/C/I classification**, not pairwise,
  cross-encoder or learning-to-rank. It uses the notebook's exact prompt (tested byte-identical),
  greedy decoding, and a fallback to I if the output doesn't parse.
- **Inference fidelity:** the 4-bit NF4 double-quantized base from Colab is emulated locally with fp16
  compute. On the 500 saved official-test pairs this agrees with the saved Colab adapter predictions
  **97.4%** (macro-F1 0.2419 vs 0.2417) and with the frozen base 99.8% (`classifier_reproduction.json`).
  A full-precision bf16 base agreed only 63.4% (macro-F1 0.146), so it was rejected.
- **Ordering:** the production `retrieval.rerank_with_sft` formula,
  `0.8·gain/3 + 0.2·normalized RRF score`, with gains E = 3, S = 2, C = 1, I = 0. The label dominates;
  ties keep retrieval order.
- **Invariant:** for every query, the reranked IDs are asserted to be a permutation of the fixed
  candidate IDs. Only the order changes.

## Metric families (ESCI judgments are partial)

Only **13.4%** of candidates carry a judgment for their query.

- **Full-rank candidate metrics** keep all 40 positions and give unjudged products gain 0:
  - NDCG@K (ideal = all judged). The ideal DCG comes from all known judgments for the query.
  - NDCG@K (ideal = candidates). The ideal DCG comes from the fixed candidates' known labels. This
    shows how well the fixed set is ordered; the denominator is identical before and after.
  - MRR is the reciprocal rank of the first known E/S. P@5 counts known E/S in the top 5 out of 5.
- **Judged-pool diagnostics** drop unjudged candidates *after* ranking. They show whether the
  result is only an artifact of unjudged items. They are **not** full NDCG.

## Results (200 test queries; 95% paired bootstrap intervals, 2,000 resamples)

| Metric | Pre-rerank (hybrid) | Post-Qwen | Δ [95% CI] | Improved / worsened / unchanged |
| --- | ---: | ---: | --- | --- |
| NDCG@5 (ideal = all judged) | 0.2545 | 0.1982 | **−0.0562 [−0.0848, −0.0311]** | 33 / 60 / 107 |
| NDCG@10 (ideal = all judged) | 0.2410 | 0.1926 | **−0.0484 [−0.0695, −0.0292]** | 48 / 73 / 79 |
| NDCG@5 (ideal = candidates) | 0.3054 | 0.2356 | −0.0698 [−0.1092, −0.0335] | 33 / 60 / 107 |
| NDCG@10 (ideal = candidates) | 0.3354 | 0.2643 | −0.0711 [−0.1061, −0.0380] | 48 / 73 / 79 |
| MRR | 0.4278 | 0.3928 | −0.0351 [−0.0747, +0.0032] | 44 / 46 / 110 |
| P@5 (unjudged = non-relevant) | 0.2600 | 0.2090 | −0.0510 [−0.0760, −0.0270] | 17 / 53 / 130 |
| *Judged-pool NDCG@5 (diagnostic)* | 0.7018 | 0.6861 | −0.0157 [−0.0316, −0.0005] | 26 / 47 / 127 |
| *Judged-pool NDCG@10 (diagnostic)* | 0.7297 | 0.7159 | −0.0139 [−0.0256, −0.0025] | 30 / 50 / 120 |
| *Judged-pool MRR (diagnostic)* | 0.7517 | 0.7521 | +0.0004 [−0.0175, +0.0171] | 8 / 6 / 186 |
| *Judged Precision@5 (diagnostic)* | 0.7017 | 0.7007 | −0.0010 [−0.0100, +0.0070] | 6 / 7 / 187 |

Reference (uses test labels, so not a system): an oracle reordering of the same fixed candidates by
known gain reaches NDCG@10 0.5021 (ideal = all judged) and 0.7950 (ideal = candidates). There is
substantial headroom in ordering; Qwen moves away from it.

## Answers

1. **Did Qwen improve NDCG@5?** No. It **worsened** from 0.2545 to 0.1982 (−0.056; the CI excludes 0).
2. **Did Qwen improve NDCG@10?** No. It **worsened** from 0.2410 to 0.1926 (−0.048; the CI excludes 0).
3. **Did it improve MRR?** No. It fell from 0.4278 to 0.3928 (−0.035), but the CI [−0.075, +0.003]
   includes 0, so MRR shows no significant change.
4. **Queries improved (NDCG@10):** 48 of 200.
5. **Queries worsened (NDCG@10):** 73 of 200, with 79 unchanged.
6. **Which ESCI classes are confused?** On the 1,070 judged candidates, accuracy is 0.358 and
   macro-F1 is 0.247. The confusion matrix (rows = gold, columns = predicted E/S/C/I):

   | Gold \ Pred | E | S | C | I |
   | --- | ---: | ---: | ---: | ---: |
   | E (544) | 304 | 57 | **127** | 56 |
   | S (397) | **243** | 46 | 79 | 29 |
   | C (39) | 14 | 3 | 19 | 3 |
   | I (90) | **48** | 4 | 24 | 14 |

   - 34% of true exact matches are labelled C or I (183/544), and 27% of true substitutes are too.
   - Most substitutes are labelled E (243/397, harmless for ordering).
   - Most true irrelevants are labelled E (48/90).
   - On the 6,930 unjudged candidates, Qwen predicts E 59%, C 19%, I 14% and S 8%. A median of 69% of
     each query's candidates are labelled E or S.
7. **What failure pattern is visible?** The classifier has little discriminative signal and many
   wrong low labels on true exact matches.
   - Because the label dominates the score, every E→C/I error on a highly ranked exact match pushes it
     below many products labelled E, most of which are unjudged or irrelevant.
   - Of the 260 known E/S products in the pre-rerank top 5, **116 were pushed out of the top 5**.
   - Examples:
     - `habanero hot sauce cholula organic` (NDCG@10 0.877 → 0.221): three Cholula habanero exact
       matches were labelled **C** and demoted below a Yellowbird organic sauce.
     - `vsco collage poster` (0.599 → 0.000): the one known exact match, labelled C, fell out of the
       top 10 behind generic collage kits labelled E.
     - `xyliwhite mouthwash` (0.779 → 0.258): four XyliWhite exact matches were labelled S or I and
       overtaken by a XyliWhite toothpaste and generic mouthwashes labelled E.
   - Improvements happen when retrieval put an irrelevant item first. For `rue`, a gift card labelled
     I dropped and rue-herb products labelled E rose (0.221 → 0.580).
   - The judged-pool diagnostics also decline, so the loss is **not** only an artifact of unjudged
     candidates being promoted.
8. **Reranking latency:** p50 is **8.84 s** per query (40 candidates), p95 is 15.42 s and the mean is
   9.07 s. That is about 220 ms per candidate on Apple M3 MPS with batched greedy generation (batch 8)
   and emulated NF4 weights. The Colab serving endpoint generates one product at a time over HTTP, so
   this is not its latency. Retrieval itself takes 15–430 ms.
9. **Is the ranking change worth the latency?** No. It adds about 9 s per query and makes NDCG@5 and
   NDCG@10 significantly worse. With this adapter, retrieval-only hybrid ranking is both better and
   about 20× faster.

## Limitations

- The 200-query sample gives wide intervals, but the NDCG losses are significant and consistent across
  metric families.
- Only 13.4% of candidates are judged. The unjudged = 0 metrics are lower bounds, and Qwen's many E
  labels on unjudged items may include genuinely relevant products. The judged-pool diagnostics, which
  are immune to that, also decline or stay flat.
- Inference used a faithful local emulation (97.4% agreement), not the original bitsandbytes CUDA
  kernels.
- This evaluates the historical 10k-example, one-epoch adapter under the current production score
  formula. It says nothing about better-trained rerankers, which are out of scope for this handoff.
