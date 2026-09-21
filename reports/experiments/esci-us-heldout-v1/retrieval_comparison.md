# Held-out retrieval comparison: BM25 vs dense vs hybrid RRF

Run `esci-us-heldout-v1`. The machine-readable source for every number here is
`retrieval_results.json`, with per-query rows in `retrieval_per_query.jsonl` and
per-query latencies in `retrieval_latency.csv`.

## Setup

- **Catalog:** every US-locale product in the official ESCI product corpus, **1,215,854 products**
  (`amazon-science/esci-data` @ `7916cdf`, SHA-256-verified against Git-LFS). It is the union of
  products judged for some ESCI query, not the whole Amazon catalog. It was built without reading
  labels.
- **Queries:** all **8,956** official ESCI test queries (US, `small_version=1`). All of them have at
  least one known E/S judgment. The 2,132 validation queries (non-test, query-disjoint) were used
  only for system selection.
- **Systems:** BM25 (k1 = 1.5, b = 0.75, over `product_text`). Dense uses
  `all-MiniLM-L6-v2` → L2-normalized vectors → FAISS `IndexFlatIP` (cosine); documents were encoded in
  fp16 on MPS and queries in fp32. Hybrid uses RRF with `score(d) = Σ 1/(60 + rank_i(d))` over the full
  BM25 and dense rankings. RRF k = 60 is the implemented default and was **not tuned**.
- **Exactness:** the indexed implementation is unit-tested to give the same rankings as
  `retrieval.retrieve` for all three strategies (`tests/test_indexed_retrieval.py`). BM25 scores are
  bit-identical.

## Metric semantics (partial judgments)

ESCI judges about 20 products per query. Only **13–15% of the retrieved top 40 carry any judgment for
the query** (`judged_at_40`). Therefore:

- **Recall@K** is *recall of known judged E/S positives*: the known positives found in the original
  top K, divided by all known positives for that query. It is not catalog-wide recall.
- **HitRate@K** means at least one known E/S positive in the top K.
- **Precision@K** counts unjudged products as non-relevant, so it is a heavy lower bound and is not
  used for claims.
- **MRR** uses the original rank of the first known E/S within the top 40.

## Results (8,956 test queries)

| System | Recall@10 | Recall@40 | HitRate@10 | HitRate@40 | MRR | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.1541 | 0.2880 | 0.6567 | 0.7921 | 0.4543 | 14.9 ms | 51.4 ms |
| Dense MiniLM + FAISS | 0.1292 | 0.2549 | 0.6112 | 0.7644 | 0.4059 | 38.3 ms | 97.4 ms |
| **Hybrid RRF** | **0.1649** | **0.3224** | **0.6967** | **0.8342** | **0.4844** | 430.3 ms* | 833.5 ms* |

Latency is measured one query at a time on an Apple M3 (16 GB) under memory pressure. Dense and hybrid
include query embedding on MPS. \*The hybrid wall time is dominated by
`exact_rank_lookup_and_fusion` (p50 361.6 ms). That step computes each fused candidate's exact rank
in the *full* 1.2M-item BM25 and dense rankings, purely so the evaluation reproduces `retrieve()`
exactly. The measured component stages have these p50 values: BM25 top-1000 in 14.3 ms, query
embedding in 14.2 ms, and FAISS top-1000 in 26.3 ms. A production fusion over the top-1000 lists
would cost roughly their sum. That figure is an **estimate from measured stages, not a measured
end-to-end latency**.

### Paired test comparisons (95% paired bootstrap over queries, 2,000 resamples)

| Comparison | ΔRecall@10 | ΔRecall@40 | ΔHitRate@40 | ΔMRR | Queries better / worse / tied (Recall@40) |
| --- | ---: | ---: | ---: | ---: | --- |
| Hybrid − BM25 | +0.0109 [0.0088, 0.0130] | **+0.0345 [0.0317, 0.0369]** | +0.0421 [0.0357, 0.0481] | +0.0300 [0.0238, 0.0366] | 3,330 / 1,634 / 3,992 |
| Dense − BM25 | −0.0249 [−0.0277, −0.0218] | −0.0331 [−0.0372, −0.0289] | −0.0277 [−0.0366, −0.0189] | −0.0484 [−0.0570, −0.0395] | 2,573 / 3,871 / 2,512 |
| Hybrid − Dense | +0.0358 [0.0335, 0.0380] | +0.0675 [0.0646, 0.0704] | +0.0698 [0.0632, 0.0768] | +0.0784 [0.0715, 0.0852] | 4,465 / 1,194 / 3,297 |

## Answers

- **Does hybrid beat BM25?** Yes, on every metric, and every interval excludes zero. Recall@40 over
  known positives rises from 0.2880 to 0.3224 (+0.0345; +12% relative), and MRR rises from 0.4543 to
  0.4844. Hybrid is worse than BM25 on 1,634 queries (18%) by Recall@40.
- **Does dense help?** Not on its own: dense alone is significantly *worse* than BM25 on all
  metrics. It helps **as a complementary signal**. Dense beats BM25 on 2,573 queries, and fusing it
  lifts every metric above both single systems.
- **Selection:** by the pre-registered rule (validation Recall@40), hybrid won with 0.3200 on
  validation, against BM25 0.2842 and dense 0.2536. Its test top-40 candidates were frozen to
  `fixed_candidates_test.jsonl` (SHA-256 `44201cf3…`) for the reranking experiment.

## Where each system wins (test Recall@40 by query slice)

| Slice | n | BM25 | Dense | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| 1–2 tokens | 1,755 | 0.2452 | 0.1989 | 0.2775 |
| 3–4 tokens | 4,582 | 0.2914 | 0.2640 | 0.3307 |
| 5–7 tokens | 2,355 | 0.3122 | 0.2797 | 0.3413 |
| 8+ tokens | 264 | 0.2958 | 0.2473 | 0.3086 |
| Contains a digit | 1,834 | 0.3218 | 0.2934 | 0.3601 |
| Contains not/without/no | 723 | 0.2750 | 0.2284 | 0.2882 |

BM25 leads dense in every slice. Hybrid leads both in every slice; its gain over BM25 is smallest for
negation queries and very long queries.

On 155 queries BM25 finds no known positive while dense finds at least 30%. On 207 queries the reverse
holds. On 142 queries hybrid loses more than 0.2 Recall@40 relative to BM25.

## Representative failures (deterministic sample, test queries)

**Lexical (BM25) failures**, where the query and product use different surface forms:

- `post-partum depression workbook`: BM25 Recall@40 is 0.00 and dense is 1.00. The tokenizer splits
  "post-partum" into `post`/`partum`, while the relevant product says "Postpartum", so BM25 ranks a
  "Post-Partum Bracelet" and generic depression workbooks. Dense matches the concept.
- `airmax red`: BM25 is 0.00 and dense is 0.44. The token `airmax` exactly matches Ubiquiti "airMAX"
  radios, while the relevant Nike products say "Air Max". Hybrid only partly recovers, to 0.19,
  because BM25's confident wrong list still contributes.
- `* batteries not included`: BM25 is 0.00 and dense is 0.44. The query consists of very common
  tokens, so BM25 returns power tools and remotes. Dense ranks the film title first.

**Semantic (dense) failures**, involving rare entities, misspellings and short generic queries:

- `king dedede chrisymas`: dense is 0.00 and BM25 is 0.64. The rare character name and the misspelling
  push MiniLM toward unrelated media titles ("Desierto", "The King and I"). BM25's exact match on
  `dedede` finds the Kirby merchandise.
- `name tag clip`: dense is 0.00 and BM25 is 0.67. Dense returns video titles beginning "Clip: …",
  while BM25 finds badge holders.
- `chase dreams not cowboys`: dense is 0.00 and BM25 is 0.67. Dense follows "cowboys" and "dream
  chasers". Hybrid is also 0.00 here, so fusion can lose a correct BM25 list.

**Hybrid losing to BM25:** on `makita impact drill` (BM25 0.44 → hybrid 0.19) and
`10oz tervis cups without lid` (0.85 → 0.53), dense promotes topically similar but wrong-type products
(drill bits, a 24 oz lid) into the fused top 40.

## Limitations

- Recall and HitRate count only **known** ESCI judgments. Unjudged retrieved products may be
  relevant, so absolute values are lower bounds. Comparisons between systems are also affected by
  judgment bias: ESCI's pools came from Amazon's production search, which may favour lexical matches.
- There is a single run and a single embedding model. RRF k = 60 and fusion depth were not tuned,
  which was deliberate.
- Document embeddings are fp16 (minimum cosine 0.9995 to fp32 on a 3k sample) while queries are fp32.
- The application's `retrieve()` re-tokenizes the whole catalog per query and cannot serve 1.2M
  products interactively. The benchmark uses an equivalence-tested indexed implementation.
