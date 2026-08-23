# Portfolio case study: retrieval quality under a constrained budget

## Problem

Lexical product search is fast and interpretable, but it cannot reliably distinguish an exact product from a close substitute, accessory, or irrelevant lexical match. This project asks whether a small two-stage system improves that ordering while making its costs and failures visible.

## Why this architecture

BM25 is the baseline because it is cheap, inspectable, and often competitive for product titles. MiniLM + FAISS adds semantic candidate coverage without an external vector service. Reciprocal Rank Fusion combines their independently useful signals without training another model. Qwen2.5 QLoRA then applies the expensive learned relevance judgment only to a bounded candidate set.

This design trades more index-build work and remote reranker latency for potentially better semantic ordering. It does not assume the more complex system wins: all stages are evaluated against BM25.

## Evaluation protocol

- Amazon ESCI US small-version data, query-disjoint train/validation, official test held out.
- Recall@K and MRR treat Exact and Substitute as acceptable; NDCG uses E/S/C/I gains 3/2/1/0.
- Compare BM25, dense, hybrid RRF, and each candidate generator with the same QLoRA reranker.
- Compare a matched label-balanced random training set against a hard-negative set that contains only known high-retrieval S/C/I judgments.
- Record config, split, sample size, model/index version, training time, p50/p95 retrieval and reranking latency, and failure state for every run.

## Verified findings

Pending. The repository has no official ESCI catalog, FAISS artifact, or completed adapter evidence. Results are reported only from generated experiment artifacts; unavailable runs are shown as unavailable rather than replaced by a fallback.

## Failure behavior

- An absent or catalog-mismatched FAISS index returns `dense_index_unavailable` for dense/hybrid requests.
- An absent adapter returns `qlora_unavailable` for requested reranking.
- BM25 remains available as an explicit baseline; neither failure silently changes the selected system.

## Limits and next step

ESCI has incomplete product judgments and no genuine user histories. Consequently this is a relevance-ranking study, not a personalized recommendation claim. The sensible next project would use an interaction dataset to evaluate recommendation and cold-start behavior on its own terms.
