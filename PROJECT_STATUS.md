# Project status

Implemented locally:

- Strict adapter-backed HTTP model client with no production fallback.
- BM25, dense FAISS, and deterministic hybrid-RRF candidate retrieval interfaces.
- QLoRA ESCI reranking using ordinal label gains rather than fabricated confidence.
- Configured experiment runner with metric, provenance, latency, and unavailable-run evidence.
- Matched known-label hard-negative mining for the QLoRA ablation.
- Stateful constraints, result references, feedback memory, and traces.
- Grounded summaries with citations derived from fine-tuned reranker outputs.
- Official ESCI parquet preparation and evaluation scripts.
- Colab QLoRA notebook with frozen-base versus adapter evaluation and serving.
- React interface with model status, relevance labels, scores, citations, and feedback.
- Automated tests, including a 503 no-fallback assertion.

Manual/external work still required:

- Prepare the official ESCI catalog and build the matching FAISS index before dense/hybrid evaluation.
- Run both Colab training variants on a GPU to create actual adapter weights and measured results.
- Keep its authenticated ngrok model endpoint running when using the local application.
- Optionally prepare the full local ESCI search catalog. The repository ships only a clearly marked six-row UI fixture.

No retrieval or training metric is claimed until the configured experiment or notebook has run.
