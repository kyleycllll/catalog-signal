from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .agent import select_tools
from .embeddings import DenseIndexUnavailable, FaissTextEmbeddingIndex
from .memory import append_history
from .model_client import FineTunedModelClient, FineTunedModelUnavailable, ModelClient
from .retrieval import rank_without_reranker, rerank_with_sft, retrieve, retrieve_candidates
from .schemas import (
    Citation,
    EvaluationRequest,
    FeedbackEvent,
    Intent,
    Product,
    SearchRequest,
    SearchResponse,
    SessionState,
)
from .store import SessionStore


load_dotenv()
logger = logging.getLogger("product_discovery.search")

app = FastAPI(title="Adaptive Product Search", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_catalog() -> tuple[list[Product], Path, bool]:
    configured = os.getenv("PRODUCT_CATALOG_PATH")
    choices = (
        [Path(configured), Path("data/sample_esci_catalog.json")]
        if configured
        else [Path("data/processed/esci_products.json"), Path("data/sample_esci_catalog.json")]
    )
    for path in choices:
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            return [Product.model_validate(row) for row in rows], path, path.name == "sample_esci_catalog.json"
    raise RuntimeError("No product catalog found. Run scripts/prepare_esci.py or add data/sample_esci_catalog.json.")


def load_dense_index(products: list[Product]) -> tuple[FaissTextEmbeddingIndex | None, str | None, Path]:
    path = Path(os.getenv("DENSE_INDEX_PATH", "data/indexes/text_embeddings.faiss"))
    try:
        return FaissTextEmbeddingIndex.load(path, products), None, path
    except (DenseIndexUnavailable, OSError, ValueError, KeyError, TypeError) as exc:
        return None, str(exc), path


catalog, catalog_path, sample_catalog = load_catalog()
dense_index, dense_index_error, dense_index_path = load_dense_index(catalog)
store = SessionStore()
app.state.model_client = FineTunedModelClient()
app.state.dense_index = dense_index
app.state.dense_index_error = dense_index_error
app.state.dense_index_path = dense_index_path


def model_client(request: Request) -> ModelClient:
    return request.app.state.model_client


def requested_dense_index(request: Request):
    index = request.app.state.dense_index
    if index is None:
        raise DenseIndexUnavailable(request.app.state.dense_index_error or "Dense index is unavailable")
    return index


def unavailable(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=503, detail={"code": code, "message": message})


@app.get("/health")
def health():
    index = app.state.dense_index
    return {
        "status": "ok",
        "catalog_products": len(catalog),
        "catalog_path": str(catalog_path),
        "sample_catalog": sample_catalog,
        "fine_tuned_model_configured": bool(getattr(app.state.model_client, "configured", True)),
        "dense_index_available": index is not None,
        "dense_index_path": str(app.state.dense_index_path),
        "dense_index_version": getattr(getattr(index, "metadata", None), "version", None),
        "dense_index_error": app.state.dense_index_error,
    }


@app.get("/model-info")
async def model_info(request: Request):
    try:
        remote = await model_client(request).health()
    except FineTunedModelUnavailable as exc:
        raise unavailable("qlora_unavailable", str(exc)) from exc
    return remote.model_dump()


@app.post("/sessions", response_model=SessionState)
def create_session():
    state = SessionState(id=str(uuid4()))
    store.save(state)
    return state


def session_or_404(session_id: str) -> SessionState:
    state = store.get(session_id)
    if not state:
        raise HTTPException(404, "Unknown session")
    return state


@app.get("/sessions/{session_id}", response_model=SessionState)
def get_session(session_id: str):
    return session_or_404(session_id)


@app.post("/sessions/{session_id}/search", response_model=SearchResponse)
async def search(session_id: str, body: SearchRequest, request: Request):
    started = time.perf_counter()
    stages: dict[str, float] = {}
    state = session_or_404(session_id)
    model_version: str | None = None
    index_version: str | None = None
    try:
        # The adapter is trained for ESCI relevance, not intent extraction or prose generation.
        intent = Intent(search_strategy=[body.strategy, "sft_esci_reranker"] if body.rerank else [body.strategy])
        recent_context = " ".join(row["content"] for row in state.history[-4:] if row["role"] == "user")
        query = f"{recent_context} {body.message}".strip()

        retrieval_started = time.perf_counter()
        index = requested_dense_index(request) if body.strategy in {"dense", "hybrid"} else None
        candidates = retrieve(
            query,
            catalog,
            state.constraints,
            strategy=body.strategy,
            limit=body.candidate_k,
            dense_index=index,
        )
        stages["candidate_generation"] = round((time.perf_counter() - retrieval_started) * 1000, 2)
        index_version = getattr(getattr(index, "metadata", None), "version", None)

        if body.rerank:
            rerank_started = time.perf_counter()
            health_state = await model_client(request).health()
            ranked = await rerank_with_sft(query, candidates.products, candidates.scores, state, model_client(request))
            stages["qlora_reranking"] = round((time.perf_counter() - rerank_started) * 1000, 2)
            model_version = health_state.model_version
        else:
            ranked = rank_without_reranker(candidates)
    except DenseIndexUnavailable as exc:
        logger.warning("search_failed %s", json.dumps({"code": "dense_index_unavailable", "strategy": body.strategy}))
        raise unavailable("dense_index_unavailable", str(exc)) from exc
    except FineTunedModelUnavailable as exc:
        logger.warning("search_failed %s", json.dumps({"code": "qlora_unavailable", "strategy": body.strategy}))
        raise unavailable("qlora_unavailable", str(exc)) from exc

    ranked = ranked[:10]
    if body.rerank:
        useful = [row for row in ranked if row.relevance_label and row.relevance_label.value != "I"][:3]
        if useful:
            label_names = {"E": "exact match", "S": "substitute", "C": "complement"}
            descriptions = [
                f"[{index}] {row.product.title} ({label_names[row.relevance_label.value]})"
                for index, row in enumerate(useful, 1)
            ]
            answer = "The fine-tuned ESCI reranker selected " + "; ".join(descriptions) + "."
            citations = [Citation(product_id=row.product.id, rank=index) for index, row in enumerate(useful, 1)]
        else:
            answer = "The fine-tuned ESCI reranker did not identify a relevant catalog match."
            citations = []
    else:
        answer = f"Showing {body.strategy} candidate-retrieval results without QLoRA reranking."
        citations = []

    tools = select_tools(intent, body.strategy, body.rerank)
    trace_id = str(uuid4())
    append_history(state, "user", body.message)
    append_history(state, "assistant", answer)
    state.prior_result_ids = [row.product.id for row in ranked]
    total_latency = round((time.perf_counter() - started) * 1000, 2)
    state.traces.append(
        {
            "trace_id": trace_id,
            "tools": tools,
            "retrieval_strategy": body.strategy,
            "reranking_applied": body.rerank,
            "candidate_count": len(candidates.products),
            "stage_latency_ms": stages,
            "index_version": index_version,
            "model_version": model_version,
            "constraints": state.constraints.model_dump(),
        }
    )
    store.save(state)
    logger.info(
        "search_completed %s",
        json.dumps(
            {
                "trace_id": trace_id,
                "strategy": body.strategy,
                "rerank": body.rerank,
                "candidate_count": len(candidates.products),
                "latency_ms": total_latency,
            }
        ),
    )
    return SearchResponse(
        answer=answer,
        citations=citations,
        parsed_intent=intent,
        persistent_constraints=state.constraints,
        tools_selected=tools,
        results=ranked,
        model_version=model_version,
        model_kind="fine_tuned_adapter" if body.rerank else "not_used",
        latency_ms=total_latency,
        trace_id=trace_id,
        retrieval_strategy=body.strategy,
        reranking_applied=body.rerank,
        candidate_count=len(candidates.products),
        stage_latency_ms=stages,
        index_version=index_version,
    )


@app.post("/sessions/{session_id}/feedback", response_model=SessionState)
def feedback(session_id: str, event: FeedbackEvent):
    state = session_or_404(session_id)
    if event.product_id not in {product.id for product in catalog}:
        raise HTTPException(404, "Unknown product")
    state.feedback.append(event)
    if event.kind.value in {"like", "save"}:
        state.selected_product_ids.append(event.product_id)
    if event.kind.value in {"dislike", "not_relevant", "wrong_style", "wrong_category"}:
        state.rejected_product_ids.append(event.product_id)
    store.save(state)
    return state


@app.get("/products/{product_id}", response_model=Product)
def product_lookup(product_id: str):
    for product in catalog:
        if product.id == product_id:
            return product
    raise HTTPException(404, "Unknown product")


@app.post("/evaluate/bm25-baseline")
def evaluate_bm25_baseline(body: EvaluationRequest):
    """Small API baseline; use scripts/run_search_experiments.py for official reports."""
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    for case in body.cases:
        candidates, scores = retrieve_candidates(case.query, catalog, case.constraints, limit=len(catalog))
        ranked_ids = [product.id for product in sorted(candidates, key=lambda product: scores[product.id], reverse=True)]
        hits_at_5 += bool(set(ranked_ids[:5]) & set(case.relevant_product_ids))
        first = next(
            (index + 1 for index, value in enumerate(ranked_ids) if value in case.relevant_product_ids), None
        )
        reciprocal_ranks.append(1 / first if first else 0.0)
    total = len(body.cases)
    return {
        "system": "bm25_baseline_only",
        "cases": total,
        "recall_at_5": hits_at_5 / total,
        "mrr": sum(reciprocal_ranks) / total,
    }
