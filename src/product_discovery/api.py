"""HTTP surface for the one production product-search architecture."""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    QueryAnalysisResponse,
    RetrievalDecisionResponse,
    SearchRequest,
    SearchResponse,
)
from .search_pipeline import SearchPipeline, load_production_pipeline


logger = logging.getLogger("product_discovery.search")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load immutable serving assets exactly once before accepting requests."""
    if getattr(application.state, "pipeline", None) is None:
        try:
            application.state.pipeline = load_production_pipeline()
            application.state.startup_error = None
            pipeline = application.state.pipeline
            logger.info(
                "search_assets_loaded %s",
                json.dumps(
                    {
                        "catalog_products": len(pipeline.catalog),
                        "model_version": pipeline.model_version,
                        "index_version": pipeline.index_version,
                    }
                ),
            )
        except Exception as exc:  # Keep liveness available; readiness reports the actionable failure.
            application.state.pipeline = None
            application.state.startup_error = str(exc)
            logger.exception("search_assets_failed_to_load")
    yield


app = FastAPI(title="Adaptive Product Search", version="1.0.0", lifespan=lifespan)
app.state.pipeline = None
app.state.startup_error = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ready_pipeline(request: Request) -> SearchPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        message = getattr(request.app.state, "startup_error", None) or "Search assets are still loading."
        raise HTTPException(status_code=503, detail={"code": "search_not_ready", "message": message})
    return pipeline


@app.get("/health")
def health(request: Request):
    """Liveness status; use ``/ready`` for dependency/readiness checks."""
    pipeline = getattr(request.app.state, "pipeline", None)
    return {
        "status": "ok" if pipeline is not None else "degraded",
        "architecture": "query_understanding->indexed_hybrid->cross_encoder->constraint_adjustment",
        "catalog_products": len(pipeline.catalog) if pipeline is not None else None,
        "startup_error": getattr(request.app.state, "startup_error", None),
    }


@app.get("/ready")
def ready(request: Request):
    pipeline = ready_pipeline(request)
    return {
        "status": "ready",
        "catalog_products": len(pipeline.catalog),
        "model_version": pipeline.model_version,
        "index_version": pipeline.index_version,
    }


@app.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, request: Request):
    """Run the single normal product-search path with fixed internal decisions."""
    pipeline = ready_pipeline(request)
    request_id = str(uuid4())
    started = time.perf_counter()
    response = pipeline.search(body.query)
    total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
    stages = {**response.stage_latency_ms, "total_request": total_latency_ms}
    retrieval_latency_ms = round(
        stages["bm25_retrieval"] + stages["dense_retrieval"] + stages["fusion"], 2
    )
    logger.info(
        "search_completed %s",
        json.dumps(
            {
                "request_id": request_id,
                "query_latency_ms": stages["query_understanding"],
                "retrieval_latency_ms": retrieval_latency_ms,
                "reranker_latency_ms": stages["cross_encoder_reranking"],
                "total_latency_ms": total_latency_ms,
                "candidate_count": response.candidate_count,
                "bm25_weight": response.retrieval_decision.bm25_weight,
                "dense_weight": response.retrieval_decision.dense_weight,
                "model_version": pipeline.model_version,
                "index_version": pipeline.index_version,
            }
        ),
    )
    return SearchResponse(
        query=body.query,
        query_analysis=QueryAnalysisResponse.model_validate(response.analysis.model_dump()),
        retrieval_decision=RetrievalDecisionResponse.model_validate(
            response.retrieval_decision.model_dump()
        ),
        results=response.results,
        latency_ms=total_latency_ms,
        request_id=request_id,
        candidate_count=response.candidate_count,
        stage_latency_ms=stages,
        model_version=pipeline.model_version,
        index_version=pipeline.index_version,
    )
