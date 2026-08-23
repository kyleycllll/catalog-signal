from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RelevanceLabel(str, Enum):
    exact = "E"
    substitute = "S"
    complement = "C"
    irrelevant = "I"


RetrievalStrategy = Literal["bm25", "dense", "hybrid"]


class Intent(BaseModel):
    category: str | None = None
    required_attributes: list[str] = Field(default_factory=list)
    preferred_attributes: list[str] = Field(default_factory=list)
    excluded_attributes: list[str] = Field(default_factory=list)
    max_price: float | None = None
    min_price: float | None = None
    referenced_result: int | None = None
    clarification_required: bool = False
    search_strategy: list[str] = Field(default_factory=lambda: ["bm25", "sft_reranker"])


class Product(BaseModel):
    id: str
    title: str
    description: str = ""
    category: str = "unknown"
    price: float | None = None
    colour: str | None = None
    brand: str | None = None
    material: str | None = None
    attributes: list[str] = Field(default_factory=list)
    bullet_points: str | None = None
    locale: str = "us"
    image_path: str | None = None


class Constraints(BaseModel):
    category: str | None = None
    required_attributes: list[str] = Field(default_factory=list)
    preferred_attributes: list[str] = Field(default_factory=list)
    excluded_attributes: list[str] = Field(default_factory=list)
    max_price: float | None = None
    min_price: float | None = None


class FeedbackKind(str, Enum):
    like = "like"
    dislike = "dislike"
    save = "save"
    not_relevant = "not_relevant"
    too_expensive = "too_expensive"
    wrong_style = "wrong_style"
    wrong_category = "wrong_category"


class FeedbackEvent(BaseModel):
    product_id: str
    kind: FeedbackKind
    note: str | None = None


class SessionState(BaseModel):
    id: str
    history: list[dict[str, str]] = Field(default_factory=list)
    context_summary: str = ""
    constraints: Constraints = Field(default_factory=Constraints)
    prior_result_ids: list[str] = Field(default_factory=list)
    selected_product_ids: list[str] = Field(default_factory=list)
    rejected_product_ids: list[str] = Field(default_factory=list)
    feedback: list[FeedbackEvent] = Field(default_factory=list)
    traces: list[dict[str, Any]] = Field(default_factory=list)


class SearchRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    strategy: RetrievalStrategy = "hybrid"
    rerank: bool = True
    candidate_k: int = Field(default=40, ge=1, le=200)


class ModelPrediction(BaseModel):
    product_id: str
    label: RelevanceLabel
    # Optional until the notebook produces a validation-calibrated probability estimate.
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str = ""


class RankedResult(BaseModel):
    product: Product
    score: float
    scores: dict[str, float]
    relevance_label: RelevanceLabel | None = None
    explanation: str = ""


class Citation(BaseModel):
    product_id: str
    rank: int = Field(ge=1)


class CitedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class SearchResponse(BaseModel):
    answer: str
    citations: list[Citation]
    parsed_intent: Intent
    persistent_constraints: Constraints
    tools_selected: list[str]
    results: list[RankedResult]
    model_version: str | None = None
    model_kind: Literal["fine_tuned_adapter", "not_used"] = "fine_tuned_adapter"
    latency_ms: float
    trace_id: str
    retrieval_strategy: RetrievalStrategy
    reranking_applied: bool
    candidate_count: int
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    index_version: str | None = None


class ModelHealth(BaseModel):
    status: Literal["ok"]
    model_kind: Literal["fine_tuned_adapter"]
    adapter_loaded: bool
    model_version: str
    base_model: str


class EvaluationCase(BaseModel):
    query: str
    relevant_product_ids: list[str]
    constraints: Constraints = Field(default_factory=Constraints)


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1)
