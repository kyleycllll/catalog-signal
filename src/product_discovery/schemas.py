from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RelevanceLabel(str, Enum):
    exact = "E"
    substitute = "S"
    complement = "C"
    irrelevant = "I"


RetrievalStrategy = Literal["bm25", "dense", "hybrid"]


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


class SearchRequest(BaseModel):
    """The deliberately small public contract for production product search."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)


class RankedResult(BaseModel):
    product: Product
    score: float
    scores: dict[str, float]
    relevance_label: RelevanceLabel | None = None
    explanation: str = ""


class QueryAnalysisResponse(BaseModel):
    original_query: str
    normalized_query: str
    retrieval_query: str
    tokens: list[str]
    brands: list[str]
    colors: list[str]
    model_tokens: list[str]
    numeric_tokens: list[str]
    exclusions: list[str]
    contains_negation: bool


class RetrievalDecisionResponse(BaseModel):
    """The internal, deterministic adaptive-fusion decision for a search."""

    bm25_weight: float
    dense_weight: float
    lexical_rarity: float | None
    reasons: list[str]


class SearchResponse(BaseModel):
    query: str
    query_analysis: QueryAnalysisResponse
    retrieval_decision: RetrievalDecisionResponse
    results: list[RankedResult]
    latency_ms: float
    request_id: str
    candidate_count: int
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    model_version: str
    index_version: str


class EvaluationCase(BaseModel):
    query: str
    relevant_product_ids: list[str]
    constraints: Constraints = Field(default_factory=Constraints)


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1)
