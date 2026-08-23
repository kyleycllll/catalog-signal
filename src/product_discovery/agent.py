"""Auditable search-agent planning and reference resolution."""
from __future__ import annotations

from .schemas import Intent, RetrievalStrategy, SessionState


def select_tools(
    intent: Intent, strategy: RetrievalStrategy = "hybrid", rerank: bool = True
) -> list[str]:
    tools = [f"{strategy}_candidate_retrieval"]
    if any([intent.category, intent.required_attributes, intent.excluded_attributes, intent.max_price is not None, intent.min_price is not None]):
        tools.insert(0, "metadata_filter")
    if intent.referenced_result is not None:
        tools.insert(0, "conversation_memory_lookup")
    if rerank:
        tools.append("sft_esci_reranker")
    tools.append("grounded_result_summary")
    return tools


def referenced_product_id(intent: Intent, state: SessionState) -> str | None:
    if intent.referenced_result is None:
        return None
    position = intent.referenced_result - 1
    if position < 0 or position >= len(state.prior_result_ids):
        return None
    return state.prior_result_ids[position]
