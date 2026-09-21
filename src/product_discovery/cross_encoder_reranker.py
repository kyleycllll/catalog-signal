"""Compact discriminative ESCI reranker: a MiniLM cross-encoder with a 4-way head.

One forward pass per (query, product) pair returns E/S/C/I probabilities. It can be
used two ways in ``retrieval.rerank_with_sft``-style scoring: the argmax label (same
interface as the Qwen classifier), or the expected gain ``sum_c p(c) * gain(c)``,
which gives a continuous relevance score. Product text uses the same fields and
900-character caps as the Qwen prompt.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .evaluation import LABELS
from .local_reranker import _clean, prompt_row
from .reranker_training import expected_gain
from .schemas import ModelHealth, ModelPrediction, Product, RelevanceLabel


def cross_encoder_product_text(row: Mapping[str, Any]) -> str:
    parts = [
        _clean(row.get("product_title")),
        f"Brand: {_clean(row.get('product_brand'))}",
        f"Color: {_clean(row.get('product_color'))}",
        _clean(row.get("product_bullet_point"))[:900],
        _clean(row.get("product_description"))[:900],
    ]
    return " | ".join(part for part in parts if part and not part.endswith(": "))


class CrossEncoderReranker:
    """Batched inference; ``rerank`` exposes the ModelClient surface (argmax label)."""

    def __init__(self, model_dir: str, device: str | None = None, batch_size: int = 64, max_length: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        if [self.model.config.id2label[i] for i in range(len(LABELS))] != list(LABELS):
            raise ValueError("Cross-encoder head must be ordered E, S, C, I")

    def probabilities(self, queries: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        torch = self.torch
        output: list[list[float]] = []
        texts = [cross_encoder_product_text(row) for row in rows]
        for start in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer(
                list(queries[start : start + self.batch_size]), texts[start : start + self.batch_size],
                truncation="only_second", max_length=self.max_length, padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits.float()
            output.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return output

    def score_products(self, query: str, products: list[Product]) -> tuple[dict[str, str], dict[str, float], dict[str, list[float]]]:
        probs = self.probabilities([query] * len(products), [prompt_row(query, p) for p in products])
        labels = {p.id: LABELS[max(range(len(LABELS)), key=row.__getitem__)] for p, row in zip(products, probs)}
        gains = {p.id: expected_gain(row) for p, row in zip(products, probs)}
        return labels, gains, {p.id: row for p, row in zip(products, probs)}

    async def health(self) -> ModelHealth:
        return ModelHealth(status="ok", model_kind="fine_tuned_adapter", adapter_loaded=True,
                           model_version=f"cross-encoder:{self.model_dir}", base_model=self.model_dir)

    async def rerank(self, query: str, products: list[Product]) -> list[ModelPrediction]:
        labels, _, probs = self.score_products(query, products)
        return [
            ModelPrediction(product_id=p.id, label=RelevanceLabel(labels[p.id]), confidence=max(probs[p.id]),
                            rationale=f"Cross-encoder ESCI prediction: {labels[p.id]}")
            for p in products
        ]
