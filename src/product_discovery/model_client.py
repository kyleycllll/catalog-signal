from __future__ import annotations

import os
from typing import Protocol

import httpx

from .schemas import CitedAnswer, Constraints, Intent, ModelHealth, ModelPrediction, Product, RankedResult


class FineTunedModelUnavailable(RuntimeError):
    """Raised when the required adapter-backed inference service cannot be used."""


class ModelClient(Protocol):
    async def health(self) -> ModelHealth: ...
    async def parse_intent(self, message: str, constraints: Constraints, history: list[dict[str, str]]) -> Intent: ...
    async def rerank(self, query: str, products: list[Product]) -> list[ModelPrediction]: ...
    async def answer(self, query: str, constraints: Constraints, results: list[RankedResult]) -> CitedAnswer: ...


class FineTunedModelClient:
    """HTTP client for the Colab service that has a PEFT/LoRA adapter loaded."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 90.0):
        self.base_url = (base_url if base_url is not None else os.getenv("FINETUNED_MODEL_URL", "")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("FINETUNED_MODEL_API_KEY", "")
        self.timeout = timeout
        self._verified: ModelHealth | None = None

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _request(self, method: str, payload: dict | None = None) -> dict:
        if not self.configured:
            raise FineTunedModelUnavailable(
                "FINETUNED_MODEL_URL is not set. Run the Colab serving cell and copy its HTTPS URL into .env."
            )
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers()) as client:
                response = await client.request("GET" if method == "health" else "POST", f"{self.base_url}/{method}", json=payload)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise FineTunedModelUnavailable(f"Fine-tuned model service failed at /{method}: {exc}") from exc

    async def health(self) -> ModelHealth:
        health = ModelHealth.model_validate(await self._request("health"))
        if not health.adapter_loaded:
            raise FineTunedModelUnavailable("The remote service is running, but its LoRA adapter is not loaded.")
        self._verified = health
        return health

    async def _ensure_adapter(self) -> ModelHealth:
        return self._verified or await self.health()

    async def parse_intent(self, message: str, constraints: Constraints, history: list[dict[str, str]]) -> Intent:
        await self._ensure_adapter()
        data = await self._request("parse-intent", {"message": message, "constraints": constraints.model_dump(), "history": history[-6:]})
        return Intent.model_validate(data)

    async def rerank(self, query: str, products: list[Product]) -> list[ModelPrediction]:
        await self._ensure_adapter()
        data = await self._request("rerank", {"query": query, "products": [p.model_dump() for p in products]})
        predictions = [ModelPrediction.model_validate(row) for row in data.get("predictions", data)]
        expected, received = {p.id for p in products}, {p.product_id for p in predictions}
        if expected != received:
            raise FineTunedModelUnavailable("Fine-tuned reranker returned incomplete or unknown product IDs.")
        return predictions

    async def answer(self, query: str, constraints: Constraints, results: list[RankedResult]) -> CitedAnswer:
        await self._ensure_adapter()
        data = await self._request(
            "answer",
            {"query": query, "constraints": constraints.model_dump(), "results": [r.model_dump(mode="json") for r in results[:5]]},
        )
        answer = CitedAnswer.model_validate(data)
        valid_ids = {row.product.id for row in results[:5]}
        if any(c.product_id not in valid_ids for c in answer.citations):
            raise FineTunedModelUnavailable("Answer contained a citation outside the retrieved context.")
        return answer
