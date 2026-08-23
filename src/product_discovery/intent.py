"""Intent extraction delegated exclusively to the adapter-backed model service."""
from __future__ import annotations

from .model_client import ModelClient
from .schemas import Constraints, Intent


async def extract_intent(message: str, current: Constraints, history: list[dict[str, str]], model: ModelClient) -> Intent:
    return await model.parse_intent(message, current, history)
