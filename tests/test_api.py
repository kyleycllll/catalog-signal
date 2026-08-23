from fastapi.testclient import TestClient
import pytest

import product_discovery.api as api_module
from product_discovery.api import app
from product_discovery.model_client import FineTunedModelUnavailable
from product_discovery.schemas import Citation, CitedAnswer, Intent, ModelHealth, ModelPrediction, RelevanceLabel
from product_discovery.store import SessionStore


class FakeFineTunedModel:
    configured = True

    async def health(self):
        return ModelHealth(status="ok", model_kind="fine_tuned_adapter", adapter_loaded=True, model_version="test-esci-lora", base_model="test-base")

    async def parse_intent(self, message, constraints, history):
        return Intent(category="backpack", required_attributes=["black"], max_price=80)

    async def rerank(self, query, products):
        return [ModelPrediction(product_id=p.id, label=RelevanceLabel.exact if p.category == "backpack" else RelevanceLabel.irrelevant, confidence=.95, rationale="Test adapter prediction") for p in products]

    async def answer(self, query, constraints, results):
        citations = [Citation(product_id=results[0].product.id, rank=1)] if results else []
        return CitedAnswer(answer="The first result is the strongest adapter-ranked match.", citations=citations)


class UnavailableModel:
    configured = False

    async def health(self):
        raise FineTunedModelUnavailable("adapter required")


@pytest.fixture(autouse=True)
def isolated_session_store(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "store", SessionStore(str(tmp_path / "sessions.json")))


def test_session_search_uses_fine_tuned_model_and_feedback():
    app.state.model_client = FakeFineTunedModel()
    client = TestClient(app)
    session = client.post("/sessions").json()["id"]
    response = client.post(
        f"/sessions/{session}/search",
        json={"message": "black university backpack under 80", "strategy": "bm25"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_kind"] == "fine_tuned_adapter"
    assert payload["model_version"] == "test-esci-lora"
    assert payload["results"][0]["relevance_label"] == "E"
    assert payload["retrieval_strategy"] == "bm25"
    assert payload["stage_latency_ms"]["candidate_generation"] >= 0
    product_id = payload["results"][0]["product"]["id"]
    assert client.post(f"/sessions/{session}/feedback", json={"product_id": product_id, "kind": "like"}).status_code == 200


def test_search_refuses_to_fallback_when_adapter_is_unavailable():
    app.state.model_client = UnavailableModel()
    client = TestClient(app)
    session = client.post("/sessions").json()["id"]
    response = client.post(f"/sessions/{session}/search", json={"message": "backpack", "strategy": "bm25"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "qlora_unavailable"
    assert "adapter required" in response.json()["detail"]["message"]


def test_dense_search_never_silently_downgrades_to_bm25():
    app.state.dense_index = None
    app.state.dense_index_error = "dense index unavailable for test"
    client = TestClient(app)
    session = client.post("/sessions").json()["id"]
    response = client.post(
        f"/sessions/{session}/search", json={"message": "backpack", "strategy": "dense", "rerank": False}
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "dense_index_unavailable"
