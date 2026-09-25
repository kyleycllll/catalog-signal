from fastapi.testclient import TestClient
import pytest

from product_discovery.api import app
from product_discovery.indexed_retrieval import RetrievalResult
from product_discovery.schemas import Product
from product_discovery.search_pipeline import ProductListCatalog, SearchPipeline


class FakeIndexedRetriever:
    def __init__(self):
        self.queries: list[tuple[str, int]] = []

    def lexical_rarity(self, _query: str) -> float:
        return 7.2

    def hybrid_search(
        self,
        query: str,
        k: int,
        *,
        bm25_weight: float,
        dense_weight: float,
        field_boost,
        field_boost_depth: int,
    ) -> RetrievalResult:
        self.queries.append((query, k))
        adjustments = field_boost([0, 1])
        assert field_boost_depth == 120
        return RetrievalResult(
            rows=[0, 1],
            scores=[0.0318, 0.0312],
            component_ranks={"bm25": [1, 2], "dense": [3, 1]},
            timings_ms={
                "bm25_top_depth": 1.0,
                "query_embedding": 2.0,
                "faiss_top_depth": 3.0,
                "exact_rank_lookup_and_fusion": 0.5,
            },
            field_score_adjustments=adjustments,
        )


class FakeCrossEncoder:
    model_dir = "tests/fake-cross-encoder"

    def score_products(self, _query, products):
        assert [product.id for product in products] == ["with-lid", "without-lid"]
        return (
            {"with-lid": "E", "without-lid": "S"},
            {"with-lid": 2.8, "without-lid": 2.3},
            {},
        )


@pytest.fixture
def pipeline():
    catalog = ProductListCatalog(
        [
            Product(id="with-lid", title="Tervis Insulated Cup with Lid", brand="Tervis"),
            Product(id="without-lid", title="Tervis Insulated Cup", brand="Tervis"),
        ]
    )
    retriever = FakeIndexedRetriever()
    pipeline = SearchPipeline(
        catalog=catalog,
        catalog_ids=catalog.ids,
        retriever=retriever,
        reranker=FakeCrossEncoder(),
        index_version="test-index",
        model_version="cross-encoder:test",
    )
    return pipeline, retriever


@pytest.fixture
def client(pipeline):
    app.state.pipeline = pipeline[0]
    app.state.startup_error = None
    with TestClient(app) as test_client:
        yield test_client
    app.state.pipeline = None
    app.state.startup_error = None


def test_search_uses_indexed_hybrid_then_cross_encoder(client, pipeline):
    response = client.post("/search", json={"query": "Tervis cups without lid"})

    assert response.status_code == 200
    payload = response.json()
    assert pipeline[1].queries == [("tervis cups", 40)]
    assert payload["query_analysis"]["exclusions"] == ["lid"]
    assert payload["retrieval_decision"] == {
        "bm25_weight": 0.68,
        "dense_weight": 0.32,
        "lexical_rarity": 7.2,
        "reasons": ["recognized brand: tervis", "explicit exclusion"],
    }
    # Expected gain initially favours the lidded cup, but the deterministic
    # customer exclusion is applied after cross-encoder scoring.
    assert [row["product"]["id"] for row in payload["results"]] == ["without-lid", "with-lid"]
    assert payload["results"][1]["scores"]["negation_penalty"] == 0.75
    assert payload["results"][0]["scores"]["adaptive_bm25_weight"] == 0.68
    assert payload["results"][0]["scores"]["field_lexical_boost"] == 1.25
    assert payload["model_version"] == "cross-encoder:test"
    assert payload["index_version"] == "test-index"
    assert set(payload["stage_latency_ms"]) == {
        "query_understanding",
        "bm25_retrieval",
        "dense_retrieval",
        "fusion",
        "cross_encoder_reranking",
        "total_request",
    }


def test_search_contract_rejects_production_knobs(client):
    response = client.post("/search", json={"query": "tervis cup", "strategy": "bm25"})

    assert response.status_code == 422


def test_health_and_ready_report_loaded_pipeline(client):
    assert client.get("/health").json()["status"] == "ok"
    readiness = client.get("/ready")
    assert readiness.status_code == 200
    assert readiness.json()["catalog_products"] == 2
