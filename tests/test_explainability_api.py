"""Integration tests for the explainability API endpoints (US-014).

Covers:
  POST /explainability/lime
  POST /explainability/shap
  GET  /explainability/explanations
  GET  /explainability/explanations/{id}
  GET  /explainability/compare/{event_id}
"""

from __future__ import annotations

import pytest_asyncio


@pytest_asyncio.fixture
async def inference_event_id(client) -> str:
    """Create an inference event to use for explanation requests."""
    resp = await client.post(
        "/events",
        json={
            "event_type": "inference_request",
            "actor": "test_explainer",
            "payload": {"image_hash": "abc123", "model_id": "test-model"},
        },
    )
    assert resp.status_code == 201
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# POST /explainability/lime
# ---------------------------------------------------------------------------


class TestLimeEndpoint:
    async def test_lime_returns_201(self, client, inference_event_id):
        resp = await client.post(
            "/explainability/lime",
            json={"event_id": inference_event_id},
        )
        assert resp.status_code == 201

    async def test_lime_response_structure(self, client, inference_event_id):
        resp = await client.post(
            "/explainability/lime",
            json={"event_id": inference_event_id},
        )
        data = resp.json()
        assert data["method"] == "lime"
        assert data["inference_event_id"] == inference_event_id
        assert "explanation_id" in data
        assert "event_id" in data
        assert "timestamp" in data
        assert data["num_superpixels"] == 50
        assert data["num_samples"] == 100
        assert data["top_k"] == 5

    async def test_lime_custom_params(self, client, inference_event_id):
        resp = await client.post(
            "/explainability/lime",
            json={
                "event_id": inference_event_id,
                "num_superpixels": 25,
                "num_samples": 50,
                "top_k": 3,
            },
        )
        data = resp.json()
        assert data["num_superpixels"] == 25
        assert data["num_samples"] == 50
        assert data["top_k"] == 3

    async def test_lime_event_not_found(self, client):
        resp = await client.post(
            "/explainability/lime",
            json={"event_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404

    async def test_lime_creates_forensic_event(self, client, inference_event_id):
        resp = await client.post(
            "/explainability/lime",
            json={"event_id": inference_event_id},
        )
        data = resp.json()
        event_resp = await client.get(f"/events/{data['event_id']}")
        assert event_resp.status_code == 200
        event = event_resp.json()
        assert event["event_type"] == "explanation_generated"
        assert event["payload"]["method"] == "lime"


# ---------------------------------------------------------------------------
# POST /explainability/shap
# ---------------------------------------------------------------------------


class TestShapEndpoint:
    async def test_shap_returns_201(self, client, inference_event_id):
        resp = await client.post(
            "/explainability/shap",
            json={"event_id": inference_event_id},
        )
        assert resp.status_code == 201

    async def test_shap_response_structure(self, client, inference_event_id):
        resp = await client.post(
            "/explainability/shap",
            json={"event_id": inference_event_id},
        )
        data = resp.json()
        assert data["method"] == "shap"
        assert data["inference_event_id"] == inference_event_id
        assert "explanation_id" in data
        assert "event_id" in data
        assert data["num_samples"] == 128
        assert data["random_state"] == 42

    async def test_shap_custom_params(self, client, inference_event_id):
        resp = await client.post(
            "/explainability/shap",
            json={
                "event_id": inference_event_id,
                "num_samples": 64,
                "random_state": 99,
            },
        )
        data = resp.json()
        assert data["num_samples"] == 64
        assert data["random_state"] == 99

    async def test_shap_event_not_found(self, client):
        resp = await client.post(
            "/explainability/shap",
            json={"event_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404

    async def test_shap_creates_forensic_event(self, client, inference_event_id):
        resp = await client.post(
            "/explainability/shap",
            json={"event_id": inference_event_id},
        )
        data = resp.json()
        event_resp = await client.get(f"/events/{data['event_id']}")
        assert event_resp.status_code == 200
        event = event_resp.json()
        assert event["event_type"] == "explanation_generated"
        assert event["payload"]["method"] == "shap"


# ---------------------------------------------------------------------------
# GET /explainability/explanations
# ---------------------------------------------------------------------------


class TestListExplanations:
    async def test_empty_list(self, client):
        resp = await client.get("/explainability/explanations")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "explanations" in data

    async def test_list_after_creating(self, client, inference_event_id):
        await client.post(
            "/explainability/lime",
            json={"event_id": inference_event_id},
        )
        await client.post(
            "/explainability/shap",
            json={"event_id": inference_event_id},
        )
        resp = await client.get("/explainability/explanations")
        data = resp.json()
        assert data["total"] >= 2
        methods = {e["method"] for e in data["explanations"]}
        assert "lime" in methods
        assert "shap" in methods


# ---------------------------------------------------------------------------
# GET /explainability/explanations/{id}
# ---------------------------------------------------------------------------


class TestGetExplanation:
    async def test_get_by_id(self, client, inference_event_id):
        create_resp = await client.post(
            "/explainability/lime",
            json={"event_id": inference_event_id},
        )
        expl_id = create_resp.json()["explanation_id"]
        resp = await client.get(f"/explainability/explanations/{expl_id}")
        assert resp.status_code == 200
        assert resp.json()["explanation_id"] == expl_id

    async def test_not_found(self, client):
        resp = await client.get("/explainability/explanations/nonexistent-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /explainability/compare/{event_id}
# ---------------------------------------------------------------------------


class TestCompareExplanations:
    async def test_compare_both_methods(self, client, inference_event_id):
        await client.post(
            "/explainability/lime",
            json={"event_id": inference_event_id},
        )
        await client.post(
            "/explainability/shap",
            json={"event_id": inference_event_id},
        )
        resp = await client.get(f"/explainability/compare/{inference_event_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["inference_event_id"] == inference_event_id
        assert data["total_lime"] >= 1
        assert data["total_shap"] >= 1

    async def test_compare_no_explanations(self, client, inference_event_id):
        resp = await client.get(f"/explainability/compare/{inference_event_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_lime"] == 0
        assert data["total_shap"] == 0

    async def test_compare_event_not_found(self, client):
        resp = await client.get(
            "/explainability/compare/00000000-0000-0000-0000-000000000000"
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Forensic chain integrity
# ---------------------------------------------------------------------------


class TestExplainabilityChain:
    async def test_chain_valid_after_explanations(self, client, inference_event_id):
        await client.post(
            "/explainability/lime",
            json={"event_id": inference_event_id},
        )
        await client.post(
            "/explainability/shap",
            json={"event_id": inference_event_id},
        )
        resp = await client.get("/chain/integrity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
