"""Tests for rate limiting (US-067).

Covers:
  - Per-endpoint rate limits (/did/create)
  - 429 response with Retry-After header
  - Disabling rate limits
  - Rate limit error response structure
"""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import friendlyface.api.app as app_module
from friendlyface.api.app import _db, _service, app, limiter


@pytest_asyncio.fixture
async def rl_client(tmp_path):
    """HTTP test client with rate limiting ENABLED."""
    _db.db_path = tmp_path / "rl_test.db"
    await _db.connect()
    await _db.run_migrations()
    await _service.initialize()

    _service.merkle = __import__("friendlyface.core.merkle", fromlist=["MerkleTree"]).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()
    app_module._explanations.clear()
    app_module._model_registry.clear()
    app_module._fl_simulations.clear()
    app_module._latest_compliance_report = None

    # Enable limiter and reset storage
    limiter.enabled = True
    limiter._limiter.storage.reset()

    # Lower the /did/create limit from 10/min to 2/min for fast testing
    did_limits = limiter._route_limits.get("friendlyface.api.app.create_did", [])
    original_amounts = {}
    for lim in did_limits:
        original_amounts[id(lim)] = lim.limit.amount
        lim.limit.amount = 2

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    for lim in did_limits:
        lim.limit.amount = original_amounts[id(lim)]
    limiter.enabled = False
    await _db.close()


class TestRateLimitEnforcement:
    """Rate limiter returns 429 when per-endpoint limits are exceeded."""

    async def test_rate_limit_returns_429(self, rl_client):
        """Hitting the per-endpoint rate limit returns 429."""
        for _ in range(2):
            resp = await rl_client.post("/did/create", json={"label": f"test-{_}"})
            assert resp.status_code == 201

        resp = await rl_client.post("/did/create", json={"label": "over-limit"})
        assert resp.status_code == 429

    async def test_429_has_retry_after_header(self, rl_client):
        """429 response includes Retry-After header."""
        for _ in range(2):
            await rl_client.post("/did/create", json={"label": f"test-{_}"})

        resp = await rl_client.post("/did/create", json={"label": "over-limit"})
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    async def test_429_error_structure(self, rl_client):
        """429 response has correct JSON error structure."""
        for _ in range(2):
            await rl_client.post("/did/create", json={"label": f"test-{_}"})

        resp = await rl_client.post("/did/create", json={"label": "over-limit"})
        assert resp.status_code == 429
        data = resp.json()
        assert data["error"] == "rate_limit_exceeded"
        assert "message" in data
        assert "request_id" in data


class TestRateLimitDisabled:
    """Rate limiter can be disabled."""

    async def test_disabled_limiter_allows_unlimited(self, tmp_path):
        """When limiter is disabled, no requests are rate-limited."""
        _db.db_path = tmp_path / "rl_disabled_test.db"
        await _db.connect()
        await _service.initialize()

        _service.merkle = __import__(
            "friendlyface.core.merkle", fromlist=["MerkleTree"]
        ).MerkleTree()
        _service._event_index = {}

        limiter.enabled = False

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            for _ in range(10):
                resp = await ac.get("/health")
                assert resp.status_code == 200

        await _db.close()
