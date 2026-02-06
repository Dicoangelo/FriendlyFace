"""Tests for API key authentication middleware."""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.api.app import _db, _service, app


@pytest_asyncio.fixture
async def auth_client(tmp_path):
    """HTTP test client wired to a fresh database (same pattern as conftest)."""
    _db.db_path = tmp_path / "auth_test.db"
    await _db.connect()
    await _service.initialize()

    _service.merkle = __import__("friendlyface.core.merkle", fromlist=["MerkleTree"]).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()


# ---------------------------------------------------------------------------
# Dev mode: FF_API_KEYS not set -> auth disabled, all requests pass
# ---------------------------------------------------------------------------


class TestDevMode:
    """When FF_API_KEYS is unset or empty, auth is disabled."""

    async def test_no_env_var_allows_all_requests(self, auth_client, monkeypatch):
        """Without FF_API_KEYS, protected endpoints are accessible."""
        monkeypatch.delenv("FF_API_KEYS", raising=False)
        resp = await auth_client.get("/events")
        assert resp.status_code == 200

    async def test_empty_env_var_allows_all_requests(self, auth_client, monkeypatch):
        """An empty FF_API_KEYS string disables auth."""
        monkeypatch.setenv("FF_API_KEYS", "")
        resp = await auth_client.get("/events")
        assert resp.status_code == 200

    async def test_whitespace_only_env_var_allows_all_requests(self, auth_client, monkeypatch):
        """Whitespace-only FF_API_KEYS disables auth."""
        monkeypatch.setenv("FF_API_KEYS", "   ")
        resp = await auth_client.get("/events")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth enabled: missing key -> 403
# ---------------------------------------------------------------------------


class TestAuthEnabledNoKey:
    """When FF_API_KEYS is set, requests without a key are rejected."""

    async def test_missing_key_returns_403(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "test-key-1")
        resp = await auth_client.get("/events")
        assert resp.status_code == 403
        assert "API key required" in resp.json()["detail"]

    async def test_missing_key_post_returns_403(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "test-key-1")
        resp = await auth_client.post(
            "/events",
            json={"event_type": "training_start", "actor": "test"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Auth enabled: invalid key -> 401
# ---------------------------------------------------------------------------


class TestAuthEnabledInvalidKey:
    """When FF_API_KEYS is set, requests with a bad key get 401."""

    async def test_invalid_key_header_returns_401(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "correct-key")
        resp = await auth_client.get(
            "/events",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["detail"]

    async def test_invalid_key_query_returns_401(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "correct-key")
        resp = await auth_client.get("/events", params={"api_key": "wrong-key"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Auth enabled: valid key -> success
# ---------------------------------------------------------------------------


class TestAuthEnabledValidKey:
    """When FF_API_KEYS is set, requests with a valid key succeed."""

    async def test_valid_key_via_header(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "my-secret-key")
        resp = await auth_client.get(
            "/events",
            headers={"X-API-Key": "my-secret-key"},
        )
        assert resp.status_code == 200

    async def test_valid_key_via_query_param(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "my-secret-key")
        resp = await auth_client.get("/events", params={"api_key": "my-secret-key"})
        assert resp.status_code == 200

    async def test_multiple_keys_any_valid(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "key-alpha, key-beta, key-gamma")
        for key in ("key-alpha", "key-beta", "key-gamma"):
            resp = await auth_client.get(
                "/events",
                headers={"X-API-Key": key},
            )
            assert resp.status_code == 200, f"Key {key!r} should be accepted"

    async def test_post_with_valid_key(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "post-key")
        resp = await auth_client.post(
            "/events",
            json={"event_type": "training_start", "actor": "test"},
            headers={"X-API-Key": "post-key"},
        )
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Health endpoint: always public
# ---------------------------------------------------------------------------


class TestHealthAlwaysPublic:
    """GET /health must be accessible regardless of auth configuration."""

    async def test_health_without_auth(self, auth_client, monkeypatch):
        monkeypatch.delenv("FF_API_KEYS", raising=False)
        resp = await auth_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_health_with_auth_enabled_no_key(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "secret")
        resp = await auth_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_health_with_auth_enabled_invalid_key(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "secret")
        resp = await auth_client.get(
            "/health",
            headers={"X-API-Key": "bad-key"},
        )
        assert resp.status_code == 200

    async def test_health_with_auth_enabled_valid_key(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "secret")
        resp = await auth_client.get(
            "/health",
            headers={"X-API-Key": "secret"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Header vs query param precedence
# ---------------------------------------------------------------------------


class TestKeyPrecedence:
    """X-API-Key header takes priority over api_key query parameter."""

    async def test_header_used_over_query_param(self, auth_client, monkeypatch):
        """When both are provided, the header key is used."""
        monkeypatch.setenv("FF_API_KEYS", "header-key")
        resp = await auth_client.get(
            "/events",
            headers={"X-API-Key": "header-key"},
            params={"api_key": "wrong-query-key"},
        )
        assert resp.status_code == 200
