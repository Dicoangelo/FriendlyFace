"""Tests for US-076: Auth provider factory integration + US-080: RBAC.

Verifies that require_api_key delegates to the configured auth provider
and that Bearer token, X-API-Key, and query param all work. Also tests
the require_role dependency factory.
"""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.api.app import _db, _service, app, limiter


@pytest_asyncio.fixture
async def auth_client(tmp_path):
    """HTTP test client wired to a fresh database."""
    _db.db_path = tmp_path / "auth_int_test.db"
    await _db.connect()
    await _service.initialize()

    _service.merkle = __import__("friendlyface.core.merkle", fromlist=["MerkleTree"]).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()

    limiter.enabled = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()


# ---------------------------------------------------------------------------
# Bearer token support (US-076)
# ---------------------------------------------------------------------------


class TestBearerToken:
    """Authorization: Bearer <token> header support."""

    async def test_bearer_token_accepted(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "my-bearer-key")
        resp = await auth_client.get(
            "/events",
            headers={"Authorization": "Bearer my-bearer-key"},
        )
        assert resp.status_code == 200

    async def test_bearer_token_invalid(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "correct-key")
        resp = await auth_client.get(
            "/events",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    async def test_bearer_takes_precedence_over_x_api_key(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "bearer-key")
        resp = await auth_client.get(
            "/events",
            headers={
                "Authorization": "Bearer bearer-key",
                "X-API-Key": "wrong",
            },
        )
        assert resp.status_code == 200

    async def test_bearer_case_insensitive_prefix(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "my-key")
        resp = await auth_client.get(
            "/events",
            headers={"Authorization": "bearer my-key"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Provider switching (US-076)
# ---------------------------------------------------------------------------


class TestProviderSwitching:
    """FF_AUTH_PROVIDER selects the auth backend."""

    async def test_default_provider_is_api_key(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "test-key")
        monkeypatch.delenv("FF_AUTH_PROVIDER", raising=False)
        resp = await auth_client.get(
            "/events",
            headers={"X-API-Key": "test-key"},
        )
        assert resp.status_code == 200

    async def test_supabase_provider_jwt(self, auth_client, monkeypatch):
        import jwt

        secret = "supabase-test-secret-key-1234"
        monkeypatch.setenv("FF_AUTH_PROVIDER", "supabase")
        monkeypatch.setenv("FF_SUPABASE_JWT_SECRET", secret)
        monkeypatch.setenv("FF_API_KEYS", "")

        token = jwt.encode(
            {"sub": "user-42", "role": "authenticated", "aud": "authenticated"},
            secret,
            algorithm="HS256",
        )
        resp = await auth_client.get(
            "/events",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    async def test_supabase_provider_bad_token(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_AUTH_PROVIDER", "supabase")
        monkeypatch.setenv("FF_SUPABASE_JWT_SECRET", "secret")
        monkeypatch.setenv("FF_API_KEYS", "")

        resp = await auth_client.get(
            "/events",
            headers={"Authorization": "Bearer garbage-token"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Dev mode preserved (US-076)
# ---------------------------------------------------------------------------


class TestDevModePreserved:
    """Dev mode (empty FF_API_KEYS, api_key provider) still works."""

    async def test_dev_mode_no_keys(self, auth_client, monkeypatch):
        monkeypatch.delenv("FF_API_KEYS", raising=False)
        monkeypatch.delenv("FF_AUTH_PROVIDER", raising=False)
        resp = await auth_client.get("/events")
        assert resp.status_code == 200

    async def test_dev_mode_empty_keys(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "")
        resp = await auth_client.get("/events")
        assert resp.status_code == 200

    async def test_dev_mode_whitespace_keys(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "   ")
        resp = await auth_client.get("/events")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# AuthResult on request.state (US-076)
# ---------------------------------------------------------------------------


class TestAuthResultAttached:
    """AuthResult is accessible via request.state.auth."""

    async def test_health_no_auth_state(self, auth_client, monkeypatch):
        """Public paths don't set auth state but should still work."""
        monkeypatch.setenv("FF_API_KEYS", "key")
        resp = await auth_client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# RBAC — require_role (US-080)
# ---------------------------------------------------------------------------


class TestRBAC:
    """Role-based access control via require_role dependency."""

    async def test_admin_role_backup_create(self, auth_client, monkeypatch):
        """Admin role can create backups."""
        monkeypatch.setenv("FF_API_KEYS", "admin-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"admin-key": ["admin"]}')
        resp = await auth_client.post(
            "/admin/backup",
            headers={"X-API-Key": "admin-key"},
        )
        assert resp.status_code == 201

    async def test_viewer_role_cannot_create_backup(self, auth_client, monkeypatch):
        """Viewer role is blocked from admin endpoints."""
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        resp = await auth_client.post(
            "/admin/backup",
            headers={"X-API-Key": "viewer-key"},
        )
        assert resp.status_code == 403
        assert "Requires one of roles" in resp.json()["detail"]

    async def test_viewer_role_cannot_restore_backup(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        resp = await auth_client.post(
            "/admin/backup/restore?filename=test.db",
            headers={"X-API-Key": "viewer-key"},
        )
        assert resp.status_code == 403

    async def test_viewer_can_list_backups(self, auth_client, monkeypatch):
        """Read-only admin endpoints don't require admin role."""
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        resp = await auth_client.get(
            "/admin/backups",
            headers={"X-API-Key": "viewer-key"},
        )
        assert resp.status_code == 200

    async def test_dev_mode_bypasses_role_checks(self, auth_client, monkeypatch):
        """Dev mode (no keys) bypasses all role checks."""
        monkeypatch.delenv("FF_API_KEYS", raising=False)
        resp = await auth_client.post("/admin/backup")
        assert resp.status_code == 201

    async def test_default_role_is_admin(self, auth_client, monkeypatch):
        """Keys without explicit role mapping default to admin."""
        monkeypatch.setenv("FF_API_KEYS", "default-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", "")
        resp = await auth_client.post(
            "/admin/backup",
            headers={"X-API-Key": "default-key"},
        )
        assert resp.status_code == 201

    async def test_viewer_cannot_rollback_migration(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        resp = await auth_client.post(
            "/admin/migrations/rollback",
            headers={"X-API-Key": "viewer-key"},
        )
        assert resp.status_code == 403

    async def test_viewer_cannot_generate_compliance(self, auth_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        resp = await auth_client.post(
            "/governance/compliance/generate",
            headers={"X-API-Key": "viewer-key"},
        )
        assert resp.status_code == 403
