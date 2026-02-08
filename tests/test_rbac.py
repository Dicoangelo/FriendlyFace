"""Tests for Role-Based Access Control (US-038).

Covers the Role enum, role hierarchy, require_role dependency,
and endpoint-level role enforcement.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.api.app import _db, _service, app, limiter
from friendlyface.rbac import ROLE_HIERARCHY, ROLE_INCLUDES, Role, has_role


# ---------------------------------------------------------------------------
# Unit tests — Role enum and hierarchy
# ---------------------------------------------------------------------------


class TestRoleEnum:
    """Role enum values and membership."""

    def test_all_roles_defined(self):
        assert set(Role) == {"admin", "analyst", "auditor", "subject", "viewer"}

    def test_role_is_strenum(self):
        assert str(Role.ADMIN) == "admin"
        assert f"role={Role.VIEWER}" == "role=viewer"

    def test_hierarchy_order(self):
        assert ROLE_HIERARCHY[0] == Role.ADMIN
        assert ROLE_HIERARCHY[-1] == Role.VIEWER

    def test_admin_includes_all(self):
        for role in Role:
            assert role in ROLE_INCLUDES[Role.ADMIN]

    def test_viewer_includes_only_self(self):
        assert ROLE_INCLUDES[Role.VIEWER] == frozenset({Role.VIEWER})

    def test_analyst_includes_auditor_and_viewer(self):
        includes = ROLE_INCLUDES[Role.ANALYST]
        assert Role.ANALYST in includes
        assert Role.AUDITOR in includes
        assert Role.VIEWER in includes
        assert Role.ADMIN not in includes

    def test_auditor_does_not_include_analyst(self):
        assert Role.ANALYST not in ROLE_INCLUDES[Role.AUDITOR]

    def test_subject_includes_viewer(self):
        assert Role.VIEWER in ROLE_INCLUDES[Role.SUBJECT]
        assert Role.ADMIN not in ROLE_INCLUDES[Role.SUBJECT]


# ---------------------------------------------------------------------------
# Unit tests — has_role()
# ---------------------------------------------------------------------------


class TestHasRole:
    """has_role() resolves hierarchy correctly."""

    def test_direct_match(self):
        assert has_role(["viewer"], "viewer") is True

    def test_admin_satisfies_any(self):
        for role in Role:
            assert has_role(["admin"], role.value) is True

    def test_viewer_cannot_satisfy_admin(self):
        assert has_role(["viewer"], "admin") is False

    def test_analyst_satisfies_auditor(self):
        assert has_role(["analyst"], "auditor") is True

    def test_auditor_cannot_satisfy_analyst(self):
        assert has_role(["auditor"], "analyst") is False

    def test_unknown_role_ignored(self):
        assert has_role(["unknown_role"], "admin") is False

    def test_mixed_roles(self):
        assert has_role(["viewer", "analyst"], "auditor") is True

    def test_empty_roles(self):
        assert has_role([], "viewer") is False

    def test_subject_cannot_satisfy_admin(self):
        assert has_role(["subject"], "admin") is False


# ---------------------------------------------------------------------------
# Integration tests — endpoint enforcement
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def rbac_client(tmp_path):
    """HTTP test client for RBAC endpoint tests."""
    _db.db_path = tmp_path / "rbac_test.db"
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


class TestEndpointRoles:
    """Verify role enforcement on protected endpoints."""

    # -- Admin-only endpoints --

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/admin/backup"),
            ("POST", "/admin/backup/restore?filename=x.db"),
            ("POST", "/admin/backup/cleanup"),
            ("POST", "/admin/backup/verify?filename=x.db"),
            ("POST", "/admin/migrations/rollback"),
            ("POST", "/governance/compliance/generate"),
            ("POST", "/erasure/erase/subject-1"),
        ],
    )
    async def test_viewer_blocked_from_admin_endpoints(
        self, rbac_client, monkeypatch, method, path
    ):
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        resp = await rbac_client.request(method, path, headers={"X-API-Key": "viewer-key"})
        assert resp.status_code == 403

    @pytest.mark.parametrize(
        "method,path",
        [
            ("POST", "/admin/backup"),
            ("POST", "/governance/compliance/generate"),
        ],
    )
    async def test_admin_can_access_admin_endpoints(self, rbac_client, monkeypatch, method, path):
        monkeypatch.setenv("FF_API_KEYS", "admin-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"admin-key": ["admin"]}')
        resp = await rbac_client.request(method, path, headers={"X-API-Key": "admin-key"})
        # Should not get 403 — may get 4xx for bad input but not forbidden
        assert resp.status_code != 403

    # -- Analyst endpoints --

    @pytest.mark.parametrize(
        "path",
        [
            "/fairness/audit",
        ],
    )
    async def test_viewer_blocked_from_analyst_endpoints(self, rbac_client, monkeypatch, path):
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        resp = await rbac_client.post(
            path,
            headers={"X-API-Key": "viewer-key"},
            json={},
        )
        assert resp.status_code == 403

    async def test_analyst_can_trigger_audit(self, rbac_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "analyst-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"analyst-key": ["analyst"]}')
        resp = await rbac_client.post(
            "/fairness/audit",
            headers={"X-API-Key": "analyst-key"},
            json={
                "group_results": {
                    "group_a": {"tp": 80, "fp": 10, "tn": 90, "fn": 20},
                    "group_b": {"tp": 70, "fp": 20, "tn": 80, "fn": 30},
                },
            },
        )
        assert resp.status_code != 403

    # -- Auditor endpoints --

    async def test_auditor_can_export_compliance(self, rbac_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "auditor-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"auditor-key": ["auditor"]}')
        resp = await rbac_client.get(
            "/governance/compliance/export",
            headers={"X-API-Key": "auditor-key"},
        )
        assert resp.status_code != 403

    async def test_viewer_blocked_from_compliance_export(self, rbac_client, monkeypatch):
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        resp = await rbac_client.get(
            "/governance/compliance/export",
            headers={"X-API-Key": "viewer-key"},
        )
        assert resp.status_code == 403

    # -- Role hierarchy --

    async def test_admin_inherits_analyst(self, rbac_client, monkeypatch):
        """Admin can access analyst-only endpoints via hierarchy."""
        monkeypatch.setenv("FF_API_KEYS", "admin-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"admin-key": ["admin"]}')
        resp = await rbac_client.post(
            "/fairness/audit",
            headers={"X-API-Key": "admin-key"},
            json={
                "group_results": {
                    "group_a": {"tp": 80, "fp": 10, "tn": 90, "fn": 20},
                    "group_b": {"tp": 70, "fp": 20, "tn": 80, "fn": 30},
                },
            },
        )
        assert resp.status_code != 403

    async def test_analyst_inherits_auditor(self, rbac_client, monkeypatch):
        """Analyst can access auditor endpoints via hierarchy."""
        monkeypatch.setenv("FF_API_KEYS", "analyst-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"analyst-key": ["analyst"]}')
        resp = await rbac_client.get(
            "/governance/compliance/export",
            headers={"X-API-Key": "analyst-key"},
        )
        assert resp.status_code != 403


class TestRBACToggle:
    """FF_RBAC_ENABLED flag controls enforcement."""

    async def test_rbac_disabled_allows_all(self, rbac_client, monkeypatch):
        """When RBAC is disabled, role checks are skipped."""
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        monkeypatch.setenv("FF_RBAC_ENABLED", "false")
        resp = await rbac_client.post(
            "/admin/backup",
            headers={"X-API-Key": "viewer-key"},
        )
        assert resp.status_code != 403

    async def test_rbac_enabled_enforces_roles(self, rbac_client, monkeypatch):
        """When RBAC is enabled (default), role checks apply."""
        monkeypatch.setenv("FF_API_KEYS", "viewer-key")
        monkeypatch.setenv("FF_API_KEY_ROLES", '{"viewer-key": ["viewer"]}')
        monkeypatch.setenv("FF_RBAC_ENABLED", "true")
        resp = await rbac_client.post(
            "/admin/backup",
            headers={"X-API-Key": "viewer-key"},
        )
        assert resp.status_code == 403


class TestDevModeRBAC:
    """Dev mode (no API keys) bypasses all RBAC."""

    async def test_dev_mode_bypasses_all(self, rbac_client, monkeypatch):
        monkeypatch.delenv("FF_API_KEYS", raising=False)
        resp = await rbac_client.post("/admin/backup")
        assert resp.status_code == 201

    async def test_dev_mode_bypasses_analyst_check(self, rbac_client, monkeypatch):
        monkeypatch.delenv("FF_API_KEYS", raising=False)
        resp = await rbac_client.post(
            "/fairness/audit",
            json={
                "group_results": {
                    "group_a": {"tp": 80, "fp": 10, "tn": 90, "fn": 20},
                    "group_b": {"tp": 70, "fp": 20, "tn": 80, "fn": 30},
                },
            },
        )
        assert resp.status_code != 403
