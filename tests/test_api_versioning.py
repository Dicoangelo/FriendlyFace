"""Tests for US-050 — API versioning with /api/v1/ prefix."""

from __future__ import annotations


class TestVersionEndpoint:
    """GET /api/version returns version info."""

    async def test_version_endpoint(self, client):
        resp = await client.get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1.0.0"
        assert data["api_prefix"] == "/api/v1"

    async def test_version_has_all_fields(self, client):
        resp = await client.get("/api/version")
        data = resp.json()
        assert set(data.keys()) == {"version", "api_prefix"}


class TestVersionedHealth:
    """Health check available at both /health and /api/v1/health."""

    async def test_v1_health_returns_200(self, client):
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    async def test_v1_health_matches_original(self, client):
        original = await client.get("/health")
        versioned = await client.get("/api/v1/health")
        assert original.status_code == versioned.status_code
        orig_data = original.json()
        vers_data = versioned.json()
        # Both should have the same structure and key values
        assert orig_data["status"] == vers_data["status"]
        assert orig_data["version"] == vers_data["version"]
        assert orig_data["storage_backend"] == vers_data["storage_backend"]


class TestVersionedEvents:
    """/api/v1/events works for both GET and POST."""

    async def test_v1_list_events(self, client):
        resp = await client.get("/api/v1/events")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_v1_record_event(self, client):
        resp = await client.post(
            "/api/v1/events",
            json={
                "event_type": "training_start",
                "actor": "versioning_test",
                "payload": {"note": "v1 test"},
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["actor"] == "versioning_test"


class TestVersionedDashboard:
    """/api/v1/dashboard works."""

    async def test_v1_dashboard(self, client):
        resp = await client.get("/api/v1/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_events" in data
        assert "uptime_seconds" in data


class TestOriginalRoutesStillWork:
    """Original unversioned routes must remain functional."""

    async def test_original_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_original_events_get(self, client):
        resp = await client.get("/events")
        assert resp.status_code == 200

    async def test_original_events_post(self, client):
        resp = await client.post(
            "/events",
            json={
                "event_type": "training_start",
                "actor": "compat_test",
                "payload": {},
            },
        )
        assert resp.status_code == 201

    async def test_original_dashboard(self, client):
        resp = await client.get("/dashboard")
        assert resp.status_code == 200

    async def test_original_chain_integrity(self, client):
        resp = await client.get("/chain/integrity")
        assert resp.status_code == 200

    async def test_original_merkle_root(self, client):
        resp = await client.get("/merkle/root")
        assert resp.status_code == 200
