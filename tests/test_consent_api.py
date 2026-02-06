"""Integration tests for the consent API endpoints (US-017)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.api.app import _db, _service, app
from friendlyface.api import app as app_module


@pytest_asyncio.fixture
async def client(tmp_path):
    """HTTP test client wired to a fresh database."""
    _db.db_path = tmp_path / "consent_api_test.db"
    await _db.connect()
    await _service.initialize()

    _service.merkle = __import__(
        "friendlyface.core.merkle", fromlist=["MerkleTree"]
    ).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()

    app_module._latest_compliance_report = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()


class TestConsentGrant:
    async def test_grant_returns_201(self, client):
        resp = await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        assert resp.status_code == 201

    async def test_grant_returns_record(self, client):
        resp = await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "actor": "test_agent",
        })
        data = resp.json()
        assert data["subject_id"] == "subj1"
        assert data["purpose"] == "recognition"
        assert data["granted"] is True
        assert data["id"] is not None
        assert data["timestamp"] is not None
        assert data["event_id"] is not None

    async def test_grant_with_expiry(self, client):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        resp = await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "expiry": future,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["expiry"] is not None

    async def test_grant_invalid_expiry(self, client):
        resp = await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "expiry": "not-a-date",
        })
        assert resp.status_code == 400
        assert "Invalid expiry" in resp.json()["detail"]

    async def test_grant_default_actor(self, client):
        resp = await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        assert resp.status_code == 201

    async def test_grant_multiple_purposes(self, client):
        r1 = await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        r2 = await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "training",
        })
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["id"] != r2.json()["id"]


class TestConsentRevoke:
    async def test_revoke_returns_200(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.post("/consent/revoke", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        assert resp.status_code == 200

    async def test_revoke_returns_record(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.post("/consent/revoke", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "reason": "user_request",
        })
        data = resp.json()
        assert data["subject_id"] == "subj1"
        assert data["granted"] is False
        assert data["revocation_reason"] == "user_request"

    async def test_revoke_with_reason(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.post("/consent/revoke", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "reason": "privacy_concern",
            "actor": "subject",
        })
        data = resp.json()
        assert data["revocation_reason"] == "privacy_concern"


class TestConsentStatus:
    async def test_status_no_record(self, client):
        resp = await client.get("/consent/status/unknown_subj")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_consent"] is False
        assert data["active"] is False
        assert data["record"] is None

    async def test_status_after_grant(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.get("/consent/status/subj1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["subject_id"] == "subj1"
        assert data["purpose"] == "recognition"
        assert data["has_consent"] is True
        assert data["granted"] is True
        assert data["active"] is True
        assert data["expired"] is False

    async def test_status_after_revoke(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        await client.post("/consent/revoke", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.get("/consent/status/subj1")
        data = resp.json()
        assert data["has_consent"] is True
        assert data["granted"] is False
        assert data["active"] is False

    async def test_status_with_purpose_query_param(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "training",
        })
        resp = await client.get("/consent/status/subj1")
        assert resp.json()["has_consent"] is False

        resp = await client.get("/consent/status/subj1?purpose=training")
        assert resp.json()["has_consent"] is True
        assert resp.json()["active"] is True

    async def test_status_expired_consent(self, client):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "expiry": past,
        })
        resp = await client.get("/consent/status/subj1")
        data = resp.json()
        assert data["has_consent"] is True
        assert data["granted"] is True
        assert data["expired"] is True
        assert data["active"] is False


class TestConsentHistory:
    async def test_history_empty(self, client):
        resp = await client.get("/consent/history/unknown_subj")
        assert resp.status_code == 200
        data = resp.json()
        assert data["subject_id"] == "unknown_subj"
        assert data["total"] == 0
        assert data["records"] == []

    async def test_history_preserves_all_records(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        await client.post("/consent/revoke", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })

        resp = await client.get("/consent/history/subj1?purpose=recognition")
        data = resp.json()
        assert data["total"] == 3
        assert data["records"][0]["granted"] is True
        assert data["records"][1]["granted"] is False
        assert data["records"][2]["granted"] is True

    async def test_history_multiple_purposes(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "training",
        })

        resp = await client.get("/consent/history/subj1")
        assert resp.json()["total"] == 2

        resp = await client.get("/consent/history/subj1?purpose=recognition")
        assert resp.json()["total"] == 1
        assert resp.json()["records"][0]["purpose"] == "recognition"

    async def test_history_records_contain_expected_fields(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.get("/consent/history/subj1")
        record = resp.json()["records"][0]
        assert "id" in record
        assert "subject_id" in record
        assert "purpose" in record
        assert "granted" in record
        assert "timestamp" in record
        assert "event_id" in record


class TestConsentCheck:
    async def test_check_no_consent_returns_deny(self, client):
        resp = await client.post("/consent/check", json={
            "subject_id": "unknown_subj",
            "purpose": "recognition",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["allowed"] is False
        assert data["has_consent"] is False
        assert data["subject_id"] == "unknown_subj"
        assert data["purpose"] == "recognition"

    async def test_check_granted_returns_allow(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.post("/consent/check", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        data = resp.json()
        assert data["allowed"] is True
        assert data["active"] is True

    async def test_check_revoked_returns_deny(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        await client.post("/consent/revoke", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.post("/consent/check", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        data = resp.json()
        assert data["allowed"] is False
        assert data["has_consent"] is True
        assert data["active"] is False

    async def test_check_expired_returns_deny(self, client):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "expiry": past,
        })
        resp = await client.post("/consent/check", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        data = resp.json()
        assert data["allowed"] is False
        assert data["has_consent"] is True

    async def test_check_different_purpose_returns_deny(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.post("/consent/check", json={
            "subject_id": "subj1",
            "purpose": "training",
        })
        data = resp.json()
        assert data["allowed"] is False

    async def test_check_regranted_returns_allow(self, client):
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        await client.post("/consent/revoke", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        resp = await client.post("/consent/check", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        data = resp.json()
        assert data["allowed"] is True
        assert data["active"] is True


class TestConsentLifecycle:
    async def test_full_lifecycle(self, client):
        """Grant -> check -> revoke -> check -> regrant -> check."""
        resp = await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "actor": "onboarding",
        })
        assert resp.status_code == 201

        resp = await client.post("/consent/check", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        assert resp.json()["allowed"] is True

        resp = await client.post("/consent/revoke", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "reason": "user_request",
            "actor": "subject",
        })
        assert resp.status_code == 200

        resp = await client.post("/consent/check", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        assert resp.json()["allowed"] is False

        resp = await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
            "actor": "subject",
        })
        assert resp.status_code == 201

        resp = await client.post("/consent/check", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        assert resp.json()["allowed"] is True

        resp = await client.get("/consent/history/subj1?purpose=recognition")
        assert resp.json()["total"] == 3

    async def test_consent_forensic_events_logged(self, client):
        """All consent operations should produce forensic events."""
        await client.post("/consent/grant", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })
        await client.post("/consent/revoke", json={
            "subject_id": "subj1",
            "purpose": "recognition",
        })

        resp = await client.get("/events")
        events = resp.json()
        consent_events = [
            e for e in events if e["event_type"] == "consent_update"
        ]
        assert len(consent_events) == 2

    async def test_consent_and_compliance_integration(self, client):
        """Consent data should feed into compliance reports."""
        for i in range(5):
            await client.post("/consent/grant", json={
                "subject_id": f"subj_{i}",
                "purpose": "recognition",
            })

        resp = await client.post("/governance/compliance/generate")
        assert resp.status_code == 201
        data = resp.json()
        assert data["metrics"]["consent_coverage_pct"] == 100.0


class TestGovernanceEndpointsExist:
    async def test_get_governance_compliance(self, client):
        resp = await client.get("/governance/compliance")
        assert resp.status_code == 200
        assert "report_id" in resp.json()

    async def test_post_governance_compliance_generate(self, client):
        resp = await client.post("/governance/compliance/generate")
        assert resp.status_code == 201
        assert "report_id" in resp.json()
