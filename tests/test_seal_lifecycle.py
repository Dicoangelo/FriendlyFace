"""Tests for ForensicSeal lifecycle — expiry & renewal (US-003) and revocation (US-004).

US-003 (Seal Expiry & Continuous Compliance):
  - get_seal_status returns days until expiry
  - get_seal_status reports active seal correctly
  - Expired seal is auto-marked on status check
  - Seal renewal creates a new seal
  - Renewal links to previous seal
  - Renewal of expired (non-revoked) seal still works
  - Renewed seal uses default expiry days
  - API: GET /seal/status/{id}, POST /seal/renew/{id}, v1 route

US-004 (Seal Revocation):
  - Revocation marks seal as revoked
  - Revocation is irreversible (re-revoke fails)
  - Revocation creates forensic event
  - Revoked seal fails verification
  - API: POST /seal/revoke/{id}, requires reason, v1 route
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import friendlyface.api.app as app_module
from friendlyface.api.app import _db, _service, app, limiter
from friendlyface.core.models import EventType
from friendlyface.core.service import ForensicService
from friendlyface.governance.compliance import ComplianceReporter
from friendlyface.recognition.gallery import FaceGallery
from friendlyface.seal.service import ForensicSealService
from friendlyface.storage.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    """Fresh in-memory database for each test."""
    database = Database(tmp_path / "test_seal_lifecycle.db")
    await database.connect()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def service(db):
    """Fresh forensic service for each test."""
    svc = ForensicService(db)
    await svc.initialize()
    return svc


@pytest_asyncio.fixture
async def seal_service(db, service):
    """Fresh seal service for each test."""
    reporter = ComplianceReporter(db, service)
    return ForensicSealService(db, service, reporter)


@pytest_asyncio.fixture
async def issued_seal(seal_service):
    """Issue a test seal and return its data."""
    return await seal_service.issue_seal(
        system_id="lifecycle-sys",
        system_name="LifecycleTestSystem",
        threshold=0.0,
        expiry_days=90,
    )


@pytest_asyncio.fixture
async def client(tmp_path):
    """HTTP test client wired to a fresh database."""
    _db.db_path = tmp_path / "seal_lifecycle_api_test.db"
    await _db.connect()
    await _db.run_migrations()
    await _service.initialize()

    # Reset in-memory state
    _service.merkle = __import__(
        "friendlyface.core.merkle", fromlist=["MerkleTree"]
    ).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()
    app_module._dashboard_cache["data"] = None
    app_module._dashboard_cache["timestamp"] = 0.0
    app_module._explanations.clear()
    app_module._model_registry.clear()
    app_module._fl_simulations.clear()
    app_module._latest_compliance_report = None
    app_module._auto_audit_interval = 50
    app_module._recognition_event_count = 0
    app_module._gallery = FaceGallery(_db)
    from friendlyface.recognition.pipeline import RecognitionPipeline

    app_module._recognition_pipeline = RecognitionPipeline(gallery=app_module._gallery)

    limiter.enabled = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _issue_seal_via_api(client: AsyncClient) -> dict:
    """Issue a seal through the API and return its data."""
    resp = await client.post(
        "/seal/issue",
        json={
            "system_id": "api-lifecycle-sys",
            "system_name": "API Lifecycle System",
            "threshold": 0.0,
        },
    )
    assert resp.status_code == 201
    return resp.json()


# ===========================================================================
# US-003 — Seal Expiry & Continuous Compliance
# ===========================================================================


@pytest.mark.asyncio
async def test_get_seal_status_returns_days_until_expiry(seal_service, issued_seal):
    """get_seal_status returns days_remaining reflecting the 90-day expiry."""
    result = await seal_service.get_seal_status(issued_seal["id"])

    assert "days_remaining" in result
    assert result["days_remaining"] is not None
    # 90-day seal should have ~89-90 days remaining
    assert 88 <= result["days_remaining"] <= 90


@pytest.mark.asyncio
async def test_get_seal_status_active_seal(seal_service, issued_seal):
    """get_seal_status reports active status with full metadata for a fresh seal."""
    result = await seal_service.get_seal_status(issued_seal["id"])

    assert result["status"] == "active"
    assert result["seal_id"] == issued_seal["id"]
    assert result["issued_at"] == issued_seal["issued_at"]
    assert result["expires_at"] == issued_seal["expires_at"]
    assert result["compliance_score"] == issued_seal["compliance_score"]
    assert result["system_id"] == "lifecycle-sys"
    assert result["system_name"] == "LifecycleTestSystem"


@pytest.mark.asyncio
async def test_get_seal_status_expired_seal_auto_marked(db, service):
    """get_seal_status auto-marks an expired seal from active to expired in DB."""
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    seal = await seal_svc.issue_seal(
        system_id="expire-auto",
        system_name="ExpireAutoTest",
        threshold=0.0,
        expiry_days=1,
    )

    # Set expires_at to the past in the DB
    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    await db.db.execute(
        "UPDATE seals SET expires_at = ? WHERE id = ?",
        (past, seal["id"]),
    )
    await db.db.commit()

    result = await seal_svc.get_seal_status(seal["id"])

    assert result["status"] == "expired"
    assert result["days_remaining"] == 0

    # Verify the DB was also updated
    db_seal = await db.get_seal(seal["id"])
    assert db_seal["status"] == "expired"


@pytest.mark.asyncio
async def test_renew_seal_creates_new_seal(seal_service, issued_seal):
    """renew_seal issues a new seal distinct from the original."""
    new_seal = await seal_service.renew_seal(issued_seal["id"])

    assert new_seal["id"] != issued_seal["id"]
    assert new_seal["status"] == "active"
    assert new_seal["system_id"] == issued_seal["system_id"]
    assert new_seal["system_name"] == issued_seal["system_name"]
    assert new_seal["compliance_score"] >= 0


@pytest.mark.asyncio
async def test_renew_seal_links_to_previous(seal_service, issued_seal):
    """Renewed seal's previous_seal_id points to the original seal."""
    new_seal = await seal_service.renew_seal(issued_seal["id"])

    assert new_seal["previous_seal_id"] == issued_seal["id"]

    # Verify the link is persisted in DB
    db_seal = await seal_service.db.get_seal(new_seal["id"])
    assert db_seal["previous_seal_id"] == issued_seal["id"]


@pytest.mark.asyncio
async def test_renew_seal_expired_seal_still_works(db, service):
    """An expired (non-revoked) seal can still be renewed."""
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    seal = await seal_svc.issue_seal(
        system_id="renew-expired",
        system_name="RenewExpiredTest",
        threshold=0.0,
        expiry_days=1,
    )

    # Mark as expired
    await db.update_seal_status(seal["id"], "expired")

    new_seal = await seal_svc.renew_seal(seal["id"])

    assert new_seal["id"] != seal["id"]
    assert new_seal["status"] == "active"
    assert new_seal["previous_seal_id"] == seal["id"]


@pytest.mark.asyncio
async def test_renew_custom_expiry_days(db, service):
    """Renewed seal uses the default 90-day expiry via issue_seal."""
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    # Issue original with 30-day expiry
    seal = await seal_svc.issue_seal(
        system_id="renew-custom",
        system_name="RenewCustomTest",
        threshold=0.0,
        expiry_days=30,
    )

    new_seal = await seal_svc.renew_seal(seal["id"])

    # The renewed seal uses the default expiry (90 days) since renew_seal
    # calls issue_seal without passing expiry_days
    issued = datetime.fromisoformat(new_seal["issued_at"])
    expires = datetime.fromisoformat(new_seal["expires_at"])
    delta_days = (expires - issued).days
    assert 89 <= delta_days <= 91


# ---------------------------------------------------------------------------
# US-003 — API Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_get_seal_status(client):
    """GET /seal/status/{seal_id} returns seal status with days_remaining."""
    seal_data = await _issue_seal_via_api(client)

    resp = await client.get(f"/seal/status/{seal_data['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["seal_id"] == seal_data["id"]
    assert data["status"] == "active"
    assert "days_remaining" in data
    assert data["days_remaining"] is not None
    assert data["days_remaining"] > 0
    assert data["system_id"] == "api-lifecycle-sys"


@pytest.mark.asyncio
async def test_api_renew_seal(client):
    """POST /seal/renew/{seal_id} returns 201 with new seal data."""
    seal_data = await _issue_seal_via_api(client)

    resp = await client.post(f"/seal/renew/{seal_data['id']}")
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] != seal_data["id"]
    assert data["status"] == "active"
    assert data["previous_seal_id"] == seal_data["id"]
    assert data["system_id"] == seal_data["system_id"]


@pytest.mark.asyncio
async def test_api_v1_seal_status(client):
    """GET /api/v1/seal/status/{seal_id} works via v1 router."""
    seal_data = await _issue_seal_via_api(client)

    resp = await client.get(f"/api/v1/seal/status/{seal_data['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["seal_id"] == seal_data["id"]
    assert data["status"] == "active"
    assert "days_remaining" in data


# ===========================================================================
# US-004 — Seal Revocation
# ===========================================================================


@pytest.mark.asyncio
async def test_revoke_seal_marks_revoked(seal_service, issued_seal):
    """revoke_seal sets status to revoked and returns reason."""
    result = await seal_service.revoke_seal(issued_seal["id"], "Policy violation")

    assert result["status"] == "revoked"
    assert result["reason"] == "Policy violation"
    assert result["seal_id"] == issued_seal["id"]

    # Verify the DB was updated
    db_seal = await seal_service.db.get_seal(issued_seal["id"])
    assert db_seal["status"] == "revoked"


@pytest.mark.asyncio
async def test_revoke_seal_is_irreversible(seal_service, issued_seal):
    """Revoking an already-revoked seal raises ValueError."""
    await seal_service.revoke_seal(issued_seal["id"], "First revocation")

    with pytest.raises(ValueError, match="already revoked"):
        await seal_service.revoke_seal(issued_seal["id"], "Second attempt")


@pytest.mark.asyncio
async def test_revoke_seal_creates_forensic_event(db, seal_service, issued_seal):
    """Revocation records a SEAL_REVOKED forensic event with reason."""
    await seal_service.revoke_seal(issued_seal["id"], "Audit finding")

    events = await db.get_all_events()
    revoke_events = [e for e in events if e.event_type == EventType.SEAL_REVOKED]
    assert len(revoke_events) >= 1
    latest = revoke_events[-1]
    assert latest.payload["seal_id"] == issued_seal["id"]
    assert latest.payload["reason"] == "Audit finding"


@pytest.mark.asyncio
async def test_revoked_seal_verify_returns_invalid(seal_service, issued_seal):
    """verify_seal returns valid=false for a revoked seal."""
    await seal_service.revoke_seal(issued_seal["id"], "Compromised")

    result = await seal_service.verify_seal(seal_id=issued_seal["id"])

    assert result["valid"] is False
    assert result["checks"]["revocation"]["valid"] is False
    assert "revoked" in result["checks"]["revocation"]["detail"].lower()


# ---------------------------------------------------------------------------
# US-004 — API Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_revoke_seal(client):
    """POST /seal/revoke/{seal_id} revokes the seal and returns result."""
    seal_data = await _issue_seal_via_api(client)

    resp = await client.post(
        f"/seal/revoke/{seal_data['id']}",
        json={"reason": "Non-compliance detected"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "revoked"
    assert data["reason"] == "Non-compliance detected"
    assert data["seal_id"] == seal_data["id"]

    # Verify it's actually revoked via status endpoint
    status_resp = await client.get(f"/seal/status/{seal_data['id']}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "revoked"


@pytest.mark.asyncio
async def test_api_revoke_requires_reason(client):
    """POST /seal/revoke/{seal_id} without reason returns 422."""
    seal_data = await _issue_seal_via_api(client)

    # No body at all
    resp = await client.post(f"/seal/revoke/{seal_data['id']}")
    assert resp.status_code == 422

    # Empty reason (violates min_length=1)
    resp2 = await client.post(
        f"/seal/revoke/{seal_data['id']}",
        json={"reason": ""},
    )
    assert resp2.status_code == 422


@pytest.mark.asyncio
async def test_api_v1_revoke_seal(client):
    """POST /api/v1/seal/revoke/{seal_id} works via v1 router."""
    seal_data = await _issue_seal_via_api(client)

    resp = await client.post(
        f"/api/v1/seal/revoke/{seal_data['id']}",
        json={"reason": "V1 revocation test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "revoked"
    assert data["reason"] == "V1 revocation test"
