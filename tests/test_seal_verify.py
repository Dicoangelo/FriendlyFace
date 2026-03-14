"""Tests for ForensicSeal public verification (US-002).

Covers:
  - Verification of a valid seal by credential (offline)
  - Verification of a valid seal by ID (DB lookup)
  - Verification detects expired seal
  - Verification detects revoked seal
  - Verification detects tampered/invalid credential
  - Verification returns correct check structure
  - Verification endpoints are public (no auth needed)
  - Verification with non-existent seal_id returns 404
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import friendlyface.api.app as app_module
from friendlyface.api.app import _db, _service, app, limiter
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
    database = Database(tmp_path / "test_seal_verify.db")
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
        system_id="verify-sys",
        system_name="VerifyTestSystem",
        threshold=0.0,
        expiry_days=90,
    )


@pytest_asyncio.fixture
async def client(tmp_path):
    """HTTP test client wired to a fresh database."""
    _db.db_path = tmp_path / "seal_verify_api_test.db"
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
            "system_id": "api-verify-sys",
            "system_name": "API Verify System",
            "threshold": 0.0,
        },
    )
    assert resp.status_code == 201
    return resp.json()


# ---------------------------------------------------------------------------
# Unit tests — ForensicSealService.verify_seal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_valid_seal_by_credential(seal_service, issued_seal):
    """Verify a freshly issued seal by passing its credential."""
    result = await seal_service.verify_seal(credential=issued_seal["credential"])

    assert result["valid"] is True
    assert result["checks"]["signature"]["valid"] is True
    assert result["checks"]["expiry"]["valid"] is True
    assert result["checks"]["revocation"]["valid"] is True
    assert result["checks"]["zk_proof"]["valid"] is True
    assert result["checks"]["merkle"]["valid"] is True
    assert result["issuer"]
    assert result["system_id"] == "verify-sys"
    assert result["system_name"] == "VerifyTestSystem"
    assert result["compliance_score"] >= 0


@pytest.mark.asyncio
async def test_verify_valid_seal_by_id(seal_service, issued_seal):
    """Verify a freshly issued seal by passing its seal_id."""
    result = await seal_service.verify_seal(seal_id=issued_seal["id"])

    assert result["valid"] is True
    assert result["checks"]["signature"]["valid"] is True
    assert result["checks"]["expiry"]["valid"] is True
    assert result["checks"]["revocation"]["valid"] is True
    assert result["checks"]["zk_proof"]["valid"] is True


@pytest.mark.asyncio
async def test_verify_detects_expired_seal(db, service):
    """Verification catches an expired seal."""
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    # Issue a seal with a very short expiry, then tamper expires_at in DB
    seal = await seal_svc.issue_seal(
        system_id="expire-sys",
        system_name="ExpireTest",
        threshold=0.0,
        expiry_days=7,
    )

    # Tamper the credential to have an expired date
    cred = seal["credential"]
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    cred["credentialSubject"]["expires_at"] = past

    # Re-sign won't happen — we're testing the expiry check specifically
    # The signature will still be invalid because we tampered, but expiry should also fail
    result = await seal_svc.verify_seal(credential=cred)

    assert result["checks"]["expiry"]["valid"] is False
    assert "expired" in result["checks"]["expiry"]["detail"].lower()


@pytest.mark.asyncio
async def test_verify_detects_revoked_seal(db, service):
    """Verification catches a revoked seal."""
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    seal = await seal_svc.issue_seal(
        system_id="revoke-sys",
        system_name="RevokeTest",
        threshold=0.0,
    )

    # Revoke the seal in DB
    await db.update_seal_status(seal["id"], "revoked", reason="Test revocation")

    result = await seal_svc.verify_seal(seal_id=seal["id"])

    assert result["valid"] is False
    assert result["checks"]["revocation"]["valid"] is False
    assert "revoked" in result["checks"]["revocation"]["detail"].lower()


@pytest.mark.asyncio
async def test_verify_detects_tampered_credential(seal_service, issued_seal):
    """Verification catches a tampered credential (invalid signature)."""
    cred = issued_seal["credential"].copy()
    # Tamper with a claim
    cred["credentialSubject"]["compliance_score"] = 99.99

    result = await seal_service.verify_seal(credential=cred)

    assert result["checks"]["signature"]["valid"] is False
    assert result["valid"] is False


@pytest.mark.asyncio
async def test_verify_returns_correct_structure(seal_service, issued_seal):
    """Verification returns the documented structure."""
    result = await seal_service.verify_seal(credential=issued_seal["credential"])

    # Top-level keys
    assert "valid" in result
    assert "checks" in result
    assert "issuer" in result
    assert "issued_at" in result
    assert "expires_at" in result
    assert "compliance_score" in result
    assert "system_id" in result
    assert "system_name" in result

    # Check sub-keys
    checks = result["checks"]
    for check_name in ["signature", "expiry", "revocation", "zk_proof", "merkle"]:
        assert check_name in checks
        assert "valid" in checks[check_name]
        assert "detail" in checks[check_name]


@pytest.mark.asyncio
async def test_verify_nonexistent_seal_id_raises(seal_service):
    """Verify with non-existent seal_id raises LookupError."""
    with pytest.raises(LookupError, match="not found"):
        await seal_service.verify_seal(seal_id="nonexistent-id")


@pytest.mark.asyncio
async def test_verify_requires_credential_or_seal_id(seal_service):
    """Verify with neither credential nor seal_id raises ValueError."""
    with pytest.raises(ValueError, match="Must provide"):
        await seal_service.verify_seal()


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_verify_seal_by_credential(client):
    """POST /seal/verify returns verification result."""
    seal_data = await _issue_seal_via_api(client)

    resp = await client.post(
        "/seal/verify",
        json={"credential": seal_data["credential"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["checks"]["signature"]["valid"] is True


@pytest.mark.asyncio
async def test_api_verify_seal_by_id(client):
    """GET /seal/verify/{seal_id} returns verification result."""
    seal_data = await _issue_seal_via_api(client)

    resp = await client.get(f"/seal/verify/{seal_data['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["checks"]["signature"]["valid"] is True


@pytest.mark.asyncio
async def test_api_verify_seal_not_found(client):
    """GET /seal/verify/{seal_id} returns 404 for non-existent seal."""
    resp = await client.get("/seal/verify/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_api_verify_is_public_no_auth(client):
    """Verification endpoints are public — no auth headers needed."""
    seal_data = await _issue_seal_via_api(client)

    # POST /seal/verify — no auth header
    resp_post = await client.post(
        "/seal/verify",
        json={"credential": seal_data["credential"]},
    )
    assert resp_post.status_code == 200

    # GET /seal/verify/{id} — no auth header
    resp_get = await client.get(f"/seal/verify/{seal_data['id']}")
    assert resp_get.status_code == 200


@pytest.mark.asyncio
async def test_api_verify_tampered_credential(client):
    """POST /seal/verify detects tampered credential."""
    seal_data = await _issue_seal_via_api(client)
    cred = seal_data["credential"]
    cred["credentialSubject"]["compliance_score"] = 99.99

    resp = await client.post("/seal/verify", json={"credential": cred})
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert data["checks"]["signature"]["valid"] is False


@pytest.mark.asyncio
async def test_api_v1_verify_seal_by_credential(client):
    """POST /api/v1/seal/verify works via v1 router."""
    seal_data = await _issue_seal_via_api(client)

    resp = await client.post(
        "/api/v1/seal/verify",
        json={"credential": seal_data["credential"]},
    )
    assert resp.status_code == 200
    assert resp.json()["valid"] is True


@pytest.mark.asyncio
async def test_api_v1_verify_seal_by_id(client):
    """GET /api/v1/seal/verify/{seal_id} works via v1 router."""
    seal_data = await _issue_seal_via_api(client)

    resp = await client.get(f"/api/v1/seal/verify/{seal_data['id']}")
    assert resp.status_code == 200
    assert resp.json()["valid"] is True
