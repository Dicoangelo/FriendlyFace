"""Tests for ForensicSeal issuance (US-001).

Covers:
  - Seal issuance with valid bundles (happy path)
  - Seal issuance fails when compliance score below threshold
  - Seal retrieval by ID
  - Seal listing with pagination
  - Seal credential is a valid W3C VC structure
  - Seal contains ZK proof
  - Seal contains Merkle root
  - Seal has correct expiry
"""

from __future__ import annotations

import json
from datetime import datetime

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
    database = Database(tmp_path / "test_seal.db")
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
async def client(tmp_path):
    """HTTP test client wired to a fresh database."""
    _db.db_path = tmp_path / "seal_api_test.db"
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
# Helper: create a bundle via the service for seal tests
# ---------------------------------------------------------------------------


async def _create_test_bundle(service: ForensicService) -> str:
    """Create a minimal forensic bundle and return its ID as string."""
    event = await service.record_event(
        event_type=EventType.INFERENCE_RESULT,
        actor="test",
        payload={"model": "pca_svm", "prediction": 0},
    )
    bundle = await service.create_bundle(event_ids=[event.id])
    return str(bundle.id)


# ---------------------------------------------------------------------------
# Unit tests — ForensicSealService
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seal_issuance_happy_path(db, service):
    """Seal is issued when compliance score meets threshold."""
    bundle_id = await _create_test_bundle(service)
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    seal = await seal_svc.issue_seal(
        system_id="sys-001",
        system_name="TestSystem",
        bundle_ids=[bundle_id],
        threshold=0.0,  # Low threshold to ensure pass
    )

    assert seal["id"]
    assert seal["system_id"] == "sys-001"
    assert seal["system_name"] == "TestSystem"
    assert seal["status"] == "active"
    assert seal["compliance_score"] >= 0


@pytest.mark.asyncio
async def test_seal_issuance_fails_below_threshold(db, service):
    """Seal issuance raises ValueError when score is below threshold."""
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    # With no data, compliance score will be 0 — use a high threshold
    with pytest.raises(ValueError, match="below threshold"):
        await seal_svc.issue_seal(
            system_id="sys-fail",
            system_name="FailSystem",
            threshold=99.0,
        )


@pytest.mark.asyncio
async def test_seal_credential_is_valid_vc(db, service):
    """Seal credential follows W3C VC structure."""
    bundle_id = await _create_test_bundle(service)
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    seal = await seal_svc.issue_seal(
        system_id="sys-vc",
        system_name="VCTest",
        bundle_ids=[bundle_id],
        threshold=0.0,
    )

    cred = seal["credential"]
    assert "@context" in cred
    assert "https://www.w3.org/2018/credentials/v1" in cred["@context"]
    assert "VerifiableCredential" in cred["type"]
    assert "ForensicSealCredential" in cred["type"]
    assert "proof" in cred
    assert cred["proof"]["type"] == "Ed25519Signature2020"
    assert "credentialSubject" in cred
    assert cred["credentialSubject"]["seal_id"] == seal["id"]


@pytest.mark.asyncio
async def test_seal_contains_zk_proof(db, service):
    """Seal contains a valid Schnorr ZK proof."""
    bundle_id = await _create_test_bundle(service)
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    seal = await seal_svc.issue_seal(
        system_id="sys-zk",
        system_name="ZKTest",
        bundle_ids=[bundle_id],
        threshold=0.0,
    )

    zk_proof = seal["zk_proof"]
    assert zk_proof is not None
    proof_data = json.loads(zk_proof)
    assert proof_data["scheme"] == "schnorr-sha256"
    assert "commitment" in proof_data
    assert "challenge" in proof_data
    assert "response" in proof_data
    assert "public_point" in proof_data

    # Verify the ZK proof
    from friendlyface.crypto.schnorr import ZKBundleVerifier

    verifier = ZKBundleVerifier()
    assert verifier.verify_bundle(zk_proof) is True


@pytest.mark.asyncio
async def test_seal_contains_merkle_root(db, service):
    """Seal contains the current Merkle root."""
    bundle_id = await _create_test_bundle(service)
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    seal = await seal_svc.issue_seal(
        system_id="sys-merkle",
        system_name="MerkleTest",
        bundle_ids=[bundle_id],
        threshold=0.0,
    )

    assert seal["merkle_root"]
    assert isinstance(seal["merkle_root"], str)
    assert len(seal["merkle_root"]) == 64  # SHA-256 hex


@pytest.mark.asyncio
async def test_seal_has_correct_expiry(db, service):
    """Seal expires_at is issued_at + expiry_days."""
    bundle_id = await _create_test_bundle(service)
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    expiry_days = 30
    seal = await seal_svc.issue_seal(
        system_id="sys-expiry",
        system_name="ExpiryTest",
        bundle_ids=[bundle_id],
        threshold=0.0,
        expiry_days=expiry_days,
    )

    issued = datetime.fromisoformat(seal["issued_at"])
    expires = datetime.fromisoformat(seal["expires_at"])
    delta = expires - issued
    assert abs(delta.days - expiry_days) <= 1  # Allow 1-day rounding


@pytest.mark.asyncio
async def test_seal_retrieval_by_id(db, service):
    """Seal can be retrieved from DB by ID."""
    bundle_id = await _create_test_bundle(service)
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    seal = await seal_svc.issue_seal(
        system_id="sys-get",
        system_name="GetTest",
        bundle_ids=[bundle_id],
        threshold=0.0,
    )

    retrieved = await db.get_seal(seal["id"])
    assert retrieved is not None
    assert retrieved["id"] == seal["id"]
    assert retrieved["system_id"] == "sys-get"
    assert retrieved["status"] == "active"


@pytest.mark.asyncio
async def test_seal_listing_with_pagination(db, service):
    """Seals can be listed with pagination and system_id filter."""
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    # Create two seals for different systems
    bundle_id = await _create_test_bundle(service)
    await seal_svc.issue_seal(
        system_id="sys-a",
        system_name="SystemA",
        bundle_ids=[bundle_id],
        threshold=0.0,
    )

    bundle_id2 = await _create_test_bundle(service)
    await seal_svc.issue_seal(
        system_id="sys-b",
        system_name="SystemB",
        bundle_ids=[bundle_id2],
        threshold=0.0,
    )

    # List all
    items, total = await db.list_seals()
    assert total == 2
    assert len(items) == 2

    # Filter by system_id
    items_a, total_a = await db.list_seals(system_id="sys-a")
    assert total_a == 1
    assert items_a[0]["system_id"] == "sys-a"

    # Pagination
    items_page, total_page = await db.list_seals(limit=1, offset=0)
    assert len(items_page) == 1
    assert total_page == 2


@pytest.mark.asyncio
async def test_seal_records_forensic_event(db, service):
    """Seal issuance records a SEAL_ISSUED forensic event."""
    bundle_id = await _create_test_bundle(service)
    reporter = ComplianceReporter(db, service)
    seal_svc = ForensicSealService(db, service, reporter)

    seal = await seal_svc.issue_seal(
        system_id="sys-event",
        system_name="EventTest",
        bundle_ids=[bundle_id],
        threshold=0.0,
    )

    # Check that a seal_issued event was recorded
    events = await db.get_all_events()
    seal_events = [e for e in events if e.event_type == EventType.SEAL_ISSUED]
    assert len(seal_events) >= 1
    latest = seal_events[-1]
    assert latest.payload["seal_id"] == seal["id"]


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_issue_seal(client):
    """POST /seal/issue returns 201 with seal data."""
    resp = await client.post(
        "/seal/issue",
        json={
            "system_id": "api-sys",
            "system_name": "API Test System",
            "threshold": 0.0,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["system_id"] == "api-sys"
    assert data["status"] == "active"
    assert "credential" in data
    assert "compliance_score" in data


@pytest.mark.asyncio
async def test_api_issue_seal_below_threshold(client):
    """POST /seal/issue returns 422 when below threshold."""
    resp = await client.post(
        "/seal/issue",
        json={
            "system_id": "api-fail",
            "system_name": "Fail System",
            "threshold": 99.0,
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_api_get_seal(client):
    """GET /seal/{id} returns seal details."""
    # First issue a seal
    issue_resp = await client.post(
        "/seal/issue",
        json={
            "system_id": "api-get",
            "system_name": "Get System",
            "threshold": 0.0,
        },
    )
    seal_id = issue_resp.json()["id"]

    # Retrieve it
    resp = await client.get(f"/seal/{seal_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == seal_id
    assert data["system_id"] == "api-get"


@pytest.mark.asyncio
async def test_api_get_seal_not_found(client):
    """GET /seal/{id} returns 404 for nonexistent seal."""
    resp = await client.get("/seal/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_api_list_seals(client):
    """GET /seals returns paginated list."""
    # Issue two seals
    await client.post(
        "/seal/issue",
        json={"system_id": "list-1", "system_name": "S1", "threshold": 0.0},
    )
    await client.post(
        "/seal/issue",
        json={"system_id": "list-2", "system_name": "S2", "threshold": 0.0},
    )

    resp = await client.get("/seals")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2
    assert "limit" in data
    assert "offset" in data


@pytest.mark.asyncio
async def test_api_list_seals_filter_system_id(client):
    """GET /seals?system_id=X filters results."""
    await client.post(
        "/seal/issue",
        json={"system_id": "filter-a", "system_name": "A", "threshold": 0.0},
    )
    await client.post(
        "/seal/issue",
        json={"system_id": "filter-b", "system_name": "B", "threshold": 0.0},
    )

    resp = await client.get("/seals?system_id=filter-a")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["system_id"] == "filter-a"
