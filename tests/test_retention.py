"""Tests for data retention policies (US-051).

Covers RetentionEngine CRUD, evaluation, database methods,
and the API endpoints.
"""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.api.app import _db, _service, app, limiter
from friendlyface.governance.retention import RetentionEngine
from friendlyface.storage.database import Database


RETENTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS retention_policies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    retention_days INTEGER NOT NULL,
    action TEXT NOT NULL DEFAULT 'erase',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS consent_records (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    granted INTEGER NOT NULL DEFAULT 1,
    timestamp TEXT NOT NULL,
    expiry TEXT,
    revocation_reason TEXT,
    event_id TEXT
);

CREATE TABLE IF NOT EXISTS subject_keys (
    subject_id TEXT PRIMARY KEY,
    encrypted_key BLOB NOT NULL,
    key_nonce BLOB NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS erasure_records (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    tables_affected TEXT NOT NULL DEFAULT '[]',
    event_count INTEGER NOT NULL DEFAULT 0,
    method TEXT NOT NULL DEFAULT 'key_deletion'
);
"""


# ---------------------------------------------------------------------------
# Database-level retention policy tests
# ---------------------------------------------------------------------------


class TestRetentionPolicyDatabase:
    """Database CRUD for retention_policies table."""

    @pytest_asyncio.fixture
    async def db(self, tmp_path):
        database = Database(tmp_path / "retention_test.db")
        await database.connect()
        await database.db.executescript(RETENTION_SCHEMA)
        await database.db.commit()
        yield database
        await database.close()

    async def test_insert_and_get(self, db):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        policy = {
            "id": "pol-1",
            "name": "Test Policy",
            "entity_type": "consent",
            "retention_days": 365,
            "action": "erase",
            "enabled": True,
            "created_at": now,
            "updated_at": now,
        }
        await db.insert_retention_policy(policy)
        result = await db.get_retention_policy("pol-1")
        assert result is not None
        assert result["name"] == "Test Policy"
        assert result["retention_days"] == 365

    async def test_get_nonexistent(self, db):
        assert await db.get_retention_policy("missing") is None

    async def test_list_empty(self, db):
        policies = await db.list_retention_policies()
        assert policies == []

    async def test_list_enabled_only(self, db):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        await db.insert_retention_policy(
            {
                "id": "enabled",
                "name": "Enabled",
                "entity_type": "consent",
                "retention_days": 30,
                "action": "erase",
                "enabled": True,
                "created_at": now,
                "updated_at": now,
            }
        )
        await db.insert_retention_policy(
            {
                "id": "disabled",
                "name": "Disabled",
                "entity_type": "consent",
                "retention_days": 30,
                "action": "erase",
                "enabled": False,
                "created_at": now,
                "updated_at": now,
            }
        )
        enabled = await db.list_retention_policies(enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0]["id"] == "enabled"

        all_policies = await db.list_retention_policies()
        assert len(all_policies) == 2

    async def test_delete(self, db):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        await db.insert_retention_policy(
            {
                "id": "del-1",
                "name": "ToDelete",
                "entity_type": "consent",
                "retention_days": 7,
                "action": "erase",
                "enabled": True,
                "created_at": now,
                "updated_at": now,
            }
        )
        assert await db.delete_retention_policy("del-1") is True
        assert await db.get_retention_policy("del-1") is None

    async def test_delete_nonexistent(self, db):
        assert await db.delete_retention_policy("ghost") is False


# ---------------------------------------------------------------------------
# RetentionEngine unit tests
# ---------------------------------------------------------------------------


class TestRetentionEngine:
    """RetentionEngine business logic."""

    @pytest_asyncio.fixture
    async def db(self, tmp_path):
        database = Database(tmp_path / "engine_test.db")
        await database.connect()
        await database.db.executescript(RETENTION_SCHEMA)
        await database.db.commit()
        yield database
        await database.close()

    async def test_create_policy(self, db):
        engine = RetentionEngine(db)
        policy = await engine.create_policy(
            name="GDPR Consent",
            entity_type="consent",
            retention_days=365,
        )
        assert policy["name"] == "GDPR Consent"
        assert policy["retention_days"] == 365
        assert policy["enabled"] is True

    async def test_list_policies(self, db):
        engine = RetentionEngine(db)
        await engine.create_policy(name="P1", entity_type="consent", retention_days=30)
        await engine.create_policy(name="P2", entity_type="subject", retention_days=90)
        policies = await engine.list_policies()
        assert len(policies) == 2

    async def test_delete_policy(self, db):
        engine = RetentionEngine(db)
        policy = await engine.create_policy(name="Temp", entity_type="consent", retention_days=7)
        assert await engine.delete_policy(policy["id"]) is True
        assert await engine.get_policy(policy["id"]) is None

    async def test_evaluate_no_policies(self, db):
        engine = RetentionEngine(db)
        result = await engine.evaluate()
        assert result["policies_evaluated"] == 0
        assert result["total_subjects_erased"] == 0

    async def test_evaluate_no_expired_subjects(self, db):
        engine = RetentionEngine(db)
        await engine.create_policy(name="Recent", entity_type="consent", retention_days=365)

        # Insert a recent consent record
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        await db.insert_consent_record(
            record_id="c1",
            subject_id="subject-fresh",
            purpose="recognition",
            granted=True,
            timestamp=now,
        )

        result = await engine.evaluate()
        assert result["policies_evaluated"] == 1
        assert result["total_subjects_erased"] == 0

    async def test_evaluate_with_expired_subject(self, db):
        engine = RetentionEngine(db)
        await engine.create_policy(name="Short", entity_type="consent", retention_days=1)

        # Insert an old consent record (2 days ago)
        await db.db.execute(
            "INSERT INTO consent_records (id, subject_id, purpose, granted, timestamp) "
            "VALUES (?, ?, ?, ?, datetime('now', '-2 days'))",
            ("c-old", "subject-expired", "recognition", 1),
        )
        await db.db.commit()

        result = await engine.evaluate()
        assert result["policies_evaluated"] == 1
        assert result["results"][0]["subjects_found"] >= 1


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def retention_client(tmp_path):
    """HTTP test client for retention tests."""
    _db.db_path = tmp_path / "retention_api_test.db"
    await _db.connect()
    await _db.run_migrations()
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


class TestRetentionEndpoints:
    """API endpoints for retention policy CRUD and evaluation."""

    async def test_create_policy(self, retention_client):
        resp = await retention_client.post(
            "/retention/policies",
            json={
                "name": "GDPR Consent",
                "entity_type": "consent",
                "retention_days": 365,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "GDPR Consent"
        assert data["retention_days"] == 365

    async def test_list_policies(self, retention_client):
        await retention_client.post(
            "/retention/policies",
            json={"name": "P1", "entity_type": "consent", "retention_days": 30},
        )
        resp = await retention_client.get("/retention/policies")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    async def test_delete_policy(self, retention_client):
        create_resp = await retention_client.post(
            "/retention/policies",
            json={"name": "ToDelete", "entity_type": "consent", "retention_days": 7},
        )
        policy_id = create_resp.json()["id"]

        del_resp = await retention_client.delete(f"/retention/policies/{policy_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["deleted"] is True

    async def test_delete_nonexistent(self, retention_client):
        resp = await retention_client.delete("/retention/policies/ghost")
        assert resp.status_code == 404

    async def test_evaluate(self, retention_client):
        resp = await retention_client.post("/retention/evaluate")
        assert resp.status_code == 200
        data = resp.json()
        assert "policies_evaluated" in data
        assert "total_subjects_erased" in data

    async def test_list_enabled_only(self, retention_client):
        await retention_client.post(
            "/retention/policies",
            json={
                "name": "Active",
                "entity_type": "consent",
                "retention_days": 30,
                "enabled": True,
            },
        )
        await retention_client.post(
            "/retention/policies",
            json={
                "name": "Inactive",
                "entity_type": "consent",
                "retention_days": 30,
                "enabled": False,
            },
        )
        resp = await retention_client.get("/retention/policies?enabled_only=true")
        assert resp.status_code == 200
        policies = resp.json()["policies"]
        assert all(p["enabled"] for p in policies)
