"""Tests for DID key persistence (US-040).

Covers Ed25519DIDKey serialization, database CRUD for did_keys,
and the API endpoints using DB-backed storage.
"""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.api.app import _db, _service, app, limiter
from friendlyface.crypto.did import Ed25519DIDKey
from friendlyface.storage.database import Database


# ---------------------------------------------------------------------------
# Unit tests — Ed25519DIDKey serialization
# ---------------------------------------------------------------------------


class TestDIDKeySerialization:
    """to_stored_form / from_stored_form round-trip."""

    def test_to_stored_form_keys(self):
        key = Ed25519DIDKey()
        form = key.to_stored_form()
        assert "did" in form
        assert "public_key" in form
        assert "private_key" in form
        assert form["key_type"] == "Ed25519"

    def test_to_stored_form_types(self):
        key = Ed25519DIDKey()
        form = key.to_stored_form()
        assert isinstance(form["public_key"], bytes)
        assert isinstance(form["private_key"], bytes)
        assert len(form["public_key"]) == 32
        assert len(form["private_key"]) == 32

    def test_round_trip(self):
        original = Ed25519DIDKey()
        form = original.to_stored_form()
        restored = Ed25519DIDKey.from_stored_form(form["private_key"])
        assert restored.did == original.did

    def test_round_trip_preserves_signing(self):
        original = Ed25519DIDKey()
        data = b"test message"
        sig = original.sign(data)

        form = original.to_stored_form()
        restored = Ed25519DIDKey.from_stored_form(form["private_key"])
        assert restored.verify(data, sig) is True

    def test_round_trip_deterministic_seed(self):
        seed = bytes.fromhex("a" * 64)
        key1 = Ed25519DIDKey.from_seed(seed)
        form = key1.to_stored_form()
        key2 = Ed25519DIDKey.from_stored_form(form["private_key"])
        assert key1.did == key2.did

    def test_from_stored_form_classmethod(self):
        key = Ed25519DIDKey()
        priv = key.to_stored_form()["private_key"]
        restored = Ed25519DIDKey.from_stored_form(priv)
        assert restored.did == key.did


# ---------------------------------------------------------------------------
# Database CRUD tests
# ---------------------------------------------------------------------------

DID_KEYS_SCHEMA = """
CREATE TABLE IF NOT EXISTS did_keys (
    did TEXT PRIMARY KEY,
    public_key BLOB NOT NULL,
    encrypted_private_key BLOB,
    key_type TEXT NOT NULL DEFAULT 'Ed25519',
    created_at TEXT NOT NULL,
    label TEXT,
    is_platform_key INTEGER NOT NULL DEFAULT 0
);
"""


class TestDIDKeyDatabase:
    """Database-level DID key CRUD."""

    @pytest_asyncio.fixture
    async def db(self, tmp_path):
        database = Database(tmp_path / "did_test.db")
        await database.connect()
        await database.db.executescript(DID_KEYS_SCHEMA)
        await database.db.commit()
        yield database
        await database.close()

    async def test_insert_and_get(self, db):
        key = Ed25519DIDKey()
        form = key.to_stored_form()
        await db.insert_did_key(
            did=form["did"],
            public_key=form["public_key"],
            encrypted_private_key=form["private_key"],
        )
        entry = await db.get_did_key(form["did"])
        assert entry is not None
        assert entry["did"] == form["did"]
        assert entry["public_key"] == form["public_key"]

    async def test_get_nonexistent(self, db):
        result = await db.get_did_key("did:key:nonexistent")
        assert result is None

    async def test_list_empty(self, db):
        keys = await db.list_did_keys()
        assert keys == []

    async def test_list_multiple(self, db):
        for _ in range(3):
            key = Ed25519DIDKey()
            form = key.to_stored_form()
            await db.insert_did_key(did=form["did"], public_key=form["public_key"])
        keys = await db.list_did_keys()
        assert len(keys) == 3

    async def test_platform_key_filter(self, db):
        k1 = Ed25519DIDKey()
        f1 = k1.to_stored_form()
        await db.insert_did_key(did=f1["did"], public_key=f1["public_key"], is_platform_key=True)

        k2 = Ed25519DIDKey()
        f2 = k2.to_stored_form()
        await db.insert_did_key(did=f2["did"], public_key=f2["public_key"], is_platform_key=False)

        platform = await db.list_did_keys(platform_only=True)
        assert len(platform) == 1
        assert platform[0]["did"] == f1["did"]

        all_keys = await db.list_did_keys()
        assert len(all_keys) == 2

    async def test_insert_with_label(self, db):
        key = Ed25519DIDKey()
        form = key.to_stored_form()
        await db.insert_did_key(did=form["did"], public_key=form["public_key"], label="my-key")
        entry = await db.get_did_key(form["did"])
        assert entry["label"] == "my-key"

    async def test_upsert_replaces(self, db):
        key = Ed25519DIDKey()
        form = key.to_stored_form()
        await db.insert_did_key(did=form["did"], public_key=form["public_key"], label="v1")
        await db.insert_did_key(did=form["did"], public_key=form["public_key"], label="v2")
        entry = await db.get_did_key(form["did"])
        assert entry["label"] == "v2"


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def did_client(tmp_path):
    """HTTP test client for DID persistence tests."""
    _db.db_path = tmp_path / "did_api_test.db"
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


class TestDIDEndpoints:
    """API endpoints use DB-backed DID storage."""

    async def test_create_and_resolve(self, did_client):
        resp = await did_client.post("/did/create", json={})
        assert resp.status_code == 201
        did = resp.json()["did"]

        resolve_resp = await did_client.get(f"/did/{did}/resolve")
        assert resolve_resp.status_code == 200
        doc = resolve_resp.json()
        assert doc["id"] == did

    async def test_resolve_nonexistent(self, did_client):
        resp = await did_client.get("/did/did:key:nonexistent/resolve")
        assert resp.status_code == 404

    async def test_create_with_seed(self, did_client):
        seed = "a" * 64
        resp = await did_client.post("/did/create", json={"seed": seed})
        assert resp.status_code == 201
        did = resp.json()["did"]

        # Same seed should produce same DID
        resp2 = await did_client.post("/did/create", json={"seed": seed})
        assert resp2.json()["did"] == did

    async def test_issue_vc_with_db_did(self, did_client):
        create_resp = await did_client.post("/did/create", json={})
        did = create_resp.json()["did"]

        vc_resp = await did_client.post(
            "/vc/issue",
            json={
                "issuer_did_id": did,
                "subject_did": "did:example:subject",
                "claims": {"name": "Test"},
            },
        )
        assert vc_resp.status_code == 201
        assert "proof" in vc_resp.json()
