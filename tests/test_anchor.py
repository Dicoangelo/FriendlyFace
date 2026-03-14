"""Tests for blockchain Merkle root anchoring (US-008)."""

from __future__ import annotations

import sys
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.crypto.anchor import (
    AnchorResult,
    BaseAnchor,
    NullAnchor,
    PolygonAnchor,
    get_anchor,
)
from friendlyface.storage.database import Database


# ---------------------------------------------------------------------------
# Unit tests — anchor module
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_anchor_returns_result():
    """NullAnchor.anchor_root returns a valid AnchorResult."""
    anchor = NullAnchor()
    result = await anchor.anchor_root("abc123")
    assert isinstance(result, AnchorResult)
    assert result.chain == "none"
    assert result.merkle_root == "abc123"
    assert result.block_number == 0
    assert result.tx_hash == "0x" + "0" * 64


@pytest.mark.asyncio
async def test_null_anchor_is_default():
    """get_anchor('none') returns NullAnchor."""
    anchor = get_anchor("none")
    assert isinstance(anchor, NullAnchor)

    # Empty string also defaults to NullAnchor
    anchor2 = get_anchor("")
    assert isinstance(anchor2, NullAnchor)


def test_polygon_anchor_requires_web3():
    """PolygonAnchor raises ImportError when web3 is not installed."""
    # Temporarily hide web3 from imports
    with patch.dict(sys.modules, {"web3": None}):
        with pytest.raises(ImportError, match="web3 is required for PolygonAnchor"):
            PolygonAnchor(private_key="0x" + "ab" * 32)


def test_base_anchor_requires_web3():
    """BaseAnchor raises ImportError when web3 is not installed."""
    with patch.dict(sys.modules, {"web3": None}):
        with pytest.raises(ImportError, match="web3 is required for BaseAnchor"):
            BaseAnchor(private_key="0x" + "ab" * 32)


def test_anchor_factory():
    """get_anchor returns correct types and validates arguments."""
    # NullAnchor — no key needed
    assert isinstance(get_anchor("none"), NullAnchor)
    assert isinstance(get_anchor("  None  "), NullAnchor)

    # Polygon without key → ValueError
    with pytest.raises(ValueError, match="FF_ANCHOR_KEY is required"):
        get_anchor("polygon")

    # Base without key → ValueError
    with pytest.raises(ValueError, match="FF_ANCHOR_KEY is required"):
        get_anchor("base")

    # Unknown chain → ValueError
    with pytest.raises(ValueError, match="Unknown anchor chain"):
        get_anchor("ethereum")


def test_anchor_result_structure():
    """AnchorResult has all expected fields and to_dict works."""
    result = AnchorResult(
        tx_hash="0xabc",
        block_number=42,
        chain="polygon",
        timestamp=1234567890.0,
        merkle_root="deadbeef",
    )
    assert result.tx_hash == "0xabc"
    assert result.block_number == 42
    assert result.chain == "polygon"
    assert result.timestamp == 1234567890.0
    assert result.merkle_root == "deadbeef"

    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["tx_hash"] == "0xabc"
    assert d["block_number"] == 42
    assert d["chain"] == "polygon"


# ---------------------------------------------------------------------------
# Database integration tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def anchor_db(tmp_path):
    """Fresh database for anchor tests."""
    database = Database(tmp_path / "anchor_test.db")
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_save_and_list_anchors_db(anchor_db):
    """save_anchor stores a record and list_anchors retrieves it."""
    anchor_data = {
        "id": str(uuid4()),
        "merkle_root": "abc123def456",
        "chain": "polygon",
        "tx_hash": "0x" + "ff" * 32,
        "block_number": 100,
        "anchored_at": "2026-01-01T00:00:00Z",
    }
    await anchor_db.save_anchor(anchor_data)

    items, total = await anchor_db.list_anchors()
    assert total == 1
    assert len(items) == 1
    assert items[0]["merkle_root"] == "abc123def456"
    assert items[0]["chain"] == "polygon"
    assert items[0]["tx_hash"] == "0x" + "ff" * 32
    assert items[0]["block_number"] == 100


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client(tmp_path):
    """HTTP test client wired to a fresh database."""
    import friendlyface.api.app as app_module
    from friendlyface.api.app import _dashboard_cache, _db, _service, app, limiter
    from friendlyface.recognition.gallery import FaceGallery

    _db.db_path = tmp_path / "anchor_api_test.db"
    await _db.connect()
    await _db.run_migrations()
    await _service.initialize()

    _service.merkle = __import__(
        "friendlyface.core.merkle", fromlist=["MerkleTree"]
    ).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()
    _dashboard_cache["data"] = None
    _dashboard_cache["timestamp"] = 0.0

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


@pytest.mark.asyncio
async def test_api_anchor_history(client):
    """GET /anchor/history returns anchors after saving one."""
    from friendlyface.api.app import _db

    anchor_data = {
        "id": str(uuid4()),
        "merkle_root": "test_root_hash",
        "chain": "none",
        "tx_hash": "0x" + "0" * 64,
        "block_number": 0,
        "anchored_at": "2026-03-14T00:00:00Z",
    }
    await _db.save_anchor(anchor_data)

    resp = await client.get("/anchor/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["merkle_root"] == "test_root_hash"


@pytest.mark.asyncio
async def test_api_anchor_history_empty(client):
    """GET /anchor/history returns empty list when no anchors exist."""
    resp = await client.get("/anchor/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["items"] == []


@pytest.mark.asyncio
async def test_api_v1_anchor_history(client):
    """GET /api/v1/anchor/history works via the versioned route."""
    resp = await client.get("/api/v1/anchor/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["items"] == []
