"""Shared fixtures for FriendlyFace tests."""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import friendlyface.api.app as app_module
from friendlyface.api.app import _dashboard_cache, _db, _service, app, limiter
from friendlyface.core.service import ForensicService
from friendlyface.storage.database import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    """Fresh in-memory database for each test."""
    database = Database(tmp_path / "test.db")
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
    # Swap the global DB for tests
    _db.db_path = tmp_path / "api_test.db"
    await _db.connect()
    await _db.run_migrations()
    await _service.initialize()

    # Reset in-memory state
    _service.merkle = __import__("friendlyface.core.merkle", fromlist=["MerkleTree"]).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()
    _dashboard_cache["data"] = None
    _dashboard_cache["timestamp"] = 0.0

    # Reset app-level in-memory stores
    app_module._explanations.clear()
    app_module._model_registry.clear()
    app_module._fl_simulations.clear()
    app_module._latest_compliance_report = None
    app_module._auto_audit_interval = 50
    app_module._recognition_event_count = 0

    # Disable rate limiter for tests
    limiter.enabled = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()
