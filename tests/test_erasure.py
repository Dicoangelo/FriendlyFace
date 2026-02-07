"""Tests for US-050: Cryptographic Erasure.

Tests SubjectKeyManager and ErasureManager for GDPR Art 17 compliance.
"""

from __future__ import annotations

import pytest

from friendlyface.governance.erasure import ErasureManager, SubjectKeyManager
from friendlyface.storage.database import Database


@pytest.fixture
async def erasure_db(tmp_path):
    """Database with migration tables for erasure testing."""
    db = Database(tmp_path / "erasure_test.db")
    await db.connect()
    await db.run_migrations()
    yield db
    await db.close()


@pytest.fixture
def key_manager(erasure_db):
    return SubjectKeyManager(erasure_db)


@pytest.fixture
def erasure_manager(erasure_db):
    return ErasureManager(erasure_db)


class TestSubjectKeyManager:
    @pytest.mark.asyncio
    async def test_create_key(self, key_manager):
        key = await key_manager.create_key("subject-1")
        assert isinstance(key, bytes)
        assert len(key) == 32

    @pytest.mark.asyncio
    async def test_get_key(self, key_manager):
        original = await key_manager.create_key("subject-2")
        retrieved = await key_manager.get_key("subject-2")
        assert retrieved == original

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, key_manager):
        result = await key_manager.get_key("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_key_creates(self, key_manager):
        key = await key_manager.get_or_create_key("subject-3")
        assert len(key) == 32

    @pytest.mark.asyncio
    async def test_get_or_create_key_returns_existing(self, key_manager):
        first = await key_manager.get_or_create_key("subject-4")
        second = await key_manager.get_or_create_key("subject-4")
        assert first == second

    @pytest.mark.asyncio
    async def test_delete_key(self, key_manager):
        await key_manager.create_key("subject-5")
        deleted = await key_manager.delete_key("subject-5")
        assert deleted is True
        # Key should be gone
        result = await key_manager.get_key("subject-5")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, key_manager):
        deleted = await key_manager.delete_key("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_key_exists(self, key_manager):
        await key_manager.create_key("subject-6")
        assert await key_manager.key_exists("subject-6") is True
        assert await key_manager.key_exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_key_exists_after_delete(self, key_manager):
        await key_manager.create_key("subject-7")
        await key_manager.delete_key("subject-7")
        assert await key_manager.key_exists("subject-7") is False


class TestErasureManager:
    @pytest.mark.asyncio
    async def test_erase_subject_with_key(self, erasure_manager, key_manager):
        await key_manager.create_key("erase-1")
        result = await erasure_manager.erase_subject("erase-1")
        assert result["status"] == "completed"
        assert result["subject_id"] == "erase-1"
        assert result["method"] == "key_deletion"
        assert "subject_keys" in result["tables_affected"]

    @pytest.mark.asyncio
    async def test_erase_subject_no_data(self, erasure_manager):
        result = await erasure_manager.erase_subject("no-data")
        assert result["status"] == "no_data"
        assert result["event_count"] == 0

    @pytest.mark.asyncio
    async def test_erase_records_erasure_record(self, erasure_manager, key_manager):
        await key_manager.create_key("erase-2")
        await erasure_manager.erase_subject("erase-2")
        status = await erasure_manager.get_erasure_status("erase-2")
        assert status["erased"] is True
        assert status["has_active_key"] is False
        assert status["erasure_record"] is not None
        assert status["erasure_record"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_erasure_status_no_erasure(self, erasure_manager):
        status = await erasure_manager.get_erasure_status("never-erased")
        assert status["erased"] is False
        assert status["erasure_record"] is None

    @pytest.mark.asyncio
    async def test_list_erasure_records_empty(self, erasure_manager):
        records, total = await erasure_manager.list_erasure_records()
        assert records == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_list_erasure_records(self, erasure_manager, key_manager):
        await key_manager.create_key("list-1")
        await key_manager.create_key("list-2")
        await erasure_manager.erase_subject("list-1")
        await erasure_manager.erase_subject("list-2")
        records, total = await erasure_manager.list_erasure_records()
        assert total == 2
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_list_erasure_records_pagination(self, erasure_manager, key_manager):
        for i in range(5):
            await key_manager.create_key(f"page-{i}")
            await erasure_manager.erase_subject(f"page-{i}")
        records, total = await erasure_manager.list_erasure_records(limit=2, offset=0)
        assert total == 5
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_double_erasure_idempotent(self, erasure_manager, key_manager):
        await key_manager.create_key("double-erase")
        first = await erasure_manager.erase_subject("double-erase")
        assert first["status"] == "completed"
        second = await erasure_manager.erase_subject("double-erase")
        assert second["status"] == "no_data"


class TestErasureEndpoints:
    @pytest.mark.asyncio
    async def test_erase_endpoint(self, client):
        resp = await client.post("/erasure/erase/test-subject")
        assert resp.status_code == 200
        body = resp.json()
        assert body["subject_id"] == "test-subject"

    @pytest.mark.asyncio
    async def test_erasure_status_endpoint(self, client):
        resp = await client.get("/erasure/status/test-subject")
        assert resp.status_code == 200
        assert "erased" in resp.json()

    @pytest.mark.asyncio
    async def test_erasure_records_endpoint(self, client):
        resp = await client.get("/erasure/records")
        assert resp.status_code == 200
        assert "items" in resp.json()
        assert "total" in resp.json()
