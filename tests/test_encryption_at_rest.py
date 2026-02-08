"""Tests for encryption at rest (US-041).

Covers the Database.connect() encryption parameters, config options,
and error handling for missing keys when encryption is required.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from friendlyface.storage.database import Database


# ---------------------------------------------------------------------------
# connect() — db_key and require_encryption parameters
# ---------------------------------------------------------------------------


class TestEncryptionAtRestConnect:
    """Database.connect() encryption parameter handling."""

    @pytest_asyncio.fixture
    async def db_path(self, tmp_path):
        return tmp_path / "enc_test.db"

    async def test_connect_without_encryption(self, db_path):
        """Default connect (no key) works as before."""
        db = Database(db_path)
        await db.connect()
        assert db._db is not None

        # Verify basic operation
        cursor = await db.db.execute("SELECT 1")
        row = await cursor.fetchone()
        assert row[0] == 1
        await db.close()

    async def test_connect_with_db_key(self, db_path):
        """Providing a db_key executes PRAGMA key.

        Note: Without SQLCipher compiled into SQLite, PRAGMA key is
        silently ignored.  We verify it does not raise an error.
        """
        db = Database(db_path)
        await db.connect(db_key="test-encryption-key-32bytes!!")
        assert db._db is not None

        # DB should still be operational
        cursor = await db.db.execute("SELECT 1")
        row = await cursor.fetchone()
        assert row[0] == 1
        await db.close()

    async def test_require_encryption_without_key_raises(self, db_path):
        """require_encryption=True without db_key raises RuntimeError."""
        db = Database(db_path)
        with pytest.raises(RuntimeError, match="FF_REQUIRE_ENCRYPTION"):
            await db.connect(require_encryption=True)

    async def test_require_encryption_with_key_succeeds(self, db_path):
        """require_encryption=True with db_key proceeds normally."""
        db = Database(db_path)
        await db.connect(db_key="my-secret-key", require_encryption=True)
        assert db._db is not None
        await db.close()

    async def test_connect_creates_schema_with_key(self, db_path):
        """Schema tables are created even when a key is provided."""
        db = Database(db_path)
        await db.connect(db_key="schema-test-key")

        cursor = await db.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in await cursor.fetchall()]
        assert "forensic_events" in tables
        assert "forensic_bundles" in tables
        assert "provenance_nodes" in tables
        await db.close()

    async def test_db_key_none_is_default(self, db_path):
        """Passing db_key=None is equivalent to no encryption."""
        db = Database(db_path)
        await db.connect(db_key=None)

        cursor = await db.db.execute("SELECT COUNT(*) FROM forensic_events")
        row = await cursor.fetchone()
        assert row[0] == 0
        await db.close()

    async def test_require_encryption_false_no_key_ok(self, db_path):
        """require_encryption=False with no key is fine (default)."""
        db = Database(db_path)
        await db.connect(require_encryption=False)
        assert db._db is not None
        await db.close()


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestEncryptionConfig:
    """Config fields for encryption at rest."""

    def test_default_db_key_is_none(self):
        from friendlyface.config import Settings

        s = Settings(FF_API_KEYS="", FF_DB_PATH=":memory:")
        assert s.db_key is None

    def test_default_require_encryption_is_false(self):
        from friendlyface.config import Settings

        s = Settings(FF_API_KEYS="", FF_DB_PATH=":memory:")
        assert s.require_encryption is False

    def test_db_key_from_env(self, monkeypatch):
        monkeypatch.setenv("FF_DB_KEY", "my-secret")
        from friendlyface.config import Settings

        s = Settings()
        assert s.db_key == "my-secret"

    def test_require_encryption_from_env(self, monkeypatch):
        monkeypatch.setenv("FF_REQUIRE_ENCRYPTION", "true")
        from friendlyface.config import Settings

        s = Settings()
        assert s.require_encryption is True


# ---------------------------------------------------------------------------
# Data operations with encryption key
# ---------------------------------------------------------------------------


class TestEncryptedDataOps:
    """Verify normal DB operations work when a key is provided."""

    async def test_insert_and_read_event(self, tmp_path):
        from datetime import datetime, timezone
        from uuid import uuid4

        from friendlyface.core.models import EventType, ForensicEvent

        db = Database(tmp_path / "ops_test.db")
        await db.connect(db_key="ops-test-key-12345")

        event = ForensicEvent(
            id=uuid4(),
            event_type=EventType.MODEL_REGISTERED,
            timestamp=datetime.now(timezone.utc),
            actor="test",
            payload={"key": "value"},
            previous_hash="GENESIS",
            event_hash="abc123",
            sequence_number=1,
        )
        await db.insert_event(event)

        fetched = await db.get_event(event.id)
        assert fetched is not None
        assert fetched.actor == "test"
        assert fetched.event_hash == "abc123"
        await db.close()

    async def test_event_count_with_key(self, tmp_path):
        db = Database(tmp_path / "count_test.db")
        await db.connect(db_key="count-key")
        count = await db.get_event_count()
        assert count == 0
        await db.close()
