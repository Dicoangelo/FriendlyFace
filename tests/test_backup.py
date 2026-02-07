"""Tests for US-043: Backup/Recovery.

Tests BackupManager for SQLite backup, verification, and restore.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from friendlyface.ops.backup import BackupManager


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create a test SQLite database with some data."""
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO test_data VALUES (1, 'hello')")
    conn.execute("INSERT INTO test_data VALUES (2, 'world')")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def backup_dir(tmp_path: Path) -> Path:
    return tmp_path / "backups"


@pytest.fixture
def backup_mgr(db_path: Path, backup_dir: Path) -> BackupManager:
    return BackupManager(db_path, backup_dir)


class TestCreateBackup:
    def test_create_backup(self, backup_mgr):
        result = backup_mgr.create_backup()
        assert result["backup_id"]
        assert result["filename"].startswith("ff_backup_")
        assert result["size_bytes"] > 0
        assert result["checksum_sha256"]

    def test_create_backup_with_label(self, backup_mgr):
        result = backup_mgr.create_backup(label="pre-deploy")
        assert result["label"] == "pre-deploy"

    def test_backup_file_exists(self, backup_mgr, backup_dir):
        result = backup_mgr.create_backup()
        assert Path(result["path"]).exists()

    def test_backup_contains_data(self, backup_mgr):
        result = backup_mgr.create_backup()
        conn = sqlite3.connect(result["path"])
        cursor = conn.execute("SELECT COUNT(*) FROM test_data")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 2

    def test_multiple_backups(self, backup_mgr):
        r1 = backup_mgr.create_backup()
        r2 = backup_mgr.create_backup()
        assert r1["filename"] != r2["filename"]


class TestListBackups:
    def test_list_empty(self, backup_mgr):
        assert backup_mgr.list_backups() == []

    def test_list_after_create(self, backup_mgr):
        backup_mgr.create_backup()
        backups = backup_mgr.list_backups()
        assert len(backups) == 1
        assert backups[0]["filename"].startswith("ff_backup_")

    def test_list_multiple(self, backup_mgr):
        backup_mgr.create_backup()
        backup_mgr.create_backup()
        backups = backup_mgr.list_backups()
        assert len(backups) == 2


class TestVerifyBackup:
    def test_verify_valid_backup(self, backup_mgr):
        result = backup_mgr.create_backup()
        verification = backup_mgr.verify_backup(result["filename"])
        assert verification["valid"] is True
        assert verification["integrity_result"] == "ok"

    def test_verify_nonexistent(self, backup_mgr):
        verification = backup_mgr.verify_backup("nonexistent.db")
        assert verification["valid"] is False
        assert "error" in verification

    def test_verify_returns_checksum(self, backup_mgr):
        result = backup_mgr.create_backup()
        verification = backup_mgr.verify_backup(result["filename"])
        assert verification["checksum_sha256"] == result["checksum_sha256"]


class TestRestoreBackup:
    def test_restore_backup(self, backup_mgr, db_path):
        # Create backup
        result = backup_mgr.create_backup()

        # Modify original database
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM test_data")
        conn.commit()
        cursor = conn.execute("SELECT COUNT(*) FROM test_data")
        assert cursor.fetchone()[0] == 0
        conn.close()

        # Restore
        restore_result = backup_mgr.restore_backup(result["filename"])
        assert restore_result["restored"] is True

        # Verify data is back
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM test_data")
        assert cursor.fetchone()[0] == 2
        conn.close()

    def test_restore_nonexistent(self, backup_mgr):
        result = backup_mgr.restore_backup("nonexistent.db")
        assert result["restored"] is False


class TestRetentionPolicy:
    def test_enforce_count_removes_oldest(self, backup_mgr):
        """Create 5 backups, enforce max_count=2 → 3 removed."""
        import time

        for _ in range(5):
            backup_mgr.create_backup()
            time.sleep(0.05)  # Ensure different mtime
        result = backup_mgr.enforce_retention(max_count=2)
        assert result["removed_count"] == 3
        assert result["remaining"] == 2

    def test_enforce_count_no_removal_needed(self, backup_mgr):
        backup_mgr.create_backup()
        result = backup_mgr.enforce_retention(max_count=10)
        assert result["removed_count"] == 0
        assert result["remaining"] == 1

    def test_enforce_age_removes_old(self, backup_mgr, backup_dir):
        """Fake an old backup by setting mtime in the past."""
        import os
        import time

        result = backup_mgr.create_backup()
        old_path = backup_dir / result["filename"]
        # Set mtime to 100 days ago
        old_time = time.time() - (100 * 86400)
        os.utime(old_path, (old_time, old_time))

        # Create a fresh one
        backup_mgr.create_backup()

        result = backup_mgr.enforce_retention(max_count=100, max_age_days=7)
        assert result["removed_count"] == 1
        assert result["remaining"] == 1

    def test_enforce_retention_empty(self, backup_mgr):
        result = backup_mgr.enforce_retention(max_count=5)
        assert result["removed_count"] == 0
        assert result["remaining"] == 0


class TestBackupStats:
    def test_stats_empty(self, backup_mgr):
        stats = backup_mgr.get_stats()
        assert stats["total_count"] == 0
        assert stats["oldest"] is None
        assert stats["newest"] is None

    def test_stats_with_backups(self, backup_mgr):
        backup_mgr.create_backup()
        backup_mgr.create_backup()
        stats = backup_mgr.get_stats()
        assert stats["total_count"] == 2
        assert stats["total_size_bytes"] > 0
        assert stats["oldest"] is not None
        assert stats["newest"] is not None


class TestBackupEndpoints:
    @pytest.mark.asyncio
    async def test_create_backup_endpoint(self, client):
        resp = await client.post("/admin/backup")
        assert resp.status_code == 201
        body = resp.json()
        assert "backup_id" in body
        assert "filename" in body

    @pytest.mark.asyncio
    async def test_list_backups_endpoint(self, client):
        resp = await client.get("/admin/backups")
        assert resp.status_code == 200
        assert "backups" in resp.json()

    @pytest.mark.asyncio
    async def test_verify_backup_endpoint(self, client):
        # Create first
        create_resp = await client.post("/admin/backup")
        filename = create_resp.json()["filename"]
        # Verify
        resp = await client.post(f"/admin/backup/verify?filename={filename}")
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    @pytest.mark.asyncio
    async def test_verify_nonexistent_backup_endpoint(self, client):
        resp = await client.post("/admin/backup/verify?filename=nope.db")
        assert resp.status_code == 200
        assert resp.json()["valid"] is False

    @pytest.mark.asyncio
    async def test_cleanup_endpoint(self, client):
        # Create a couple of backups first
        await client.post("/admin/backup")
        await client.post("/admin/backup")
        resp = await client.post("/admin/backup/cleanup")
        assert resp.status_code == 200
        assert "removed_count" in resp.json()
        assert "remaining" in resp.json()

    @pytest.mark.asyncio
    async def test_stats_endpoint(self, client):
        resp = await client.get("/admin/backup/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_count" in body
        assert "total_size_bytes" in body
