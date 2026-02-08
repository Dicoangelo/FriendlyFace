"""Tests for US-042: Migration Framework.

Verifies the lightweight SQL migration runner:
- Migration discovery from .sql files
- Idempotent application (skip already-applied)
- Tracking table creation
- Forward migration for all 7 migration files
- Migration status reporting
- Dry-run mode
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from friendlyface.storage.migrations import (
    apply_migrations,
    discover_migrations,
    get_applied_versions,
    get_migration_status,
    rollback_last,
)


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    """Create a temp directory with test migration files (up + down)."""
    d = tmp_path / "migrations"
    d.mkdir()
    (d / "001_baseline.sql").write_text(
        "CREATE TABLE IF NOT EXISTS test_table (id TEXT PRIMARY KEY, value TEXT);"
    )
    (d / "002_add_column.sql").write_text(
        "ALTER TABLE test_table ADD COLUMN extra TEXT DEFAULT 'none';"
    )
    (d / "002_add_column_down.sql").write_text(
        # SQLite doesn't support DROP COLUMN easily, so recreate table
        "CREATE TABLE IF NOT EXISTS test_table_backup AS SELECT id, value FROM test_table;"
        "DROP TABLE IF EXISTS test_table;"
        "ALTER TABLE test_table_backup RENAME TO test_table;"
    )
    (d / "003_add_index.sql").write_text(
        "CREATE INDEX IF NOT EXISTS idx_test_value ON test_table (value);"
    )
    (d / "003_add_index_down.sql").write_text("DROP INDEX IF EXISTS idx_test_value;")
    return d


@pytest.fixture
async def mem_db(tmp_path: Path):
    """In-memory SQLite connection for migration testing."""
    db = await aiosqlite.connect(tmp_path / "mig_test.db")
    yield db
    await db.close()


class TestDiscoverMigrations:
    def test_discover_from_directory(self, migrations_dir: Path):
        migs = discover_migrations(migrations_dir)
        assert len(migs) == 3
        assert migs[0][0] == "001"
        assert migs[0][1] == "baseline"
        assert migs[1][0] == "002"
        assert migs[2][0] == "003"

    def test_discover_sorted_order(self, migrations_dir: Path):
        versions = [v for v, _, _ in discover_migrations(migrations_dir)]
        assert versions == sorted(versions)

    def test_discover_empty_directory(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        assert discover_migrations(d) == []

    def test_discover_nonexistent_directory(self, tmp_path: Path):
        assert discover_migrations(tmp_path / "nope") == []

    def test_discover_real_migrations(self):
        """Verify that the actual migrations/ directory has expected files."""
        migs = discover_migrations()
        versions = [v for v, _, _ in migs]
        assert "002" in versions
        assert "007" in versions


class TestApplyMigrations:
    @pytest.mark.asyncio
    async def test_apply_all_migrations(self, mem_db, migrations_dir: Path):
        applied = await apply_migrations(mem_db, migrations_dir)
        assert applied == ["001", "002", "003"]

    @pytest.mark.asyncio
    async def test_idempotent_application(self, mem_db, migrations_dir: Path):
        first = await apply_migrations(mem_db, migrations_dir)
        assert len(first) == 3
        second = await apply_migrations(mem_db, migrations_dir)
        assert len(second) == 0

    @pytest.mark.asyncio
    async def test_incremental_application(self, mem_db, tmp_path: Path):
        d = tmp_path / "inc_migs"
        d.mkdir()
        (d / "001_first.sql").write_text(
            "CREATE TABLE IF NOT EXISTS inc_test (id TEXT PRIMARY KEY);"
        )
        applied = await apply_migrations(mem_db, d)
        assert applied == ["001"]

        # Add a second migration
        (d / "002_second.sql").write_text("ALTER TABLE inc_test ADD COLUMN name TEXT;")
        applied = await apply_migrations(mem_db, d)
        assert applied == ["002"]

    @pytest.mark.asyncio
    async def test_tracking_table_created(self, mem_db, migrations_dir: Path):
        await apply_migrations(mem_db, migrations_dir)
        cursor = await mem_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_applied_versions_tracked(self, mem_db, migrations_dir: Path):
        await apply_migrations(mem_db, migrations_dir)
        versions = await get_applied_versions(mem_db)
        assert versions == {"001", "002", "003"}

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, mem_db, migrations_dir: Path):
        applied = await apply_migrations(mem_db, migrations_dir, dry_run=True)
        assert applied == ["001", "002", "003"]
        # Nothing actually applied
        versions = await get_applied_versions(mem_db)
        assert len(versions) == 0


class TestMigrationStatus:
    @pytest.mark.asyncio
    async def test_status_all_pending(self, mem_db, migrations_dir: Path):
        status = await get_migration_status(mem_db)
        # Uses default MIGRATIONS_DIR, so total reflects real migrations
        assert "applied" in status
        assert "pending" in status
        assert "total" in status

    @pytest.mark.asyncio
    async def test_status_after_apply(self, mem_db, migrations_dir: Path):
        await apply_migrations(mem_db, migrations_dir)
        versions = await get_applied_versions(mem_db)
        assert "001" in versions
        assert "002" in versions
        assert "003" in versions


class TestRealMigrations:
    """Test that the actual project migration files are valid SQL."""

    @pytest.mark.asyncio
    async def test_apply_real_migrations(self, tmp_path: Path):
        """Apply all real migration files to a fresh database."""
        db = await aiosqlite.connect(tmp_path / "real_mig.db")
        try:
            applied = await apply_migrations(db)
            assert len(applied) >= 6
            # Verify key tables exist
            for table in [
                "audit_log",
                "subject_keys",
                "erasure_records",
                "retention_policies",
                "did_keys",
                "merkle_checkpoints",
                "face_gallery",
            ]:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                row = await cursor.fetchone()
                assert row is not None, f"Table {table} not created"
        finally:
            await db.close()


class TestRollback:
    """Test rollback_last() for migration rollback support (US-079)."""

    @pytest.mark.asyncio
    async def test_rollback_last_migration(self, mem_db, migrations_dir: Path):
        await apply_migrations(mem_db, migrations_dir)
        versions = await get_applied_versions(mem_db)
        assert "003" in versions

        result = await rollback_last(mem_db, migrations_dir)
        assert result["rolled_back"] == "003"
        assert result["dry_run"] is False

        versions = await get_applied_versions(mem_db)
        assert "003" not in versions
        assert "002" in versions

    @pytest.mark.asyncio
    async def test_rollback_dry_run(self, mem_db, migrations_dir: Path):
        await apply_migrations(mem_db, migrations_dir)

        result = await rollback_last(mem_db, migrations_dir, dry_run=True)
        assert result["rolled_back"] == "003"
        assert result["dry_run"] is True

        # Nothing actually rolled back
        versions = await get_applied_versions(mem_db)
        assert "003" in versions

    @pytest.mark.asyncio
    async def test_rollback_no_migrations(self, mem_db):
        result = await rollback_last(mem_db)
        assert "error" in result
        assert "No migrations" in result["error"]

    @pytest.mark.asyncio
    async def test_rollback_no_down_file(self, mem_db, tmp_path: Path):
        """Rollback fails if no _down.sql file exists."""
        d = tmp_path / "no_down_migs"
        d.mkdir()
        (d / "001_only_up.sql").write_text(
            "CREATE TABLE IF NOT EXISTS up_only (id TEXT PRIMARY KEY);"
        )
        await apply_migrations(mem_db, d)

        result = await rollback_last(mem_db, d)
        assert "error" in result
        assert "No down migration" in result["error"]

    @pytest.mark.asyncio
    async def test_rollback_successive(self, mem_db, migrations_dir: Path):
        """Roll back multiple migrations one at a time."""
        await apply_migrations(mem_db, migrations_dir)
        assert len(await get_applied_versions(mem_db)) == 3

        await rollback_last(mem_db, migrations_dir)
        assert len(await get_applied_versions(mem_db)) == 2

        await rollback_last(mem_db, migrations_dir)
        assert len(await get_applied_versions(mem_db)) == 1

    @pytest.mark.asyncio
    async def test_discover_skips_down_files(self, migrations_dir: Path):
        migs = discover_migrations(migrations_dir)
        names = [n for _, n, _ in migs]
        assert not any("down" in n for n in names)

    @pytest.mark.asyncio
    async def test_rollback_real_migrations(self, tmp_path: Path):
        """Apply and rollback real project migrations."""
        db = await aiosqlite.connect(tmp_path / "rollback_real.db")
        try:
            await apply_migrations(db)
            versions_before = await get_applied_versions(db)
            latest = max(versions_before)
            assert int(latest) >= 7  # at least the original migrations

            result = await rollback_last(db)
            assert result["rolled_back"] == latest
        finally:
            await db.close()


class TestDatabaseRunMigrations:
    """Test the Database.run_migrations() integration."""

    @pytest.mark.asyncio
    async def test_run_migrations_via_database(self, db):
        applied = await db.run_migrations()
        assert isinstance(applied, list)
