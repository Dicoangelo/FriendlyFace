"""Lightweight SQL migration runner for FriendlyFace.

Tracks applied migrations in a `_migrations` table. Each migration is a
numbered .sql file in the `migrations/` directory. Migrations run in order
and are idempotent (skipped if already applied).

Rollback is supported via optional ``NNN_name_down.sql`` companion files.

Usage:
    from friendlyface.storage.migrations import apply_migrations
    await apply_migrations(db_connection)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger("friendlyface.migrations")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"

_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS _migrations (
    version TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
"""


async def get_applied_versions(db: aiosqlite.Connection) -> set[str]:
    """Return the set of already-applied migration versions."""
    await db.executescript(_TRACKING_TABLE_SQL)
    cursor = await db.execute("SELECT version FROM _migrations ORDER BY version")
    rows = await cursor.fetchall()
    return {row[0] for row in rows}


def discover_migrations(directory: Path | None = None) -> list[tuple[str, str, str]]:
    """Discover migration files sorted by version number.

    Returns list of (version, name, sql_content) tuples.
    Files must be named like ``002_audit_log.sql`` (``_down.sql`` files are excluded).
    """
    d = directory or MIGRATIONS_DIR
    if not d.is_dir():
        return []

    migrations: list[tuple[str, str, str]] = []
    for f in sorted(d.glob("*.sql")):
        # Skip down-migration files
        if f.stem.endswith("_down"):
            continue
        parts = f.stem.split("_", 1)
        if len(parts) < 2:
            continue
        version = parts[0]  # e.g. "002"
        name = parts[1]  # e.g. "audit_log"
        sql = f.read_text(encoding="utf-8")
        migrations.append((version, name, sql))
    return migrations


def _find_down_migration(version: str, name: str, directory: Path | None = None) -> str | None:
    """Find the down-migration SQL for a given version.

    Looks for ``NNN_name_down.sql`` in the migrations directory.
    """
    d = directory or MIGRATIONS_DIR
    down_file = d / f"{version}_{name}_down.sql"
    if down_file.is_file():
        return down_file.read_text(encoding="utf-8")
    return None


async def apply_migrations(
    db: aiosqlite.Connection,
    directory: Path | None = None,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Apply pending migrations in order.

    Returns list of newly applied migration versions.
    """
    applied = await get_applied_versions(db)
    all_migrations = discover_migrations(directory)

    newly_applied: list[str] = []
    for version, name, sql in all_migrations:
        if version in applied:
            logger.debug("Migration %s (%s) already applied, skipping", version, name)
            continue

        if dry_run:
            logger.info("Would apply migration %s: %s", version, name)
            newly_applied.append(version)
            continue

        logger.info("Applying migration %s: %s", version, name)
        await db.executescript(sql)
        await db.execute(
            "INSERT INTO _migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (version, name, datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()
        newly_applied.append(version)
        logger.info("Migration %s applied successfully", version)

    if not newly_applied:
        logger.info("All migrations up to date")

    return newly_applied


async def rollback_last(
    db: aiosqlite.Connection,
    directory: Path | None = None,
    *,
    dry_run: bool = False,
) -> dict:
    """Roll back the most recently applied migration.

    Requires a corresponding ``NNN_name_down.sql`` file.

    Returns:
        ``{"rolled_back": "NNN", "name": "...", "dry_run": bool}`` on success,
        or ``{"error": "..."}`` if rollback is not possible.
    """
    applied = await get_applied_versions(db)
    if not applied:
        return {"error": "No migrations to roll back"}

    latest_version = max(applied)

    # Look up the name from the tracking table
    cursor = await db.execute("SELECT name FROM _migrations WHERE version = ?", (latest_version,))
    row = await cursor.fetchone()
    if row is None:
        return {"error": f"Migration {latest_version} not found in tracking table"}
    migration_name = row[0]

    down_sql = _find_down_migration(latest_version, migration_name, directory)
    if down_sql is None:
        return {"error": f"No down migration found for {latest_version}_{migration_name}"}

    if dry_run:
        return {
            "rolled_back": latest_version,
            "name": migration_name,
            "dry_run": True,
        }

    logger.info("Rolling back migration %s: %s", latest_version, migration_name)
    await db.executescript(down_sql)
    await db.execute("DELETE FROM _migrations WHERE version = ?", (latest_version,))
    await db.commit()
    logger.info("Migration %s rolled back successfully", latest_version)

    return {
        "rolled_back": latest_version,
        "name": migration_name,
        "dry_run": False,
    }


async def get_migration_status(db: aiosqlite.Connection) -> dict:
    """Return migration status information."""
    applied = await get_applied_versions(db)
    all_migrations = discover_migrations()
    pending = [v for v, _, _ in all_migrations if v not in applied]
    return {
        "applied": sorted(applied),
        "pending": pending,
        "total": len(all_migrations),
    }
