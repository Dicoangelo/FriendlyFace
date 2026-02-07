"""Backup and recovery for FriendlyFace SQLite databases.

Uses SQLite's built-in .backup() for consistent, online backups.
Supports integrity verification, restore, and retention policy enforcement.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger("friendlyface.ops.backup")


class BackupManager:
    """Manages database backup, verification, restore, and retention."""

    def __init__(self, db_path: Path | str, backup_dir: Path | str) -> None:
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, label: str | None = None) -> dict:
        """Create a backup using SQLite online backup API.

        Returns metadata about the backup.
        """
        timestamp = datetime.now(timezone.utc)
        backup_id = str(uuid4())[:8]
        filename = f"ff_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}_{backup_id}.db"
        backup_path = self.backup_dir / filename

        # Use SQLite backup API for consistent snapshot
        source = sqlite3.connect(self.db_path)
        dest = sqlite3.connect(backup_path)
        try:
            source.backup(dest)
        finally:
            dest.close()
            source.close()

        size = backup_path.stat().st_size
        checksum = self._file_checksum(backup_path)

        logger.info("Backup created: %s (%d bytes, sha256=%s)", filename, size, checksum[:16])

        return {
            "backup_id": backup_id,
            "filename": filename,
            "path": str(backup_path),
            "timestamp": timestamp.isoformat(),
            "size_bytes": size,
            "checksum_sha256": checksum,
            "label": label,
        }

    def list_backups(self) -> list[dict]:
        """List all available backups sorted by modification time (newest first)."""
        backups = []
        for f in sorted(self.backup_dir.glob("ff_backup_*.db"), reverse=True):
            backups.append(
                {
                    "filename": f.name,
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        f.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        return backups

    def verify_backup(self, filename: str) -> dict:
        """Verify backup integrity using SQLite integrity_check."""
        backup_path = self.backup_dir / filename
        if not backup_path.exists():
            return {"valid": False, "error": f"Backup not found: {filename}"}

        try:
            conn = sqlite3.connect(backup_path)
            cursor = conn.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            conn.close()

            ok = result is not None and result[0] == "ok"
            checksum = self._file_checksum(backup_path)

            return {
                "filename": filename,
                "valid": ok,
                "integrity_result": result[0] if result else "unknown",
                "checksum_sha256": checksum,
                "size_bytes": backup_path.stat().st_size,
            }
        except sqlite3.DatabaseError as e:
            return {"filename": filename, "valid": False, "error": str(e)}

    def restore_backup(self, filename: str) -> dict:
        """Restore database from a backup file.

        The current database is replaced with the backup.
        """
        backup_path = self.backup_dir / filename
        if not backup_path.exists():
            return {"restored": False, "error": f"Backup not found: {filename}"}

        # Verify before restoring
        verification = self.verify_backup(filename)
        if not verification.get("valid"):
            return {"restored": False, "error": "Backup failed integrity check"}

        # Restore using backup API (reverse direction)
        source = sqlite3.connect(backup_path)
        dest = sqlite3.connect(self.db_path)
        try:
            source.backup(dest)
        finally:
            dest.close()
            source.close()

        logger.info("Database restored from %s", filename)

        return {
            "restored": True,
            "filename": filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def enforce_retention(
        self,
        max_count: int = 10,
        max_age_days: int | None = None,
    ) -> dict:
        """Remove old backups exceeding the retention policy.

        Args:
            max_count: Keep at most this many backups (newest first).
            max_age_days: Delete backups older than this many days (``None`` = no age limit).

        Returns:
            Summary of removed backups.
        """
        backups = sorted(
            self.backup_dir.glob("ff_backup_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,  # newest first
        )

        removed: list[str] = []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days) if max_age_days else None

        for idx, f in enumerate(backups):
            should_remove = False

            # Count-based: keep only max_count newest
            if idx >= max_count:
                should_remove = True

            # Age-based: remove if older than cutoff
            if cutoff is not None:
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    should_remove = True

            if should_remove:
                f.unlink()
                removed.append(f.name)
                logger.info("Retention policy removed backup: %s", f.name)

        return {
            "removed": removed,
            "removed_count": len(removed),
            "remaining": len(backups) - len(removed),
        }

    def get_stats(self) -> dict:
        """Return backup statistics."""
        backups = list(self.backup_dir.glob("ff_backup_*.db"))
        if not backups:
            return {
                "total_count": 0,
                "total_size_bytes": 0,
                "oldest": None,
                "newest": None,
            }

        sorted_backups = sorted(backups, key=lambda p: p.stat().st_mtime)
        total_size = sum(f.stat().st_size for f in backups)
        oldest_mtime = datetime.fromtimestamp(
            sorted_backups[0].stat().st_mtime, tz=timezone.utc
        ).isoformat()
        newest_mtime = datetime.fromtimestamp(
            sorted_backups[-1].stat().st_mtime, tz=timezone.utc
        ).isoformat()

        return {
            "total_count": len(backups),
            "total_size_bytes": total_size,
            "oldest": oldest_mtime,
            "newest": newest_mtime,
        }

    @staticmethod
    def _file_checksum(path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
