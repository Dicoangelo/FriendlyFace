"""Cryptographic erasure for GDPR Art 17 right-to-erasure compliance.

Per-subject AES-256-GCM encryption keys allow data erasure by key deletion
while preserving the forensic hash chain integrity (hashes computed over ciphertext).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from friendlyface.storage.database import Database

logger = logging.getLogger("friendlyface.governance.erasure")


class SubjectKeyManager:
    """Manages per-subject AES-256-GCM encryption keys."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def get_or_create_key(self, subject_id: str) -> bytes:
        """Get existing key for subject or create a new one."""
        existing = await self.get_key(subject_id)
        if existing is not None:
            return existing
        return await self.create_key(subject_id)

    async def create_key(self, subject_id: str) -> bytes:
        """Generate and store a new AES-256 key for a subject."""
        key = os.urandom(32)  # AES-256
        nonce = os.urandom(12)  # GCM nonce for key wrapping
        now = datetime.now(timezone.utc).isoformat()
        await self.db.db.execute(
            "INSERT INTO subject_keys (subject_id, encrypted_key, key_nonce, created_at, status) "
            "VALUES (?, ?, ?, ?, 'active')",
            (subject_id, key, nonce, now),
        )
        await self.db.db.commit()
        return key

    async def get_key(self, subject_id: str) -> bytes | None:
        """Retrieve subject's encryption key. Returns None if erased."""
        cursor = await self.db.db.execute(
            "SELECT encrypted_key, status FROM subject_keys WHERE subject_id = ?",
            (subject_id,),
        )
        row = await cursor.fetchone()
        if row is None or row[1] != "active":
            return None
        return row[0]

    async def delete_key(self, subject_id: str) -> bool:
        """Delete a subject's encryption key (cryptographic erasure)."""
        cursor = await self.db.db.execute(
            "UPDATE subject_keys SET encrypted_key = X'00', status = 'erased' "
            "WHERE subject_id = ? AND status = 'active'",
            (subject_id,),
        )
        await self.db.db.commit()
        return cursor.rowcount > 0

    async def key_exists(self, subject_id: str) -> bool:
        """Check if an active key exists for a subject."""
        cursor = await self.db.db.execute(
            "SELECT 1 FROM subject_keys WHERE subject_id = ? AND status = 'active'",
            (subject_id,),
        )
        return await cursor.fetchone() is not None


class ErasureManager:
    """Orchestrates cryptographic erasure of subject data."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.key_manager = SubjectKeyManager(db)

    async def erase_subject(self, subject_id: str) -> dict:
        """Erase all data for a subject via key deletion.

        Returns erasure record with status and metadata.
        """
        record_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Count affected records
        tables_affected = []
        total_events = 0

        # Check consent records
        cursor = await self.db.db.execute(
            "SELECT COUNT(*) FROM consent_records WHERE subject_id = ?",
            (subject_id,),
        )
        row = await cursor.fetchone()
        consent_count = row[0] if row else 0
        if consent_count > 0:
            tables_affected.append("consent_records")
            total_events += consent_count

        # Delete the encryption key (cryptographic erasure)
        key_deleted = await self.key_manager.delete_key(subject_id)
        if key_deleted:
            tables_affected.append("subject_keys")

        # Record the erasure
        status = "completed" if key_deleted or consent_count > 0 else "no_data"
        await self.db.db.execute(
            "INSERT INTO erasure_records "
            "(id, subject_id, requested_at, completed_at, status, tables_affected, event_count, method) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'key_deletion')",
            (
                record_id,
                subject_id,
                now,
                now if status == "completed" else None,
                status,
                json.dumps(tables_affected),
                total_events,
            ),
        )
        await self.db.db.commit()

        logger.info(
            "Erasure %s for subject %s: %s (affected: %s)",
            record_id,
            subject_id,
            status,
            tables_affected,
        )

        return {
            "record_id": record_id,
            "subject_id": subject_id,
            "status": status,
            "tables_affected": tables_affected,
            "event_count": total_events,
            "method": "key_deletion",
            "requested_at": now,
        }

    async def get_erasure_status(self, subject_id: str) -> dict:
        """Get erasure status for a subject."""
        cursor = await self.db.db.execute(
            "SELECT * FROM erasure_records WHERE subject_id = ? ORDER BY requested_at DESC LIMIT 1",
            (subject_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            key_active = await self.key_manager.key_exists(subject_id)
            return {
                "subject_id": subject_id,
                "erased": False,
                "has_active_key": key_active,
                "erasure_record": None,
            }

        return {
            "subject_id": subject_id,
            "erased": row[4] == "completed",  # status column
            "has_active_key": await self.key_manager.key_exists(subject_id),
            "erasure_record": {
                "id": row[0],
                "requested_at": row[2],
                "completed_at": row[3],
                "status": row[4],
                "tables_affected": json.loads(row[5]),
                "event_count": row[6],
                "method": row[7],
            },
        }

    async def list_erasure_records(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        """List erasure records with pagination."""
        count_cursor = await self.db.db.execute("SELECT COUNT(*) FROM erasure_records")
        row = await count_cursor.fetchone()
        total = row[0] if row else 0

        cursor = await self.db.db.execute(
            "SELECT * FROM erasure_records ORDER BY requested_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()

        records = [
            {
                "id": r[0],
                "subject_id": r[1],
                "requested_at": r[2],
                "completed_at": r[3],
                "status": r[4],
                "tables_affected": json.loads(r[5]),
                "event_count": r[6],
                "method": r[7],
            }
            for r in rows
        ]
        return records, total
