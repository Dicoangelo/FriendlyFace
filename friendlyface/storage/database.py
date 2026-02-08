"""Async SQLite storage layer for the Blockchain Forensic Layer.

Uses aiosqlite for async access. Repository pattern for clean separation.
Designed for dev/testing — production would use Supabase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite

from friendlyface.core.models import (
    BiasAuditRecord,
    BundleStatus,
    EventType,
    ForensicBundle,
    ForensicEvent,
    MerkleProof,
    ProvenanceNode,
    ProvenanceRelation,
)

import os

DEFAULT_DB_PATH = Path(os.environ.get("FF_DB_PATH", "friendlyface.db"))

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS forensic_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    previous_hash TEXT NOT NULL DEFAULT 'GENESIS',
    event_hash TEXT NOT NULL,
    sequence_number INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS provenance_nodes (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    parents TEXT NOT NULL DEFAULT '[]',
    relations TEXT NOT NULL DEFAULT '[]',
    node_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS forensic_bundles (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    event_ids TEXT NOT NULL DEFAULT '[]',
    merkle_root TEXT NOT NULL DEFAULT '',
    merkle_proofs TEXT NOT NULL DEFAULT '[]',
    provenance_chain TEXT NOT NULL DEFAULT '[]',
    bias_audit TEXT,
    recognition_artifacts TEXT,
    fl_artifacts TEXT,
    bias_report TEXT,
    explanation_artifacts TEXT,
    zk_proof_placeholder TEXT,
    did_credential_placeholder TEXT,
    bundle_hash TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS bias_audits (
    id TEXT PRIMARY KEY,
    event_id TEXT,
    timestamp TEXT NOT NULL,
    demographic_parity_gap REAL NOT NULL,
    equalized_odds_gap REAL NOT NULL,
    groups_evaluated TEXT NOT NULL DEFAULT '[]',
    compliant INTEGER NOT NULL DEFAULT 1,
    details TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS consent_records (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    granted INTEGER NOT NULL DEFAULT 1,
    timestamp TEXT NOT NULL,
    expiry TEXT,
    revocation_reason TEXT,
    event_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_consent_subject_purpose
    ON consent_records (subject_id, purpose, timestamp);

CREATE INDEX IF NOT EXISTS idx_events_event_type
    ON forensic_events (event_type);

CREATE INDEX IF NOT EXISTS idx_events_timestamp
    ON forensic_events (timestamp);

CREATE INDEX IF NOT EXISTS idx_events_sequence_number
    ON forensic_events (sequence_number);

CREATE INDEX IF NOT EXISTS idx_bundles_status
    ON forensic_bundles (status);

CREATE INDEX IF NOT EXISTS idx_bundles_created_at
    ON forensic_bundles (created_at);

CREATE INDEX IF NOT EXISTS idx_provenance_entity
    ON provenance_nodes (entity_type, entity_id);

CREATE INDEX IF NOT EXISTS idx_bias_audits_timestamp
    ON bias_audits (timestamp);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    details TEXT NOT NULL DEFAULT '{}',
    ip_address TEXT,
    user_agent TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp
    ON audit_log (timestamp);

CREATE INDEX IF NOT EXISTS idx_audit_log_actor
    ON audit_log (actor);

CREATE INDEX IF NOT EXISTS idx_audit_log_action
    ON audit_log (action);

CREATE TABLE IF NOT EXISTS subject_keys (
    subject_id TEXT PRIMARY KEY,
    encrypted_key BLOB NOT NULL,
    key_nonce BLOB NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS erasure_records (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    tables_affected TEXT NOT NULL DEFAULT '[]',
    event_count INTEGER NOT NULL DEFAULT 0,
    method TEXT NOT NULL DEFAULT 'key_deletion'
);

CREATE INDEX IF NOT EXISTS idx_erasure_records_subject
    ON erasure_records (subject_id);

CREATE INDEX IF NOT EXISTS idx_erasure_records_status
    ON erasure_records (status);

CREATE TABLE IF NOT EXISTS retention_policies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    retention_days INTEGER NOT NULL,
    action TEXT NOT NULL DEFAULT 'erase',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retention_policies_entity
    ON retention_policies (entity_type);

CREATE TABLE IF NOT EXISTS did_keys (
    did TEXT PRIMARY KEY,
    public_key BLOB NOT NULL,
    encrypted_private_key BLOB,
    key_type TEXT NOT NULL DEFAULT 'Ed25519',
    created_at TEXT NOT NULL,
    label TEXT,
    is_platform_key INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_did_keys_platform
    ON did_keys (is_platform_key);

CREATE TABLE IF NOT EXISTS merkle_checkpoints (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    leaf_count INTEGER NOT NULL,
    root_hash TEXT NOT NULL,
    leaves_json TEXT NOT NULL,
    event_index_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_merkle_checkpoints_leaf_count
    ON merkle_checkpoints (leaf_count DESC);

CREATE TABLE IF NOT EXISTS face_gallery (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL DEFAULT 512,
    model_version TEXT NOT NULL,
    quality_score REAL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_face_gallery_subject
    ON face_gallery (subject_id);

CREATE INDEX IF NOT EXISTS idx_face_gallery_model
    ON face_gallery (model_version);
"""


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self, db_key: str | None = None, require_encryption: bool = False) -> None:
        if require_encryption and not db_key:
            raise RuntimeError(
                "FF_REQUIRE_ENCRYPTION is set but FF_DB_KEY is not configured. "
                "Provide an encryption key or disable the requirement."
            )

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        # Encryption at rest — activate SQLCipher PRAGMA key if provided.
        # PRAGMA does not support parameterized queries; db_key comes from
        # server config (FF_DB_KEY env var), not user input.
        if db_key:
            escaped = db_key.replace("'", "''")
            await self._db.execute(f"PRAGMA key = '{escaped}'")

        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()

    async def run_migrations(self) -> list[str]:
        """Apply pending SQL migrations. Returns list of applied versions."""
        from friendlyface.storage.migrations import apply_migrations

        return await apply_migrations(self.db)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    # --- ForensicEvent ---

    async def insert_event(self, event: ForensicEvent) -> None:
        await self.db.execute(
            """INSERT INTO forensic_events
               (id, event_type, timestamp, actor, payload, previous_hash, event_hash, sequence_number)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(event.id),
                event.event_type.value,
                event.timestamp.isoformat(),
                event.actor,
                json.dumps(event.payload),
                event.previous_hash,
                event.event_hash,
                event.sequence_number,
            ),
        )
        await self.db.commit()

    async def get_event(self, event_id: UUID) -> ForensicEvent | None:
        cursor = await self.db.execute(
            "SELECT * FROM forensic_events WHERE id = ?", (str(event_id),)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    async def get_latest_event(self) -> ForensicEvent | None:
        cursor = await self.db.execute(
            "SELECT * FROM forensic_events ORDER BY sequence_number DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    async def get_all_events(self) -> list[ForensicEvent]:
        cursor = await self.db.execute("SELECT * FROM forensic_events ORDER BY sequence_number ASC")
        rows = await cursor.fetchall()
        return [self._row_to_event(r) for r in rows]

    async def get_events_paginated(self, limit: int = 50, offset: int = 0) -> list[ForensicEvent]:
        """Return a page of events ordered by sequence number."""
        cursor = await self.db.execute(
            "SELECT * FROM forensic_events ORDER BY sequence_number ASC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_event(r) for r in rows]

    async def get_events_filtered(
        self,
        limit: int = 50,
        offset: int = 0,
        event_type: str | None = None,
        actor: str | None = None,
    ) -> tuple[list[ForensicEvent], int]:
        """Return filtered and paginated events with total count."""
        conditions = []
        params: list[Any] = []
        if event_type is not None:
            conditions.append("event_type = ?")
            params.append(event_type)
        if actor is not None:
            conditions.append("actor = ?")
            params.append(actor)

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        count_cursor = await self.db.execute(
            f"SELECT COUNT(*) FROM forensic_events{where}",
            params,  # noqa: S608
        )
        row = await count_cursor.fetchone()
        total = row[0] if row else 0

        params.extend([limit, offset])
        cursor = await self.db.execute(
            f"SELECT * FROM forensic_events{where} ORDER BY sequence_number ASC LIMIT ? OFFSET ?",  # noqa: S608
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_event(r) for r in rows], total

    async def get_events_by_ids(self, event_ids: list[UUID]) -> list[ForensicEvent]:
        """Batch-fetch events by a list of IDs (avoids N+1 queries)."""
        if not event_ids:
            return []
        placeholders = ",".join("?" for _ in event_ids)
        cursor = await self.db.execute(
            f"SELECT * FROM forensic_events WHERE id IN ({placeholders}) "  # noqa: S608  # nosec B608
            "ORDER BY sequence_number ASC",
            [str(eid) for eid in event_ids],
        )
        rows = await cursor.fetchall()
        return [self._row_to_event(r) for r in rows]

    async def get_event_count(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM forensic_events")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_bundle_count(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM forensic_bundles")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_provenance_count(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM provenance_nodes")
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_events_by_type(self) -> dict[str, int]:
        cursor = await self.db.execute(
            "SELECT event_type, COUNT(*) as cnt FROM forensic_events GROUP BY event_type"
        )
        rows = await cursor.fetchall()
        return {row["event_type"]: row["cnt"] for row in rows}

    async def get_recent_events(self, limit: int = 10) -> list[dict[str, Any]]:
        cursor = await self.db.execute(
            "SELECT id, event_type, actor, timestamp FROM forensic_events "
            "ORDER BY sequence_number DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "event_type": row["event_type"],
                "actor": row["actor"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    def _row_to_event(self, row: aiosqlite.Row) -> ForensicEvent:
        return ForensicEvent(
            id=UUID(row["id"]),
            event_type=EventType(row["event_type"]),
            timestamp=row["timestamp"],
            actor=row["actor"],
            payload=json.loads(row["payload"]),
            previous_hash=row["previous_hash"],
            event_hash=row["event_hash"],
            sequence_number=row["sequence_number"],
        )

    # --- ProvenanceNode ---

    async def insert_provenance_node(self, node: ProvenanceNode) -> None:
        await self.db.execute(
            """INSERT INTO provenance_nodes
               (id, entity_type, entity_id, created_at, metadata, parents, relations, node_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(node.id),
                node.entity_type,
                node.entity_id,
                node.created_at.isoformat(),
                json.dumps(node.metadata),
                json.dumps([str(p) for p in node.parents]),
                json.dumps([r.value for r in node.relations]),
                node.node_hash,
            ),
        )
        await self.db.commit()

    async def get_provenance_node(self, node_id: UUID) -> ProvenanceNode | None:
        cursor = await self.db.execute(
            "SELECT * FROM provenance_nodes WHERE id = ?", (str(node_id),)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_provenance(row)

    def _row_to_provenance(self, row: aiosqlite.Row) -> ProvenanceNode:
        return ProvenanceNode(
            id=UUID(row["id"]),
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
            parents=[UUID(p) for p in json.loads(row["parents"])],
            relations=[ProvenanceRelation(r) for r in json.loads(row["relations"])],
            node_hash=row["node_hash"],
        )

    # --- ForensicBundle ---

    async def insert_bundle(self, bundle: ForensicBundle) -> None:
        await self.db.execute(
            """INSERT INTO forensic_bundles
               (id, created_at, status, event_ids, merkle_root, merkle_proofs,
                provenance_chain, bias_audit, recognition_artifacts, fl_artifacts,
                bias_report, explanation_artifacts, zk_proof_placeholder,
                did_credential_placeholder, bundle_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(bundle.id),
                bundle.created_at.isoformat(),
                bundle.status.value,
                json.dumps([str(e) for e in bundle.event_ids]),
                bundle.merkle_root,
                json.dumps([p.model_dump() for p in bundle.merkle_proofs]),
                json.dumps([str(p) for p in bundle.provenance_chain]),
                bundle.bias_audit.model_dump_json() if bundle.bias_audit else None,
                json.dumps(bundle.recognition_artifacts) if bundle.recognition_artifacts else None,
                json.dumps(bundle.fl_artifacts) if bundle.fl_artifacts else None,
                json.dumps(bundle.bias_report) if bundle.bias_report else None,
                json.dumps(bundle.explanation_artifacts) if bundle.explanation_artifacts else None,
                bundle.zk_proof_placeholder,
                bundle.did_credential_placeholder,
                bundle.bundle_hash,
            ),
        )
        await self.db.commit()

    async def get_bundle(self, bundle_id: UUID) -> ForensicBundle | None:
        cursor = await self.db.execute(
            "SELECT * FROM forensic_bundles WHERE id = ?", (str(bundle_id),)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_bundle(row)

    async def update_bundle_status(self, bundle_id: UUID, status: BundleStatus) -> None:
        await self.db.execute(
            "UPDATE forensic_bundles SET status = ? WHERE id = ?",
            (status.value, str(bundle_id)),
        )
        await self.db.commit()

    def _row_to_bundle(self, row: aiosqlite.Row) -> ForensicBundle:
        bias = None
        if row["bias_audit"]:
            bias = BiasAuditRecord.model_validate_json(row["bias_audit"])

        proofs = [MerkleProof.model_validate(p) for p in json.loads(row["merkle_proofs"])]

        recognition = (
            json.loads(row["recognition_artifacts"]) if row["recognition_artifacts"] else None
        )
        fl = json.loads(row["fl_artifacts"]) if row["fl_artifacts"] else None
        bias_rpt = json.loads(row["bias_report"]) if row["bias_report"] else None
        explanation = (
            json.loads(row["explanation_artifacts"]) if row["explanation_artifacts"] else None
        )

        return ForensicBundle(
            id=UUID(row["id"]),
            created_at=row["created_at"],
            status=BundleStatus(row["status"]),
            event_ids=[UUID(e) for e in json.loads(row["event_ids"])],
            merkle_root=row["merkle_root"],
            merkle_proofs=proofs,
            provenance_chain=[UUID(p) for p in json.loads(row["provenance_chain"])],
            bias_audit=bias,
            recognition_artifacts=recognition,
            fl_artifacts=fl,
            bias_report=bias_rpt,
            explanation_artifacts=explanation,
            zk_proof_placeholder=row["zk_proof_placeholder"],
            did_credential_placeholder=row["did_credential_placeholder"],
            bundle_hash=row["bundle_hash"],
        )

    # --- BiasAudit ---

    async def insert_bias_audit(self, audit: BiasAuditRecord) -> None:
        await self.db.execute(
            """INSERT INTO bias_audits
               (id, event_id, timestamp, demographic_parity_gap, equalized_odds_gap,
                groups_evaluated, compliant, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(audit.id),
                str(audit.event_id) if audit.event_id else None,
                audit.timestamp.isoformat(),
                audit.demographic_parity_gap,
                audit.equalized_odds_gap,
                json.dumps(audit.groups_evaluated),
                int(audit.compliant),
                json.dumps(audit.details),
            ),
        )
        await self.db.commit()

    async def get_bias_audit(self, audit_id: UUID) -> BiasAuditRecord | None:
        cursor = await self.db.execute("SELECT * FROM bias_audits WHERE id = ?", (str(audit_id),))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_bias_audit(row)

    async def get_all_bias_audits(self) -> list[BiasAuditRecord]:
        cursor = await self.db.execute("SELECT * FROM bias_audits ORDER BY timestamp ASC")
        rows = await cursor.fetchall()
        return [self._row_to_bias_audit(r) for r in rows]

    async def get_bias_audits_paginated(
        self, limit: int = 50, offset: int = 0
    ) -> list[BiasAuditRecord]:
        """Return a page of bias audits ordered by timestamp."""
        cursor = await self.db.execute(
            "SELECT * FROM bias_audits ORDER BY timestamp ASC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_bias_audit(r) for r in rows]

    async def get_bias_audit_count(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM bias_audits")
        row = await cursor.fetchone()
        return row[0] if row else 0

    def _row_to_bias_audit(self, row: aiosqlite.Row) -> BiasAuditRecord:
        return BiasAuditRecord(
            id=UUID(row["id"]),
            event_id=UUID(row["event_id"]) if row["event_id"] else None,
            timestamp=row["timestamp"],
            demographic_parity_gap=row["demographic_parity_gap"],
            equalized_odds_gap=row["equalized_odds_gap"],
            groups_evaluated=json.loads(row["groups_evaluated"]),
            compliant=bool(row["compliant"]),
            details=json.loads(row["details"]),
        )

    # --- Consent Records ---

    async def insert_consent_record(
        self,
        record_id: str,
        subject_id: str,
        purpose: str,
        granted: bool,
        timestamp: str,
        expiry: str | None = None,
        revocation_reason: str | None = None,
        event_id: str | None = None,
    ) -> None:
        """Append a consent record. Never updates or deletes existing records."""
        await self.db.execute(
            """INSERT INTO consent_records
               (id, subject_id, purpose, granted, timestamp, expiry, revocation_reason, event_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record_id,
                subject_id,
                purpose,
                int(granted),
                timestamp,
                expiry,
                revocation_reason,
                event_id,
            ),
        )
        await self.db.commit()

    async def get_latest_consent(self, subject_id: str, purpose: str) -> dict[str, Any] | None:
        """Get the most recent consent record for a subject+purpose pair."""
        cursor = await self.db.execute(
            """SELECT * FROM consent_records
               WHERE subject_id = ? AND purpose = ?
               ORDER BY timestamp DESC LIMIT 1""",
            (subject_id, purpose),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_consent(row)

    async def get_consent_history(
        self, subject_id: str, purpose: str | None = None
    ) -> list[dict[str, Any]]:
        """Get full consent history for a subject, optionally filtered by purpose."""
        if purpose is not None:
            cursor = await self.db.execute(
                """SELECT * FROM consent_records
                   WHERE subject_id = ? AND purpose = ?
                   ORDER BY timestamp ASC""",
                (subject_id, purpose),
            )
        else:
            cursor = await self.db.execute(
                """SELECT * FROM consent_records
                   WHERE subject_id = ?
                   ORDER BY timestamp ASC""",
                (subject_id,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_consent(r) for r in rows]

    def _row_to_consent(self, row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "subject_id": row["subject_id"],
            "purpose": row["purpose"],
            "granted": bool(row["granted"]),
            "timestamp": row["timestamp"],
            "expiry": row["expiry"],
            "revocation_reason": row["revocation_reason"],
            "event_id": row["event_id"],
        }

    # --- Compliance query helpers ---

    async def get_consent_coverage_stats(self) -> dict[str, Any]:
        """Get consent coverage statistics for compliance reporting.

        Returns dict with total_subjects, subjects_with_active_consent,
        and coverage_pct.
        """
        cursor = await self.db.execute("SELECT COUNT(DISTINCT subject_id) FROM consent_records")
        row = await cursor.fetchone()
        total_subjects = row[0] if row else 0

        if total_subjects == 0:
            return {
                "total_subjects": 0,
                "subjects_with_active_consent": 0,
                "coverage_pct": 0.0,
            }

        # For each subject, check if they have at least one active consent
        # (latest record per subject+purpose is granted=1)
        cursor = await self.db.execute(
            "SELECT COUNT(DISTINCT subject_id) FROM ("
            "  SELECT subject_id, purpose, granted,"
            "         ROW_NUMBER() OVER ("
            "             PARTITION BY subject_id, purpose"
            "             ORDER BY timestamp DESC"
            "         ) as rn"
            "  FROM consent_records"
            ") WHERE rn = 1 AND granted = 1"
        )
        row = await cursor.fetchone()
        active = row[0] if row else 0

        coverage = (active / total_subjects * 100.0) if total_subjects > 0 else 0.0
        return {
            "total_subjects": total_subjects,
            "subjects_with_active_consent": active,
            "coverage_pct": round(coverage, 2),
        }

    async def get_bias_audit_stats(self) -> dict[str, Any]:
        """Get bias audit statistics for compliance reporting."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM bias_audits")
        row = await cursor.fetchone()
        total = row[0] if row else 0

        cursor = await self.db.execute("SELECT COUNT(*) FROM bias_audits WHERE compliant = 1")
        row = await cursor.fetchone()
        compliant = row[0] if row else 0

        pass_rate = (compliant / total * 100.0) if total > 0 else 0.0
        return {
            "total_audits": total,
            "compliant_audits": compliant,
            "pass_rate_pct": round(pass_rate, 2),
        }

    async def get_explanation_coverage_stats(self) -> dict[str, Any]:
        """Get explanation coverage statistics for compliance reporting."""
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM forensic_events WHERE event_type = 'inference_result'"
        )
        row = await cursor.fetchone()
        total_inferences = row[0] if row else 0

        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM forensic_events WHERE event_type = 'explanation_generated'"
        )
        row = await cursor.fetchone()
        total_explanations = row[0] if row else 0

        coverage = (total_explanations / total_inferences * 100.0) if total_inferences > 0 else 0.0
        return {
            "total_inferences": total_inferences,
            "total_explanations": total_explanations,
            "coverage_pct": round(coverage, 2),
        }

    async def get_bundle_integrity_stats(self) -> dict[str, Any]:
        """Get forensic bundle integrity statistics for compliance reporting."""
        cursor = await self.db.execute("SELECT COUNT(*) FROM forensic_bundles")
        row = await cursor.fetchone()
        total = row[0] if row else 0

        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM forensic_bundles WHERE status = 'verified'"
        )
        row = await cursor.fetchone()
        verified = row[0] if row else 0

        integrity = (verified / total * 100.0) if total > 0 else 0.0
        return {
            "total_bundles": total,
            "verified_bundles": verified,
            "integrity_pct": round(integrity, 2),
        }

    # --- Merkle Checkpoints ---

    async def insert_merkle_checkpoint(self, checkpoint: dict) -> None:
        """Persist a Merkle tree checkpoint."""
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        await self.db.execute(
            "INSERT INTO merkle_checkpoints "
            "(id, created_at, leaf_count, root_hash, leaves_json, event_index_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                checkpoint["id"],
                now,
                checkpoint["leaf_count"],
                checkpoint["root_hash"],
                json.dumps(checkpoint["leaves"]),
                json.dumps(checkpoint["event_index"]),
            ),
        )
        await self.db.commit()

    async def get_latest_merkle_checkpoint(self) -> dict | None:
        """Get the most recent Merkle checkpoint."""
        cursor = await self.db.execute(
            "SELECT * FROM merkle_checkpoints ORDER BY leaf_count DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "leaf_count": row[2],
            "root_hash": row[3],
            "leaves": json.loads(row[4]),
            "event_index": json.loads(row[5]),
        }

    # --- DID Keys (US-040) ---

    async def insert_did_key(
        self,
        did: str,
        public_key: bytes,
        encrypted_private_key: bytes | None = None,
        key_type: str = "Ed25519",
        created_at: str | None = None,
        label: str | None = None,
        is_platform_key: bool = False,
    ) -> None:
        """Persist a DID key entry."""
        now = (
            created_at
            or __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        )
        await self.db.execute(
            "INSERT OR REPLACE INTO did_keys "
            "(did, public_key, encrypted_private_key, key_type, created_at, label, is_platform_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (did, public_key, encrypted_private_key, key_type, now, label, int(is_platform_key)),
        )
        await self.db.commit()

    async def get_did_key(self, did: str) -> dict[str, Any] | None:
        """Retrieve a stored DID key by its DID identifier."""
        cursor = await self.db.execute("SELECT * FROM did_keys WHERE did = ?", (did,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "did": row[0],
            "public_key": row[1],
            "encrypted_private_key": row[2],
            "key_type": row[3],
            "created_at": row[4],
            "label": row[5],
            "is_platform_key": bool(row[6]),
        }

    async def list_did_keys(self, platform_only: bool = False) -> list[dict[str, Any]]:
        """List stored DID keys, optionally filtering to platform keys only."""
        if platform_only:
            cursor = await self.db.execute(
                "SELECT * FROM did_keys WHERE is_platform_key = 1 ORDER BY created_at ASC"
            )
        else:
            cursor = await self.db.execute("SELECT * FROM did_keys ORDER BY created_at ASC")
        rows = await cursor.fetchall()
        return [
            {
                "did": r[0],
                "public_key": r[1],
                "encrypted_private_key": r[2],
                "key_type": r[3],
                "created_at": r[4],
                "label": r[5],
                "is_platform_key": bool(r[6]),
            }
            for r in rows
        ]

    # --- Retention Policies (US-051) ---

    async def insert_retention_policy(self, policy: dict[str, Any]) -> None:
        """Insert or replace a retention policy."""
        await self.db.execute(
            "INSERT OR REPLACE INTO retention_policies "
            "(id, name, entity_type, retention_days, action, enabled, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                policy["id"],
                policy["name"],
                policy["entity_type"],
                policy["retention_days"],
                policy.get("action", "erase"),
                int(policy.get("enabled", True)),
                policy["created_at"],
                policy["updated_at"],
            ),
        )
        await self.db.commit()

    async def get_retention_policy(self, policy_id: str) -> dict[str, Any] | None:
        """Get a retention policy by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM retention_policies WHERE id = ?", (policy_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_retention_policy(row)

    async def list_retention_policies(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        """List all retention policies."""
        if enabled_only:
            cursor = await self.db.execute(
                "SELECT * FROM retention_policies WHERE enabled = 1 ORDER BY created_at ASC"
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM retention_policies ORDER BY created_at ASC"
            )
        rows = await cursor.fetchall()
        return [self._row_to_retention_policy(r) for r in rows]

    async def delete_retention_policy(self, policy_id: str) -> bool:
        """Delete a retention policy. Returns True if a row was deleted."""
        cursor = await self.db.execute("DELETE FROM retention_policies WHERE id = ?", (policy_id,))
        await self.db.commit()
        return cursor.rowcount > 0

    def _row_to_retention_policy(self, row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "name": row[1],
            "entity_type": row[2],
            "retention_days": row[3],
            "action": row[4],
            "enabled": bool(row[5]),
            "created_at": row[6],
            "updated_at": row[7],
        }

    async def get_subjects_exceeding_retention(
        self, entity_type: str, retention_days: int
    ) -> list[str]:
        """Find subjects with consent records older than retention_days.

        Returns list of subject_ids whose earliest consent is past the
        retention window.
        """
        cursor = await self.db.execute(
            "SELECT DISTINCT subject_id FROM consent_records "
            "WHERE julianday('now') - julianday(timestamp) > ? "
            "ORDER BY subject_id",
            (retention_days,),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]
