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

DEFAULT_DB_PATH = Path("friendlyface.db")

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
"""


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()

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

    async def get_event_count(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM forensic_events")
        row = await cursor.fetchone()
        return row[0] if row else 0

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
