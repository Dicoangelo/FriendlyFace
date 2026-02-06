"""Async SQLite storage layer for the Blockchain Forensic Layer.

Uses aiosqlite for async access. Repository pattern for clean separation.
Designed for dev/testing — production would use Supabase.
"""

from __future__ import annotations

import json
from pathlib import Path
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
        cursor = await self.db.execute(
            "SELECT * FROM forensic_events ORDER BY sequence_number ASC"
        )
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
                provenance_chain, bias_audit, zk_proof_placeholder,
                did_credential_placeholder, bundle_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(bundle.id),
                bundle.created_at.isoformat(),
                bundle.status.value,
                json.dumps([str(e) for e in bundle.event_ids]),
                bundle.merkle_root,
                json.dumps([p.model_dump() for p in bundle.merkle_proofs]),
                json.dumps([str(p) for p in bundle.provenance_chain]),
                bundle.bias_audit.model_dump_json() if bundle.bias_audit else None,
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

        return ForensicBundle(
            id=UUID(row["id"]),
            created_at=row["created_at"],
            status=BundleStatus(row["status"]),
            event_ids=[UUID(e) for e in json.loads(row["event_ids"])],
            merkle_root=row["merkle_root"],
            merkle_proofs=proofs,
            provenance_chain=[UUID(p) for p in json.loads(row["provenance_chain"])],
            bias_audit=bias,
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
