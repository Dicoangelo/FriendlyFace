"""Async Supabase storage layer for the Blockchain Forensic Layer.

Uses supabase-py for async access. Same repository interface as database.py.
Designed for production use -- selectable via FF_STORAGE=supabase env var.
"""

from __future__ import annotations

import json
import os
from uuid import UUID

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


class SupabaseDatabase:
    """Async Supabase database wrapper with the same interface as Database."""

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
    ) -> None:
        self.url = url or os.environ.get("SUPABASE_URL", "")
        self.key = key or os.environ.get("SUPABASE_KEY", "")
        self._client: object | None = None

    async def connect(self) -> None:
        from supabase import acreate_client

        self._client = await acreate_client(self.url, self.key)

    async def close(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("Supabase client not connected. Call connect() first.")
        return self._client

    # --- ForensicEvent ---

    async def insert_event(self, event: ForensicEvent) -> None:
        await (
            self.client.table("forensic_events")
            .insert(
                {
                    "id": str(event.id),
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "actor": event.actor,
                    "payload": json.dumps(event.payload),
                    "previous_hash": event.previous_hash,
                    "event_hash": event.event_hash,
                    "sequence_number": event.sequence_number,
                }
            )
            .execute()
        )

    async def get_event(self, event_id: UUID) -> ForensicEvent | None:
        response = await (
            self.client.table("forensic_events").select("*").eq("id", str(event_id)).execute()
        )
        if not response.data:
            return None
        return self._row_to_event(response.data[0])

    async def get_latest_event(self) -> ForensicEvent | None:
        response = await (
            self.client.table("forensic_events")
            .select("*")
            .order("sequence_number", desc=True)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return self._row_to_event(response.data[0])

    async def get_all_events(self) -> list[ForensicEvent]:
        response = await (
            self.client.table("forensic_events")
            .select("*")
            .order("sequence_number", desc=False)
            .execute()
        )
        return [self._row_to_event(r) for r in response.data]

    async def get_event_count(self) -> int:
        response = await self.client.table("forensic_events").select("id", count="exact").execute()
        return response.count if response.count is not None else 0

    def _row_to_event(self, row: dict) -> ForensicEvent:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return ForensicEvent(
            id=UUID(row["id"]),
            event_type=EventType(row["event_type"]),
            timestamp=row["timestamp"],
            actor=row["actor"],
            payload=payload,
            previous_hash=row["previous_hash"],
            event_hash=row["event_hash"],
            sequence_number=row["sequence_number"],
        )

    # --- ProvenanceNode ---

    async def insert_provenance_node(self, node: ProvenanceNode) -> None:
        await (
            self.client.table("provenance_nodes")
            .insert(
                {
                    "id": str(node.id),
                    "entity_type": node.entity_type,
                    "entity_id": node.entity_id,
                    "created_at": node.created_at.isoformat(),
                    "metadata": json.dumps(node.metadata),
                    "parents": json.dumps([str(p) for p in node.parents]),
                    "relations": json.dumps([r.value for r in node.relations]),
                    "node_hash": node.node_hash,
                }
            )
            .execute()
        )

    async def get_provenance_node(self, node_id: UUID) -> ProvenanceNode | None:
        response = await (
            self.client.table("provenance_nodes").select("*").eq("id", str(node_id)).execute()
        )
        if not response.data:
            return None
        return self._row_to_provenance(response.data[0])

    def _row_to_provenance(self, row: dict) -> ProvenanceNode:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        parents = row["parents"]
        if isinstance(parents, str):
            parents = json.loads(parents)
        relations = row["relations"]
        if isinstance(relations, str):
            relations = json.loads(relations)

        return ProvenanceNode(
            id=UUID(row["id"]),
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            created_at=row["created_at"],
            metadata=metadata,
            parents=[UUID(p) for p in parents],
            relations=[ProvenanceRelation(r) for r in relations],
            node_hash=row["node_hash"],
        )

    # --- ForensicBundle ---

    async def insert_bundle(self, bundle: ForensicBundle) -> None:
        await (
            self.client.table("forensic_bundles")
            .insert(
                {
                    "id": str(bundle.id),
                    "created_at": bundle.created_at.isoformat(),
                    "status": bundle.status.value,
                    "event_ids": json.dumps([str(e) for e in bundle.event_ids]),
                    "merkle_root": bundle.merkle_root,
                    "merkle_proofs": json.dumps([p.model_dump() for p in bundle.merkle_proofs]),
                    "provenance_chain": json.dumps([str(p) for p in bundle.provenance_chain]),
                    "bias_audit": bundle.bias_audit.model_dump_json()
                    if bundle.bias_audit
                    else None,
                    "recognition_artifacts": json.dumps(bundle.recognition_artifacts)
                    if bundle.recognition_artifacts
                    else None,
                    "fl_artifacts": json.dumps(bundle.fl_artifacts)
                    if bundle.fl_artifacts
                    else None,
                    "bias_report": json.dumps(bundle.bias_report) if bundle.bias_report else None,
                    "explanation_artifacts": json.dumps(bundle.explanation_artifacts)
                    if bundle.explanation_artifacts
                    else None,
                    "zk_proof_placeholder": bundle.zk_proof_placeholder,
                    "did_credential_placeholder": bundle.did_credential_placeholder,
                    "bundle_hash": bundle.bundle_hash,
                }
            )
            .execute()
        )

    async def get_bundle(self, bundle_id: UUID) -> ForensicBundle | None:
        response = await (
            self.client.table("forensic_bundles").select("*").eq("id", str(bundle_id)).execute()
        )
        if not response.data:
            return None
        return self._row_to_bundle(response.data[0])

    async def update_bundle_status(self, bundle_id: UUID, status: BundleStatus) -> None:
        await (
            self.client.table("forensic_bundles")
            .update({"status": status.value})
            .eq("id", str(bundle_id))
            .execute()
        )

    def _row_to_bundle(self, row: dict) -> ForensicBundle:
        bias = None
        if row["bias_audit"]:
            bias = BiasAuditRecord.model_validate_json(row["bias_audit"])

        merkle_proofs_raw = row["merkle_proofs"]
        if isinstance(merkle_proofs_raw, str):
            merkle_proofs_raw = json.loads(merkle_proofs_raw)
        proofs = [MerkleProof.model_validate(p) for p in merkle_proofs_raw]

        event_ids_raw = row["event_ids"]
        if isinstance(event_ids_raw, str):
            event_ids_raw = json.loads(event_ids_raw)

        provenance_chain_raw = row["provenance_chain"]
        if isinstance(provenance_chain_raw, str):
            provenance_chain_raw = json.loads(provenance_chain_raw)

        def _load_json_field(raw):
            if raw is None:
                return None
            if isinstance(raw, str):
                return json.loads(raw)
            return raw

        return ForensicBundle(
            id=UUID(row["id"]),
            created_at=row["created_at"],
            status=BundleStatus(row["status"]),
            event_ids=[UUID(e) for e in event_ids_raw],
            merkle_root=row["merkle_root"],
            merkle_proofs=proofs,
            provenance_chain=[UUID(p) for p in provenance_chain_raw],
            bias_audit=bias,
            recognition_artifacts=_load_json_field(row.get("recognition_artifacts")),
            fl_artifacts=_load_json_field(row.get("fl_artifacts")),
            bias_report=_load_json_field(row.get("bias_report")),
            explanation_artifacts=_load_json_field(row.get("explanation_artifacts")),
            zk_proof_placeholder=row["zk_proof_placeholder"],
            did_credential_placeholder=row["did_credential_placeholder"],
            bundle_hash=row["bundle_hash"],
        )

    # --- BiasAudit ---

    async def insert_bias_audit(self, audit: BiasAuditRecord) -> None:
        await (
            self.client.table("bias_audits")
            .insert(
                {
                    "id": str(audit.id),
                    "event_id": str(audit.event_id) if audit.event_id else None,
                    "timestamp": audit.timestamp.isoformat(),
                    "demographic_parity_gap": audit.demographic_parity_gap,
                    "equalized_odds_gap": audit.equalized_odds_gap,
                    "groups_evaluated": json.dumps(audit.groups_evaluated),
                    "compliant": audit.compliant,
                    "details": json.dumps(audit.details),
                }
            )
            .execute()
        )
