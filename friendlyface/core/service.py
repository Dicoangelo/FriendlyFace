"""Forensic service layer — orchestrates event chains, Merkle tree, provenance, and bundles."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from friendlyface.core.merkle import MerkleTree
from friendlyface.core.models import (
    BiasAuditRecord,
    BundleStatus,
    EventType,
    ForensicBundle,
    ForensicEvent,
    MerkleProof,
    ProvenanceRelation,
)
from friendlyface.core.provenance import ProvenanceDAG
from friendlyface.storage.database import Database


class ForensicService:
    """Orchestrates the Blockchain Forensic Layer."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.merkle = MerkleTree()
        self.provenance = ProvenanceDAG()
        self._event_index: dict[UUID, int] = {}  # event_id -> leaf_index

    async def initialize(self) -> None:
        """Rebuild in-memory state from persisted events."""
        events = await self.db.get_all_events()
        for event in events:
            leaf_idx = self.merkle.leaf_count
            self.merkle.add_leaf(event.event_hash)
            self._event_index[event.id] = leaf_idx

    async def record_event(
        self,
        event_type: EventType,
        actor: str,
        payload: dict[str, Any] | None = None,
    ) -> ForensicEvent:
        """Record a new forensic event with hash chaining."""
        latest = await self.db.get_latest_event()
        previous_hash = latest.event_hash if latest else "GENESIS"
        seq = (latest.sequence_number + 1) if latest else 0

        event = ForensicEvent(
            event_type=event_type,
            actor=actor,
            payload=payload or {},
            previous_hash=previous_hash,
            sequence_number=seq,
        ).seal()

        await self.db.insert_event(event)

        leaf_idx = self.merkle.leaf_count
        self.merkle.add_leaf(event.event_hash)
        self._event_index[event.id] = leaf_idx

        return event

    async def get_event(self, event_id: UUID) -> ForensicEvent | None:
        return await self.db.get_event(event_id)

    async def get_all_events(self) -> list[ForensicEvent]:
        return await self.db.get_all_events()

    def get_merkle_root(self) -> str | None:
        return self.merkle.root

    def get_merkle_proof(self, event_id: UUID) -> MerkleProof | None:
        leaf_idx = self._event_index.get(event_id)
        if leaf_idx is None:
            return None
        return self.merkle.get_proof(leaf_idx)

    def verify_merkle_proof(self, proof: MerkleProof) -> bool:
        return self.merkle.verify_proof(proof)

    async def verify_chain_integrity(self) -> dict[str, Any]:
        """Verify the entire hash chain of events."""
        events = await self.db.get_all_events()
        if not events:
            return {"valid": True, "count": 0, "errors": []}

        errors: list[str] = []
        for i, event in enumerate(events):
            if not event.verify():
                errors.append(f"Event {event.id} hash mismatch at seq {event.sequence_number}")
            if i == 0:
                if event.previous_hash != "GENESIS":
                    errors.append(f"First event {event.id} has non-GENESIS previous_hash")
            else:
                if event.previous_hash != events[i - 1].event_hash:
                    errors.append(
                        f"Event {event.id} previous_hash doesn't match "
                        f"event {events[i-1].id} hash"
                    )

        return {"valid": len(errors) == 0, "count": len(events), "errors": errors}

    async def create_bundle(
        self,
        event_ids: list[UUID],
        provenance_node_ids: list[UUID] | None = None,
        bias_audit: BiasAuditRecord | None = None,
    ) -> ForensicBundle:
        """Create a self-verifiable forensic bundle."""
        # Gather Merkle proofs for each event
        proofs: list[MerkleProof] = []
        for eid in event_ids:
            proof = self.get_merkle_proof(eid)
            if proof is not None:
                proofs.append(proof)

        bundle = ForensicBundle(
            event_ids=event_ids,
            merkle_root=self.merkle.root or "",
            merkle_proofs=proofs,
            provenance_chain=provenance_node_ids or [],
            bias_audit=bias_audit,
        ).seal()

        await self.db.insert_bundle(bundle)
        if bias_audit:
            await self.db.insert_bias_audit(bias_audit)
        return bundle

    async def get_bundle(self, bundle_id: UUID) -> ForensicBundle | None:
        return await self.db.get_bundle(bundle_id)

    async def verify_bundle(self, bundle_id: UUID) -> dict[str, Any]:
        """Verify a forensic bundle's integrity."""
        bundle = await self.db.get_bundle(bundle_id)
        if bundle is None:
            return {"valid": False, "error": "Bundle not found"}

        results: dict[str, Any] = {"bundle_id": str(bundle_id)}

        # 1. Bundle hash integrity
        results["bundle_hash_valid"] = bundle.verify()

        # 2. Merkle proof verification
        proof_results = []
        for proof in bundle.merkle_proofs:
            proof_results.append({
                "leaf_hash": proof.leaf_hash,
                "valid": proof.verify(),
            })
        results["merkle_proofs_valid"] = all(p["valid"] for p in proof_results)
        results["merkle_proofs"] = proof_results

        # 3. Provenance chain verification
        prov_valid = True
        for nid in bundle.provenance_chain:
            if not self.provenance.verify_node(nid):
                prov_valid = False
                break
        results["provenance_valid"] = prov_valid

        # 4. Overall
        results["valid"] = (
            results["bundle_hash_valid"]
            and results["merkle_proofs_valid"]
            and results["provenance_valid"]
        )

        # Update status
        new_status = BundleStatus.VERIFIED if results["valid"] else BundleStatus.TAMPERED
        await self.db.update_bundle_status(bundle_id, new_status)
        results["status"] = new_status.value

        return results

    def add_provenance_node(
        self,
        entity_type: str,
        entity_id: str,
        parents: list[UUID] | None = None,
        relations: list[ProvenanceRelation] | None = None,
        metadata: dict | None = None,
    ):
        """Add a provenance node and persist it."""
        return self.provenance.add_node(
            entity_type=entity_type,
            entity_id=entity_id,
            parents=parents,
            relations=relations,
            metadata=metadata,
        )

    def get_provenance_chain(self, node_id: UUID):
        return self.provenance.get_chain(node_id)
