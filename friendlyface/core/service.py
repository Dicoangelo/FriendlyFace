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
        recognition_artifacts: dict | None = None,
        fl_artifacts: dict | None = None,
        bias_report: dict | None = None,
        explanation_artifacts: dict | None = None,
        layer_filters: list[str] | None = None,
    ) -> ForensicBundle:
        """Create a self-verifiable forensic bundle.

        If layer_filters is provided, only the specified layers are included.
        Valid filters: "recognition", "fl", "bias", "explanation".
        When layer_filters is None, all provided artifacts are included.
        """
        # Apply layer filters if specified
        if layer_filters is not None:
            filters = set(layer_filters)
            if "recognition" not in filters:
                recognition_artifacts = None
            if "fl" not in filters:
                fl_artifacts = None
            if "bias" not in filters:
                bias_report = None
                bias_audit = None
            if "explanation" not in filters:
                explanation_artifacts = None

        # Collect artifacts from events when not explicitly provided
        if recognition_artifacts is None and (layer_filters is None or "recognition" in (layer_filters or [])):
            recognition_artifacts = await self._collect_recognition_artifacts(event_ids)
        if fl_artifacts is None and (layer_filters is None or "fl" in (layer_filters or [])):
            fl_artifacts = await self._collect_fl_artifacts(event_ids)
        if bias_report is None and (layer_filters is None or "bias" in (layer_filters or [])):
            bias_report = await self._collect_bias_artifacts(event_ids)
        if explanation_artifacts is None and (layer_filters is None or "explanation" in (layer_filters or [])):
            explanation_artifacts = await self._collect_explanation_artifacts(event_ids)

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
            recognition_artifacts=recognition_artifacts,
            fl_artifacts=fl_artifacts,
            bias_report=bias_report,
            explanation_artifacts=explanation_artifacts,
        ).seal()

        await self.db.insert_bundle(bundle)
        if bias_audit:
            await self.db.insert_bias_audit(bias_audit)
        return bundle

    async def _collect_recognition_artifacts(self, event_ids: list[UUID]) -> dict | None:
        """Collect recognition layer artifacts from events."""
        artifacts: dict[str, Any] = {"inference_events": [], "training_events": []}
        for eid in event_ids:
            event = await self.db.get_event(eid)
            if event is None:
                continue
            if event.event_type == EventType.INFERENCE_RESULT:
                artifacts["inference_events"].append({
                    "event_id": str(event.id),
                    "event_hash": event.event_hash,
                    "payload": event.payload,
                })
            elif event.event_type in (EventType.TRAINING_START, EventType.TRAINING_COMPLETE):
                artifacts["training_events"].append({
                    "event_id": str(event.id),
                    "event_hash": event.event_hash,
                    "payload": event.payload,
                })
        if not artifacts["inference_events"] and not artifacts["training_events"]:
            return None
        return artifacts

    async def _collect_fl_artifacts(self, event_ids: list[UUID]) -> dict | None:
        """Collect federated learning layer artifacts from events."""
        rounds: list[dict[str, Any]] = []
        security_alerts: list[dict[str, Any]] = []
        for eid in event_ids:
            event = await self.db.get_event(eid)
            if event is None:
                continue
            if event.event_type == EventType.FL_ROUND:
                rounds.append({
                    "event_id": str(event.id),
                    "event_hash": event.event_hash,
                    "payload": event.payload,
                })
            elif event.event_type == EventType.SECURITY_ALERT:
                if event.payload.get("alert_type") == "data_poisoning":
                    security_alerts.append({
                        "event_id": str(event.id),
                        "event_hash": event.event_hash,
                        "payload": event.payload,
                    })
        if not rounds and not security_alerts:
            return None
        return {"rounds": rounds, "security_alerts": security_alerts}

    async def _collect_bias_artifacts(self, event_ids: list[UUID]) -> dict | None:
        """Collect bias audit artifacts from events."""
        audits: list[dict[str, Any]] = []
        for eid in event_ids:
            event = await self.db.get_event(eid)
            if event is None:
                continue
            if event.event_type == EventType.BIAS_AUDIT:
                audits.append({
                    "event_id": str(event.id),
                    "event_hash": event.event_hash,
                    "payload": event.payload,
                })
        if not audits:
            return None
        return {"audits": audits}

    async def _collect_explanation_artifacts(self, event_ids: list[UUID]) -> dict | None:
        """Collect explanation layer artifacts from events."""
        explanations: list[dict[str, Any]] = []
        for eid in event_ids:
            event = await self.db.get_event(eid)
            if event is None:
                continue
            if event.event_type == EventType.EXPLANATION_GENERATED:
                explanations.append({
                    "event_id": str(event.id),
                    "event_hash": event.event_hash,
                    "payload": event.payload,
                })
        if not explanations:
            return None
        return {"explanations": explanations}

    async def get_bundle(self, bundle_id: UUID) -> ForensicBundle | None:
        return await self.db.get_bundle(bundle_id)

    async def verify_bundle(self, bundle_id: UUID) -> dict[str, Any]:
        """Verify a forensic bundle's integrity including all layer artifacts."""
        bundle = await self.db.get_bundle(bundle_id)
        if bundle is None:
            return {"valid": False, "error": "Bundle not found"}

        results: dict[str, Any] = {"bundle_id": str(bundle_id)}

        # 1. Bundle hash integrity (covers all fields including layer artifacts)
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

        # 4. Layer artifact integrity checks
        layer_results: dict[str, Any] = {}
        if bundle.recognition_artifacts is not None:
            layer_results["recognition"] = await self._verify_layer_events(
                bundle.recognition_artifacts, "recognition"
            )
        if bundle.fl_artifacts is not None:
            layer_results["fl"] = await self._verify_layer_events(
                bundle.fl_artifacts, "fl"
            )
        if bundle.bias_report is not None:
            layer_results["bias"] = await self._verify_layer_events(
                bundle.bias_report, "bias"
            )
        if bundle.explanation_artifacts is not None:
            layer_results["explanation"] = await self._verify_layer_events(
                bundle.explanation_artifacts, "explanation"
            )
        results["layer_artifacts"] = layer_results
        layers_valid = all(lr["valid"] for lr in layer_results.values()) if layer_results else True

        # 5. Overall
        results["valid"] = (
            results["bundle_hash_valid"]
            and results["merkle_proofs_valid"]
            and results["provenance_valid"]
            and layers_valid
        )

        # Update status
        new_status = BundleStatus.VERIFIED if results["valid"] else BundleStatus.TAMPERED
        await self.db.update_bundle_status(bundle_id, new_status)
        results["status"] = new_status.value

        return results

    async def _verify_layer_events(
        self, artifacts: dict, layer_name: str
    ) -> dict[str, Any]:
        """Verify that events referenced in layer artifacts exist and have valid hashes."""
        result: dict[str, Any] = {"layer": layer_name, "valid": True, "errors": []}
        # Extract event entries from the artifacts dict
        all_entries: list[dict] = []
        for key, value in artifacts.items():
            if isinstance(value, list):
                all_entries.extend(e for e in value if isinstance(e, dict) and "event_id" in e)
        for entry in all_entries:
            eid_str = entry.get("event_id")
            if eid_str is None:
                continue
            from uuid import UUID as _UUID
            event = await self.db.get_event(_UUID(eid_str))
            if event is None:
                result["valid"] = False
                result["errors"].append(f"Event {eid_str} not found")
                continue
            stored_hash = entry.get("event_hash")
            if stored_hash and stored_hash != event.event_hash:
                result["valid"] = False
                result["errors"].append(
                    f"Event {eid_str} hash mismatch in {layer_name} artifacts"
                )
        return result

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
