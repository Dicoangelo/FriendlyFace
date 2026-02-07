"""Domain models for the Blockchain Forensic Layer.

Based on Mohammed's ICDF2C 2024 forensic-friendly schema:
- ForensicEvent: Immutable, hash-chained event records
- MerkleNode: Nodes in the append-only forensic Merkle tree
- ProvenanceNode: DAG nodes tracking data lineage (training→model→inference→explanation→bundle)
- ForensicBundle: Self-verifiable output artifact
- BiasAuditRecord: Demographic parity + equalized odds per event
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> UUID:
    return uuid4()


def canonical_json(data: dict[str, Any]) -> str:
    """Deterministic JSON serialization for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    MODEL_REGISTERED = "model_registered"
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESULT = "inference_result"
    EXPLANATION_GENERATED = "explanation_generated"
    BIAS_AUDIT = "bias_audit"
    CONSENT_RECORDED = "consent_recorded"
    CONSENT_UPDATE = "consent_update"
    BUNDLE_CREATED = "bundle_created"
    FL_ROUND = "fl_round"
    SECURITY_ALERT = "security_alert"
    COMPLIANCE_REPORT = "compliance_report"


class ProvenanceRelation(str, Enum):
    DERIVED_FROM = "derived_from"
    GENERATED_BY = "generated_by"
    USED = "used"
    ATTRIBUTED_TO = "attributed_to"


class BundleStatus(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"
    VERIFIED = "verified"
    TAMPERED = "tampered"


# ---------------------------------------------------------------------------
# ForensicEvent — the fundamental record
# ---------------------------------------------------------------------------


class ForensicEvent(BaseModel):
    """Immutable forensic event with SHA-256 hash chaining.

    Each event hashes its own content + the previous event's hash to form
    an unbreakable chain (Mohammed ICDF2C 2024 pattern).
    """

    id: UUID = Field(default_factory=_new_id)
    event_type: EventType
    timestamp: datetime = Field(default_factory=_utcnow)
    actor: str = Field(description="Who/what triggered the event")
    payload: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str = Field(default="GENESIS", description="Hash of the preceding event")
    event_hash: str = Field(default="", description="SHA-256 of canonical(event) + previous_hash")
    sequence_number: int = Field(default=0)

    def compute_hash(self) -> str:
        """Compute SHA-256 over canonical JSON of hashable fields + previous_hash."""
        hashable = {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
            "sequence_number": self.sequence_number,
        }
        return sha256_hex(canonical_json(hashable))

    def seal(self) -> ForensicEvent:
        """Compute and set event_hash. Returns self for chaining."""
        self.event_hash = self.compute_hash()
        return self

    def verify(self) -> bool:
        """Check that event_hash matches recomputed hash."""
        return self.event_hash == self.compute_hash()


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------


class MerkleNode(BaseModel):
    """Node in the append-only forensic Merkle tree (BioZero pattern)."""

    hash: str
    left: str | None = None
    right: str | None = None
    level: int = 0
    index: int = 0


class MerkleProof(BaseModel):
    """Inclusion proof for a leaf in the Merkle tree."""

    leaf_hash: str
    leaf_index: int
    proof_hashes: list[str]
    proof_directions: list[str]  # "left" or "right" — sibling position
    root_hash: str

    def verify(self) -> bool:
        """Verify inclusion proof against root."""
        current = self.leaf_hash
        for sibling, direction in zip(self.proof_hashes, self.proof_directions):
            if direction == "left":
                current = sha256_hex(sibling + current)
            else:
                current = sha256_hex(current + sibling)
        return current == self.root_hash


# ---------------------------------------------------------------------------
# Provenance DAG
# ---------------------------------------------------------------------------


class ProvenanceNode(BaseModel):
    """Node in the forensic provenance DAG.

    Chain: training → model → inference → explanation → bundle
    (Mohammed ICDF2C 2024 schema)
    """

    id: UUID = Field(default_factory=_new_id)
    entity_type: str = Field(description="E.g. 'dataset', 'model', 'inference', 'explanation'")
    entity_id: str = Field(description="Reference to the actual entity")
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parents: list[UUID] = Field(default_factory=list, description="Parent node IDs in the DAG")
    relations: list[ProvenanceRelation] = Field(
        default_factory=list, description="Relation types to parents (same order)"
    )
    node_hash: str = Field(default="")

    def compute_hash(self) -> str:
        hashable = {
            "id": str(self.id),
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "parents": [str(p) for p in self.parents],
            "relations": [r.value for r in self.relations],
        }
        return sha256_hex(canonical_json(hashable))

    def seal(self) -> ProvenanceNode:
        self.node_hash = self.compute_hash()
        return self


# ---------------------------------------------------------------------------
# ForensicBundle — the self-verifiable output artifact
# ---------------------------------------------------------------------------


class ForensicBundle(BaseModel):
    """Self-verifiable forensic bundle (Mohammed + all SOTA).

    Contains all evidence for a single recognition event:
    event chain, Merkle proof, provenance path, bias audit, and integrity hash.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: UUID = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_utcnow)
    status: BundleStatus = Field(default=BundleStatus.PENDING)

    # References
    event_ids: list[UUID] = Field(description="Ordered event IDs in this bundle")
    merkle_root: str = Field(default="")
    merkle_proofs: list[MerkleProof] = Field(default_factory=list)
    provenance_chain: list[UUID] = Field(
        default_factory=list, description="Provenance node IDs forming the chain"
    )

    # Bias audit summary
    bias_audit: BiasAuditRecord | None = None

    # Full-layer artifact fields (US-009)
    recognition_artifacts: dict[str, Any] | None = Field(default=None)
    fl_artifacts: dict[str, Any] | None = Field(default=None)
    bias_report: dict[str, Any] | None = Field(default=None)
    explanation_artifacts: dict[str, Any] | None = Field(default=None)

    # ZK proof (BioZero pattern, arXiv:2409.17509) — Schnorr non-interactive via Fiat-Shamir
    zk_proof_placeholder: str | None = Field(
        default=None,
        alias="zk_proof",
        description="Schnorr ZK proof over bundle hash",
    )
    # DID/VC (TBFL pattern, arXiv:2602.02629) — Ed25519 Verifiable Credential
    did_credential_placeholder: str | None = Field(
        default=None,
        alias="did_credential",
        description="DID-signed Verifiable Credential for this bundle",
    )

    # Integrity
    bundle_hash: str = Field(default="")

    def compute_hash(self) -> str:
        hashable: dict[str, Any] = {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "event_ids": [str(e) for e in self.event_ids],
            "merkle_root": self.merkle_root,
            "provenance_chain": [str(p) for p in self.provenance_chain],
            "bias_audit": self.bias_audit.model_dump() if self.bias_audit else None,
            "recognition_artifacts": self.recognition_artifacts,
            "fl_artifacts": self.fl_artifacts,
            "bias_report": self.bias_report,
            "explanation_artifacts": self.explanation_artifacts,
        }
        return sha256_hex(canonical_json(hashable))

    def seal(self) -> ForensicBundle:
        self.bundle_hash = self.compute_hash()
        self.status = BundleStatus.COMPLETE
        return self

    def verify(self) -> bool:
        return self.bundle_hash == self.compute_hash()


# ---------------------------------------------------------------------------
# Bias Audit (arXiv:2505.14320)
# ---------------------------------------------------------------------------


class BiasAuditRecord(BaseModel):
    """Bias audit metrics per forensic event (demographic parity + equalized odds)."""

    id: UUID = Field(default_factory=_new_id)
    event_id: UUID | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    demographic_parity_gap: float = Field(
        description="Max gap in positive prediction rates across groups"
    )
    equalized_odds_gap: float = Field(description="Max gap in TPR/FPR across groups")
    groups_evaluated: list[str] = Field(default_factory=list)
    compliant: bool = Field(default=True, description="EU AI Act Article 5/14 compliance flag")
    details: dict[str, Any] = Field(default_factory=dict)


# Update ForensicBundle forward ref
ForensicBundle.model_rebuild()
