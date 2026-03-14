"""Typed dataclasses for FriendlyFace API responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ForensicEvent:
    """A recorded forensic event."""

    event_id: str
    event_type: str
    event_hash: str
    timestamp: str
    content: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bundle:
    """A forensic bundle grouping events with cryptographic proofs."""

    bundle_id: str
    events: List[str] = field(default_factory=list)
    merkle_root: str = ""
    zk_proof: Optional[Dict[str, Any]] = None
    credential: Optional[Dict[str, Any]] = None


@dataclass
class Seal:
    """A ForensicSeal compliance certificate."""

    seal_id: str
    credential: Optional[Dict[str, Any]] = None
    verification_url: str = ""
    expires_at: str = ""
    compliance_summary: Optional[Dict[str, Any]] = None


@dataclass
class VerificationResult:
    """Result of verifying a seal or credential."""

    valid: bool
    checks: Dict[str, Any] = field(default_factory=dict)
    issuer: str = ""
    issued_at: str = ""
    expires_at: str = ""
    compliance_score: float = 0.0


@dataclass
class ConsentStatus:
    """Current consent status for a subject."""

    subject_id: str
    has_consent: bool = False
    purpose: str = ""
    granted_at: str = ""


@dataclass
class ConsentRecord:
    """A consent grant record."""

    consent_id: str
    subject_id: str
    purpose: str
    granted_by: str = ""
    granted_at: str = ""


@dataclass
class AuditResult:
    """Result of a fairness/bias audit."""

    audit_id: str
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    pass_status: bool = True


@dataclass
class RecognitionResult:
    """Result of a proxy recognition request."""

    predictions: List[Dict[str, Any]] = field(default_factory=list)
    event_ids: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
