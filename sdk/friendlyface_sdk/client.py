"""FriendlyFaceClient — main SDK entry point."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from friendlyface_sdk.models import (
    AuditResult,
    Bundle,
    ConsentRecord,
    ConsentStatus,
    ForensicEvent,
    RecognitionResult,
    Seal,
    VerificationResult,
)
from friendlyface_sdk.session import ForensicSession


class FriendlyFaceClient:
    """Client for the FriendlyFace forensic AI platform API.

    Args:
        base_url: Base URL of the FriendlyFace server (trailing slash stripped).
        api_key: Optional API key for authentication.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Send an HTTP request and raise on non-2xx status."""
        url = f"{self.base_url}{path}"
        resp = self._session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ForensicEvent:
        """Record a forensic event."""
        payload: Dict[str, Any] = {
            "event_type": event_type,
            "actor": content.get("actor", "sdk"),
            "payload": content,
        }
        if metadata:
            payload["payload"]["metadata"] = metadata
        resp = self._request("POST", "/events", json=payload)
        data = resp.json()
        return ForensicEvent(
            event_id=data.get("id", ""),
            event_type=data.get("event_type", event_type),
            event_hash=data.get("event_hash", ""),
            timestamp=data.get("timestamp", ""),
            content=data,
        )

    # ------------------------------------------------------------------
    # Consent
    # ------------------------------------------------------------------

    def check_consent(self, subject_id: str, purpose: str = "recognition") -> ConsentStatus:
        """Check consent status for a subject."""
        resp = self._request("GET", f"/consent/status/{subject_id}", params={"purpose": purpose})
        data = resp.json()
        return ConsentStatus(
            subject_id=data.get("subject_id", subject_id),
            has_consent=data.get("has_consent", False),
            purpose=data.get("purpose", purpose),
            granted_at=data.get("granted_at", ""),
        )

    def grant_consent(
        self,
        subject_id: str,
        purpose: str,
        granted_by: str = "sdk",
    ) -> ConsentRecord:
        """Grant consent for a subject+purpose pair."""
        resp = self._request(
            "POST",
            "/consent/grant",
            json={
                "subject_id": subject_id,
                "purpose": purpose,
                "actor": granted_by,
            },
        )
        data = resp.json()
        return ConsentRecord(
            consent_id=data.get("id", data.get("consent_id", "")),
            subject_id=data.get("subject_id", subject_id),
            purpose=data.get("purpose", purpose),
            granted_by=data.get("actor", granted_by),
            granted_at=data.get("granted_at", data.get("timestamp", "")),
        )

    # ------------------------------------------------------------------
    # Bundles
    # ------------------------------------------------------------------

    def create_bundle(
        self,
        event_ids: List[str],
        provenance_node_ids: Optional[List[str]] = None,
    ) -> Bundle:
        """Create a forensic bundle from event IDs."""
        body: Dict[str, Any] = {"event_ids": event_ids}
        if provenance_node_ids:
            body["provenance_node_ids"] = provenance_node_ids
        resp = self._request("POST", "/bundles", json=body)
        data = resp.json()
        return Bundle(
            bundle_id=data.get("id", data.get("bundle_id", "")),
            events=data.get("event_ids", event_ids),
            merkle_root=data.get("merkle_root", ""),
            zk_proof=data.get("zk_proof"),
            credential=data.get("credential"),
        )

    # ------------------------------------------------------------------
    # Seals
    # ------------------------------------------------------------------

    def issue_seal(
        self,
        system_id: str,
        system_name: str,
        assessment_scope: Optional[str] = None,
        bundle_ids: Optional[List[str]] = None,
    ) -> Seal:
        """Issue a ForensicSeal compliance certificate."""
        body: Dict[str, Any] = {
            "system_id": system_id,
            "system_name": system_name,
        }
        if assessment_scope is not None:
            body["assessment_scope"] = assessment_scope
        if bundle_ids is not None:
            body["bundle_ids"] = bundle_ids
        resp = self._request("POST", "/seal/issue", json=body)
        data = resp.json()
        return Seal(
            seal_id=data.get("seal_id", ""),
            credential=data.get("credential"),
            verification_url=data.get("verification_url", ""),
            expires_at=data.get("expires_at", ""),
            compliance_summary=data.get("compliance_summary"),
        )

    def verify_seal(
        self,
        credential: Optional[Dict[str, Any]] = None,
        seal_id: Optional[str] = None,
    ) -> VerificationResult:
        """Verify a ForensicSeal by credential dict or seal ID."""
        if credential is not None:
            resp = self._request("POST", "/seal/verify", json={"credential": credential})
        elif seal_id is not None:
            resp = self._request("GET", f"/seal/verify/{seal_id}")
        else:
            raise ValueError("Provide either credential or seal_id")
        data = resp.json()
        return VerificationResult(
            valid=data.get("valid", False),
            checks=data.get("checks", {}),
            issuer=data.get("issuer", ""),
            issued_at=data.get("issued_at", ""),
            expires_at=data.get("expires_at", ""),
            compliance_score=data.get("compliance_score", 0.0),
        )

    # ------------------------------------------------------------------
    # Fairness / Audit
    # ------------------------------------------------------------------

    def run_audit(
        self,
        predictions: List[Dict[str, Any]],
        demographics: List[Dict[str, Any]],
    ) -> AuditResult:
        """Run a bias/fairness audit."""
        resp = self._request(
            "POST",
            "/fairness/audit",
            json={"groups": demographics, "predictions": predictions},
        )
        data = resp.json()
        return AuditResult(
            audit_id=data.get("audit_id", data.get("id", "")),
            demographic_parity=data.get("demographic_parity", 0.0),
            equalized_odds=data.get("equalized_odds", 0.0),
            pass_status=data.get("pass", data.get("pass_status", True)),
        )

    # ------------------------------------------------------------------
    # Proxy recognition
    # ------------------------------------------------------------------

    def proxy_recognize(
        self,
        image_bytes: bytes,
        upstream_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RecognitionResult:
        """Forward a recognition request through the compliance proxy."""
        import json as _json

        files = {"image": ("image.bin", image_bytes, "application/octet-stream")}
        params: Dict[str, str] = {}
        if upstream_url:
            params["upstream_url"] = upstream_url
        if metadata:
            params["metadata_json"] = _json.dumps(metadata)
        resp = self._request("POST", "/proxy/recognize", files=files, params=params)
        data = resp.json()
        return RecognitionResult(
            predictions=data.get("matches", data.get("predictions", [])),
            event_ids=data.get("event_ids", []),
            latency_ms=data.get("latency_ms", 0.0),
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def forensic_session(self) -> ForensicSession:
        """Return a context manager that auto-bundles events on exit."""
        return ForensicSession(self)
