"""Compliance proxy that wraps any facial recognition API with forensic logging."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import httpx

from friendlyface.core.models import EventType
from friendlyface.core.service import ForensicService
from friendlyface.storage.database import Database

logger = logging.getLogger("friendlyface.proxy")


class ComplianceProxy:
    """Forwards recognition requests to upstream APIs while logging forensic events."""

    def __init__(
        self,
        forensic_service: ForensicService,
        db: Database,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.forensic_service = forensic_service
        self.db = db
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

    async def recognize(
        self,
        image_bytes: bytes,
        upstream_url: str,
        upstream_headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward recognition request to upstream, log forensic events."""
        metadata = metadata or {}
        input_hash = hashlib.sha256(image_bytes).hexdigest()

        # 1. Check consent if subject_id provided
        subject_id = metadata.get("subject_id")
        consent_status: dict[str, Any] | None = None
        if subject_id:
            consent_status = await self._check_consent(subject_id)

        # 2. Log inference_request event
        request_event = await self.forensic_service.record_event(
            event_type=EventType.INFERENCE_REQUEST,
            actor="compliance_proxy",
            payload={
                "upstream_url": upstream_url,
                "input_hash": input_hash,
                "subject_id": subject_id,
                "consent_checked": consent_status is not None,
                "consent_allowed": consent_status.get("allowed") if consent_status else None,
                "metadata": metadata,
            },
        )

        # 3. Forward to upstream
        start_time = time.monotonic()
        headers = dict(upstream_headers or {})

        try:
            upstream_response = await self._client.post(
                upstream_url,
                content=image_bytes,
                headers=headers,
            )
            latency_ms = (time.monotonic() - start_time) * 1000

            upstream_data = (
                upstream_response.json() if upstream_response.status_code == 200 else {}
            )
            upstream_status = upstream_response.status_code
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.warning("Upstream request failed: %s", exc)
            upstream_data = {}
            upstream_status = 502

            # Log failure as forensic event
            await self.forensic_service.record_event(
                event_type=EventType.INFERENCE_RESULT,
                actor="compliance_proxy",
                payload={
                    "upstream_url": upstream_url,
                    "upstream_status": upstream_status,
                    "input_hash": input_hash,
                    "latency_ms": round(latency_ms, 2),
                    "subject_id": subject_id,
                    "error": str(exc),
                },
            )

            return {
                "upstream_response": {},
                "upstream_status": upstream_status,
                "forensic": {
                    "request_event_id": str(request_event.id),
                    "result_event_id": None,
                    "input_hash": input_hash,
                    "latency_ms": round(latency_ms, 2),
                    "consent_checked": consent_status is not None,
                    "consent_allowed": (
                        consent_status.get("allowed") if consent_status else None
                    ),
                    "error": str(exc),
                },
            }

        # 4. Log inference_result event
        result_event = await self.forensic_service.record_event(
            event_type=EventType.INFERENCE_RESULT,
            actor="compliance_proxy",
            payload={
                "upstream_url": upstream_url,
                "upstream_status": upstream_status,
                "input_hash": input_hash,
                "latency_ms": round(latency_ms, 2),
                "subject_id": subject_id,
                "response_summary": _summarize_response(upstream_data),
            },
        )

        return {
            "upstream_response": upstream_data,
            "upstream_status": upstream_status,
            "forensic": {
                "request_event_id": str(request_event.id),
                "result_event_id": str(result_event.id),
                "input_hash": input_hash,
                "latency_ms": round(latency_ms, 2),
                "consent_checked": consent_status is not None,
                "consent_allowed": (
                    consent_status.get("allowed") if consent_status else None
                ),
            },
        }

    async def _check_consent(self, subject_id: str) -> dict[str, Any] | None:
        """Check consent status for a subject."""
        from friendlyface.governance.consent import ConsentManager

        mgr = ConsentManager(self.db, self.forensic_service)
        allowed = await mgr.check_consent(subject_id, "recognition")
        status = await mgr.get_consent_status(subject_id, "recognition")
        return {
            "subject_id": subject_id,
            "purpose": "recognition",
            "allowed": allowed,
            "has_consent": status["has_consent"],
            "active": status["active"],
        }

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _summarize_response(data: dict[str, Any]) -> dict[str, Any]:
    """Extract key fields from upstream response for forensic logging (avoid storing PII)."""
    summary: dict[str, Any] = {}
    if "matches" in data:
        summary["match_count"] = len(data["matches"])
    if "confidence" in data:
        summary["confidence"] = data["confidence"]
    if "face_count" in data:
        summary["face_count"] = data["face_count"]
    return summary
