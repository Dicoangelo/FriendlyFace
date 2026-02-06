"""Consent record management for the FriendlyFace platform.

Implements append-only consent tracking with forensic event logging.
Consent records are never deleted or updated in-place; every state change
(grant, revoke) appends a new record, preserving full audit history.

Consent is checked before any recognition inference on a subject.
Revoked or expired consent blocks future inference and logs the block.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from friendlyface.core.models import EventType
from friendlyface.core.service import ForensicService
from friendlyface.storage.database import Database


class ConsentError(Exception):
    """Raised when a consent check fails (revoked, expired, or missing)."""


class ConsentRecord:
    """Value object representing a single consent record."""

    def __init__(
        self,
        *,
        id: str,
        subject_id: str,
        purpose: str,
        granted: bool,
        timestamp: str,
        expiry: str | None = None,
        revocation_reason: str | None = None,
        event_id: str | None = None,
    ) -> None:
        self.id = id
        self.subject_id = subject_id
        self.purpose = purpose
        self.granted = granted
        self.timestamp = timestamp
        self.expiry = expiry
        self.revocation_reason = revocation_reason
        self.event_id = event_id

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if this consent record has passed its expiry time."""
        if self.expiry is None:
            return False
        now = now or datetime.now(timezone.utc)
        expiry_dt = datetime.fromisoformat(self.expiry)
        return now >= expiry_dt

    def is_active(self, now: datetime | None = None) -> bool:
        """Check if consent is currently active (granted and not expired)."""
        return self.granted and not self.is_expired(now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "purpose": self.purpose,
            "granted": self.granted,
            "timestamp": self.timestamp,
            "expiry": self.expiry,
            "revocation_reason": self.revocation_reason,
            "event_id": self.event_id,
        }


class ConsentManager:
    """Manages consent records with forensic event logging.

    All consent changes are:
    1. Appended to the consent_records table (never updated/deleted)
    2. Logged as ForensicEvent(event_type=EventType.CONSENT_UPDATE)
    """

    def __init__(self, db: Database, forensic_service: ForensicService) -> None:
        self.db = db
        self.forensic_service = forensic_service

    async def grant_consent(
        self,
        subject_id: str,
        purpose: str,
        *,
        expiry: datetime | None = None,
        actor: str = "system",
    ) -> ConsentRecord:
        """Grant consent for a subject+purpose pair.

        Appends a new consent record and logs a CONSENT_UPDATE forensic event.
        """
        record_id = str(uuid4())
        now = datetime.now(timezone.utc)
        expiry_str = expiry.isoformat() if expiry else None

        # Log the forensic event first
        event = await self.forensic_service.record_event(
            event_type=EventType.CONSENT_UPDATE,
            actor=actor,
            payload={
                "action": "grant",
                "subject_id": subject_id,
                "purpose": purpose,
                "consent_record_id": record_id,
                "expiry": expiry_str,
            },
        )

        # Append consent record
        await self.db.insert_consent_record(
            record_id=record_id,
            subject_id=subject_id,
            purpose=purpose,
            granted=True,
            timestamp=now.isoformat(),
            expiry=expiry_str,
            revocation_reason=None,
            event_id=str(event.id),
        )

        return ConsentRecord(
            id=record_id,
            subject_id=subject_id,
            purpose=purpose,
            granted=True,
            timestamp=now.isoformat(),
            expiry=expiry_str,
            event_id=str(event.id),
        )

    async def revoke_consent(
        self,
        subject_id: str,
        purpose: str,
        *,
        reason: str = "",
        actor: str = "system",
    ) -> ConsentRecord:
        """Revoke consent for a subject+purpose pair.

        Appends a new consent record with granted=False and logs a
        CONSENT_UPDATE forensic event.
        """
        record_id = str(uuid4())
        now = datetime.now(timezone.utc)

        # Log the forensic event
        event = await self.forensic_service.record_event(
            event_type=EventType.CONSENT_UPDATE,
            actor=actor,
            payload={
                "action": "revoke",
                "subject_id": subject_id,
                "purpose": purpose,
                "consent_record_id": record_id,
                "revocation_reason": reason,
            },
        )

        # Append consent record
        await self.db.insert_consent_record(
            record_id=record_id,
            subject_id=subject_id,
            purpose=purpose,
            granted=False,
            timestamp=now.isoformat(),
            expiry=None,
            revocation_reason=reason or None,
            event_id=str(event.id),
        )

        return ConsentRecord(
            id=record_id,
            subject_id=subject_id,
            purpose=purpose,
            granted=False,
            timestamp=now.isoformat(),
            revocation_reason=reason or None,
            event_id=str(event.id),
        )

    async def check_consent(
        self,
        subject_id: str,
        purpose: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Check if active consent exists for a subject+purpose pair.

        Returns True only if the latest consent record is granted and not expired.
        Returns False if no record exists, consent is revoked, or consent has expired.
        """
        row = await self.db.get_latest_consent(subject_id, purpose)
        if row is None:
            return False
        record = ConsentRecord(**row)
        return record.is_active(now)

    async def require_consent(
        self,
        subject_id: str,
        purpose: str,
        *,
        actor: str = "system",
        now: datetime | None = None,
    ) -> None:
        """Require active consent, raising ConsentError and logging if blocked.

        Use this before inference operations. If consent is missing, revoked,
        or expired, a CONSENT_UPDATE event is logged recording the block,
        and ConsentError is raised.
        """
        row = await self.db.get_latest_consent(subject_id, purpose)

        if row is None:
            # No consent on file — block
            await self.forensic_service.record_event(
                event_type=EventType.CONSENT_UPDATE,
                actor=actor,
                payload={
                    "action": "block",
                    "subject_id": subject_id,
                    "purpose": purpose,
                    "reason": "no_consent_record",
                },
            )
            raise ConsentError(
                f"No consent record found for subject={subject_id} purpose={purpose}"
            )

        record = ConsentRecord(**row)

        if not record.granted:
            # Consent explicitly revoked — block
            await self.forensic_service.record_event(
                event_type=EventType.CONSENT_UPDATE,
                actor=actor,
                payload={
                    "action": "block",
                    "subject_id": subject_id,
                    "purpose": purpose,
                    "reason": "consent_revoked",
                },
            )
            raise ConsentError(
                f"Consent revoked for subject={subject_id} purpose={purpose}"
            )

        if record.is_expired(now):
            # Consent has expired — block
            await self.forensic_service.record_event(
                event_type=EventType.CONSENT_UPDATE,
                actor=actor,
                payload={
                    "action": "block",
                    "subject_id": subject_id,
                    "purpose": purpose,
                    "reason": "consent_expired",
                    "expired_at": record.expiry,
                },
            )
            raise ConsentError(
                f"Consent expired for subject={subject_id} purpose={purpose}"
            )

    async def get_consent_status(
        self,
        subject_id: str,
        purpose: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Get detailed consent status for a subject+purpose pair."""
        row = await self.db.get_latest_consent(subject_id, purpose)
        if row is None:
            return {
                "subject_id": subject_id,
                "purpose": purpose,
                "has_consent": False,
                "granted": False,
                "expired": False,
                "active": False,
                "record": None,
            }

        record = ConsentRecord(**row)
        return {
            "subject_id": subject_id,
            "purpose": purpose,
            "has_consent": True,
            "granted": record.granted,
            "expired": record.is_expired(now),
            "active": record.is_active(now),
            "record": record.to_dict(),
        }

    async def get_history(
        self,
        subject_id: str,
        purpose: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get full consent history for a subject, optionally filtered by purpose."""
        rows = await self.db.get_consent_history(subject_id, purpose)
        return rows
