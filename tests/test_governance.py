"""Tests for the governance consent management system (US-015)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from friendlyface.core.models import EventType
from friendlyface.core.service import ForensicService
from friendlyface.governance.consent import ConsentError, ConsentManager, ConsentRecord
from friendlyface.storage.database import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    """Fresh database for each test."""
    database = Database(tmp_path / "consent_test.db")
    await database.connect()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def service(db):
    """Fresh forensic service for each test."""
    svc = ForensicService(db)
    await svc.initialize()
    return svc


@pytest_asyncio.fixture
async def consent_mgr(db, service):
    """ConsentManager wired to fresh db + forensic service."""
    return ConsentManager(db, service)


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------


class TestConsentUpdateEventType:
    def test_consent_update_enum_exists(self):
        """CONSENT_UPDATE must be present in EventType."""
        assert hasattr(EventType, "CONSENT_UPDATE")
        assert EventType.CONSENT_UPDATE.value == "consent_update"


# ---------------------------------------------------------------------------
# ConsentRecord value object
# ---------------------------------------------------------------------------


class TestConsentRecord:
    def test_not_expired_when_no_expiry(self):
        record = ConsentRecord(
            id="r1",
            subject_id="subj1",
            purpose="recognition",
            granted=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        assert not record.is_expired()
        assert record.is_active()

    def test_expired_when_past_expiry(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        record = ConsentRecord(
            id="r1",
            subject_id="subj1",
            purpose="recognition",
            granted=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            expiry=past,
        )
        assert record.is_expired()
        assert not record.is_active()

    def test_not_expired_when_future_expiry(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        record = ConsentRecord(
            id="r1",
            subject_id="subj1",
            purpose="recognition",
            granted=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            expiry=future,
        )
        assert not record.is_expired()
        assert record.is_active()

    def test_revoked_not_active(self):
        record = ConsentRecord(
            id="r1",
            subject_id="subj1",
            purpose="recognition",
            granted=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        assert not record.is_active()

    def test_to_dict(self):
        record = ConsentRecord(
            id="r1",
            subject_id="subj1",
            purpose="recognition",
            granted=True,
            timestamp="2025-01-01T00:00:00+00:00",
            expiry="2026-01-01T00:00:00+00:00",
            revocation_reason=None,
            event_id="evt1",
        )
        d = record.to_dict()
        assert d["id"] == "r1"
        assert d["subject_id"] == "subj1"
        assert d["granted"] is True
        assert d["event_id"] == "evt1"


# ---------------------------------------------------------------------------
# Grant consent
# ---------------------------------------------------------------------------


class TestGrantConsent:
    async def test_grant_creates_record(self, consent_mgr):
        record = await consent_mgr.grant_consent(
            "subj1", "recognition", actor="test"
        )
        assert record.subject_id == "subj1"
        assert record.purpose == "recognition"
        assert record.granted is True
        assert record.event_id is not None

    async def test_grant_logs_forensic_event(self, consent_mgr, service):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        events = await service.get_all_events()
        consent_events = [
            e for e in events if e.event_type == EventType.CONSENT_UPDATE
        ]
        assert len(consent_events) == 1
        assert consent_events[0].payload["action"] == "grant"
        assert consent_events[0].payload["subject_id"] == "subj1"
        assert consent_events[0].actor == "test"

    async def test_grant_with_expiry(self, consent_mgr):
        expiry = datetime.now(timezone.utc) + timedelta(days=30)
        record = await consent_mgr.grant_consent(
            "subj1", "recognition", expiry=expiry, actor="test"
        )
        assert record.expiry is not None
        assert not record.is_expired()

    async def test_grant_stored_in_db(self, consent_mgr, db):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        row = await db.get_latest_consent("subj1", "recognition")
        assert row is not None
        assert row["granted"] is True
        assert row["subject_id"] == "subj1"


# ---------------------------------------------------------------------------
# Revoke consent
# ---------------------------------------------------------------------------


class TestRevokeConsent:
    async def test_revoke_creates_record(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        record = await consent_mgr.revoke_consent(
            "subj1", "recognition", reason="withdrawn", actor="test"
        )
        assert record.granted is False
        assert record.revocation_reason == "withdrawn"

    async def test_revoke_logs_forensic_event(self, consent_mgr, service):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.revoke_consent(
            "subj1", "recognition", reason="withdrawn", actor="test"
        )
        events = await service.get_all_events()
        revoke_events = [
            e
            for e in events
            if e.event_type == EventType.CONSENT_UPDATE
            and e.payload.get("action") == "revoke"
        ]
        assert len(revoke_events) == 1
        assert revoke_events[0].payload["revocation_reason"] == "withdrawn"

    async def test_revoke_overrides_grant(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        assert await consent_mgr.check_consent("subj1", "recognition") is True

        await consent_mgr.revoke_consent("subj1", "recognition", actor="test")
        assert await consent_mgr.check_consent("subj1", "recognition") is False


# ---------------------------------------------------------------------------
# Check consent
# ---------------------------------------------------------------------------


class TestCheckConsent:
    async def test_no_record_returns_false(self, consent_mgr):
        result = await consent_mgr.check_consent("unknown", "recognition")
        assert result is False

    async def test_granted_returns_true(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        assert await consent_mgr.check_consent("subj1", "recognition") is True

    async def test_revoked_returns_false(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.revoke_consent("subj1", "recognition", actor="test")
        assert await consent_mgr.check_consent("subj1", "recognition") is False

    async def test_expired_returns_false(self, consent_mgr):
        past_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
        await consent_mgr.grant_consent(
            "subj1", "recognition", expiry=past_expiry, actor="test"
        )
        assert await consent_mgr.check_consent("subj1", "recognition") is False

    async def test_not_expired_returns_true(self, consent_mgr):
        future_expiry = datetime.now(timezone.utc) + timedelta(days=30)
        await consent_mgr.grant_consent(
            "subj1", "recognition", expiry=future_expiry, actor="test"
        )
        assert await consent_mgr.check_consent("subj1", "recognition") is True

    async def test_different_purposes_independent(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        assert await consent_mgr.check_consent("subj1", "recognition") is True
        assert await consent_mgr.check_consent("subj1", "training") is False

    async def test_regrant_after_revoke(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.revoke_consent("subj1", "recognition", actor="test")
        assert await consent_mgr.check_consent("subj1", "recognition") is False

        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        assert await consent_mgr.check_consent("subj1", "recognition") is True


# ---------------------------------------------------------------------------
# Require consent (blocks inference)
# ---------------------------------------------------------------------------


class TestRequireConsent:
    async def test_passes_when_granted(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        # Should not raise
        await consent_mgr.require_consent("subj1", "recognition", actor="test")

    async def test_raises_when_no_consent(self, consent_mgr):
        with pytest.raises(ConsentError, match="No consent record found"):
            await consent_mgr.require_consent(
                "subj1", "recognition", actor="test"
            )

    async def test_raises_when_revoked(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.revoke_consent("subj1", "recognition", actor="test")
        with pytest.raises(ConsentError, match="Consent revoked"):
            await consent_mgr.require_consent(
                "subj1", "recognition", actor="test"
            )

    async def test_raises_when_expired(self, consent_mgr):
        past_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
        await consent_mgr.grant_consent(
            "subj1", "recognition", expiry=past_expiry, actor="test"
        )
        with pytest.raises(ConsentError, match="Consent expired"):
            await consent_mgr.require_consent(
                "subj1", "recognition", actor="test"
            )

    async def test_block_logs_forensic_event_no_consent(
        self, consent_mgr, service
    ):
        with pytest.raises(ConsentError):
            await consent_mgr.require_consent(
                "subj1", "recognition", actor="test"
            )
        events = await service.get_all_events()
        block_events = [
            e
            for e in events
            if e.event_type == EventType.CONSENT_UPDATE
            and e.payload.get("action") == "block"
        ]
        assert len(block_events) == 1
        assert block_events[0].payload["reason"] == "no_consent_record"

    async def test_block_logs_forensic_event_revoked(
        self, consent_mgr, service
    ):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.revoke_consent("subj1", "recognition", actor="test")
        with pytest.raises(ConsentError):
            await consent_mgr.require_consent(
                "subj1", "recognition", actor="test"
            )
        events = await service.get_all_events()
        block_events = [
            e
            for e in events
            if e.event_type == EventType.CONSENT_UPDATE
            and e.payload.get("action") == "block"
        ]
        assert len(block_events) == 1
        assert block_events[0].payload["reason"] == "consent_revoked"

    async def test_block_logs_forensic_event_expired(
        self, consent_mgr, service
    ):
        past_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
        await consent_mgr.grant_consent(
            "subj1", "recognition", expiry=past_expiry, actor="test"
        )
        with pytest.raises(ConsentError):
            await consent_mgr.require_consent(
                "subj1", "recognition", actor="test"
            )
        events = await service.get_all_events()
        block_events = [
            e
            for e in events
            if e.event_type == EventType.CONSENT_UPDATE
            and e.payload.get("action") == "block"
        ]
        assert len(block_events) == 1
        assert block_events[0].payload["reason"] == "consent_expired"


# ---------------------------------------------------------------------------
# Consent status
# ---------------------------------------------------------------------------


class TestConsentStatus:
    async def test_status_no_record(self, consent_mgr):
        status = await consent_mgr.get_consent_status("subj1", "recognition")
        assert status["has_consent"] is False
        assert status["active"] is False
        assert status["record"] is None

    async def test_status_active(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        status = await consent_mgr.get_consent_status("subj1", "recognition")
        assert status["has_consent"] is True
        assert status["granted"] is True
        assert status["active"] is True
        assert status["expired"] is False

    async def test_status_revoked(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.revoke_consent("subj1", "recognition", actor="test")
        status = await consent_mgr.get_consent_status("subj1", "recognition")
        assert status["has_consent"] is True
        assert status["granted"] is False
        assert status["active"] is False

    async def test_status_expired(self, consent_mgr):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        await consent_mgr.grant_consent(
            "subj1", "recognition", expiry=past, actor="test"
        )
        status = await consent_mgr.get_consent_status("subj1", "recognition")
        assert status["has_consent"] is True
        assert status["granted"] is True
        assert status["expired"] is True
        assert status["active"] is False


# ---------------------------------------------------------------------------
# History (append-only)
# ---------------------------------------------------------------------------


class TestConsentHistory:
    async def test_empty_history(self, consent_mgr):
        history = await consent_mgr.get_history("subj1")
        assert history == []

    async def test_history_preserves_all_records(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.revoke_consent("subj1", "recognition", actor="test")
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")

        history = await consent_mgr.get_history("subj1", "recognition")
        assert len(history) == 3
        assert history[0]["granted"] is True
        assert history[1]["granted"] is False
        assert history[2]["granted"] is True

    async def test_history_multiple_purposes(self, consent_mgr):
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.grant_consent("subj1", "training", actor="test")

        # All purposes
        all_history = await consent_mgr.get_history("subj1")
        assert len(all_history) == 2

        # Filtered by purpose
        recog = await consent_mgr.get_history("subj1", "recognition")
        assert len(recog) == 1
        assert recog[0]["purpose"] == "recognition"

    async def test_history_never_deletes(self, consent_mgr, db):
        """Verify that revoking does not delete the original grant record."""
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.revoke_consent("subj1", "recognition", actor="test")

        history = await db.get_consent_history("subj1", "recognition")
        assert len(history) == 2
        # Original grant still present
        assert history[0]["granted"] is True
        # Revocation appended
        assert history[1]["granted"] is False


# ---------------------------------------------------------------------------
# Expiry edge cases
# ---------------------------------------------------------------------------


class TestConsentExpiry:
    async def test_check_with_custom_now(self, consent_mgr):
        """Consent valid at one time, expired at another."""
        expiry = datetime(2025, 6, 1, tzinfo=timezone.utc)
        await consent_mgr.grant_consent(
            "subj1", "recognition", expiry=expiry, actor="test"
        )

        before_expiry = datetime(2025, 5, 1, tzinfo=timezone.utc)
        assert (
            await consent_mgr.check_consent(
                "subj1", "recognition", now=before_expiry
            )
            is True
        )

        after_expiry = datetime(2025, 7, 1, tzinfo=timezone.utc)
        assert (
            await consent_mgr.check_consent(
                "subj1", "recognition", now=after_expiry
            )
            is False
        )

    async def test_require_with_custom_now_expired(self, consent_mgr):
        expiry = datetime(2025, 6, 1, tzinfo=timezone.utc)
        await consent_mgr.grant_consent(
            "subj1", "recognition", expiry=expiry, actor="test"
        )
        after = datetime(2025, 7, 1, tzinfo=timezone.utc)
        with pytest.raises(ConsentError, match="Consent expired"):
            await consent_mgr.require_consent(
                "subj1", "recognition", actor="test", now=after
            )


# ---------------------------------------------------------------------------
# Forensic event chain integrity
# ---------------------------------------------------------------------------


class TestForensicChainIntegrity:
    async def test_all_consent_events_in_chain(self, consent_mgr, service):
        """Every consent operation must produce a valid hash-chained event."""
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")
        await consent_mgr.revoke_consent("subj1", "recognition", actor="test")

        # Two CONSENT_UPDATE events expected
        events = await service.get_all_events()
        consent_events = [
            e for e in events if e.event_type == EventType.CONSENT_UPDATE
        ]
        assert len(consent_events) == 2

        # All events must verify
        for event in events:
            assert event.verify(), f"Event {event.id} hash verification failed"

        # Chain integrity check
        integrity = await service.verify_chain_integrity()
        assert integrity["valid"] is True

    async def test_consent_events_chained_correctly(
        self, consent_mgr, service
    ):
        """Consent events should be properly hash-chained with other events."""
        # Record a non-consent event first
        await service.record_event(
            event_type=EventType.TRAINING_START,
            actor="test",
            payload={"model": "pca"},
        )
        # Then consent
        await consent_mgr.grant_consent("subj1", "recognition", actor="test")

        events = await service.get_all_events()
        assert len(events) == 2
        # Consent event's previous_hash should reference the training event
        assert events[1].previous_hash == events[0].event_hash
