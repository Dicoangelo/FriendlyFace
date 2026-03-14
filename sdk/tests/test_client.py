"""Tests for FriendlyFaceClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from friendlyface_sdk.client import FriendlyFaceClient
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


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


class TestClientInit:
    def test_base_url_normalization(self):
        with patch("friendlyface_sdk.client.requests.Session"):
            c = FriendlyFaceClient("http://example.com/")
            assert c.base_url == "http://example.com"

    def test_base_url_no_trailing_slash(self):
        with patch("friendlyface_sdk.client.requests.Session"):
            c = FriendlyFaceClient("http://example.com")
            assert c.base_url == "http://example.com"

    def test_auth_header_set(self, client, mock_session):
        mock_session.headers.__setitem__.assert_called_with(
            "Authorization", "Bearer test-key"
        )

    def test_no_auth_header_when_no_key(self, mock_session):
        with patch("friendlyface_sdk.client.requests.Session") as cls:
            inst = MagicMock()
            cls.return_value = inst
            c = FriendlyFaceClient("http://localhost:8000")
            assert c.api_key is None


class TestLogEvent:
    def test_log_event(self, client, mock_session):
        mock_session.request.return_value = _mock_response({
            "id": "evt-1",
            "event_type": "training_start",
            "event_hash": "abc123",
            "timestamp": "2026-01-01T00:00:00Z",
        })
        result = client.log_event("training_start", {"actor": "sdk", "model": "v1"})
        assert isinstance(result, ForensicEvent)
        assert result.event_id == "evt-1"
        assert result.event_type == "training_start"
        assert result.event_hash == "abc123"
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[0] == ("POST", "http://localhost:8000/events")


class TestCheckConsent:
    def test_check_consent(self, client, mock_session):
        mock_session.request.return_value = _mock_response({
            "subject_id": "subj-1",
            "has_consent": True,
            "purpose": "recognition",
            "granted_at": "2026-01-01T00:00:00Z",
        })
        result = client.check_consent("subj-1")
        assert isinstance(result, ConsentStatus)
        assert result.subject_id == "subj-1"
        assert result.has_consent is True
        call_args = mock_session.request.call_args
        assert call_args[0] == ("GET", "http://localhost:8000/consent/status/subj-1")


class TestGrantConsent:
    def test_grant_consent(self, client, mock_session):
        mock_session.request.return_value = _mock_response({
            "id": "cons-1",
            "subject_id": "subj-1",
            "purpose": "recognition",
            "actor": "admin",
            "granted_at": "2026-01-01T00:00:00Z",
        })
        result = client.grant_consent("subj-1", "recognition", granted_by="admin")
        assert isinstance(result, ConsentRecord)
        assert result.consent_id == "cons-1"
        assert result.subject_id == "subj-1"
        call_args = mock_session.request.call_args
        assert call_args[0] == ("POST", "http://localhost:8000/consent/grant")


class TestCreateBundle:
    def test_create_bundle(self, client, mock_session):
        mock_session.request.return_value = _mock_response({
            "id": "bnd-1",
            "event_ids": ["evt-1", "evt-2"],
            "merkle_root": "root123",
            "zk_proof": {"proof": "data"},
            "credential": {"type": "vc"},
        })
        result = client.create_bundle(["evt-1", "evt-2"])
        assert isinstance(result, Bundle)
        assert result.bundle_id == "bnd-1"
        assert result.merkle_root == "root123"
        assert result.zk_proof == {"proof": "data"}


class TestIssueSeal:
    def test_issue_seal(self, client, mock_session):
        mock_session.request.return_value = _mock_response({
            "seal_id": "seal-1",
            "credential": {"type": "ForensicSeal"},
            "verification_url": "http://example.com/verify/seal-1",
            "expires_at": "2026-06-01T00:00:00Z",
            "compliance_summary": {"score": 95},
        })
        result = client.issue_seal("sys-1", "TestSystem")
        assert isinstance(result, Seal)
        assert result.seal_id == "seal-1"
        assert result.verification_url == "http://example.com/verify/seal-1"
        call_args = mock_session.request.call_args
        assert call_args[1]["json"]["system_id"] == "sys-1"


class TestVerifySeal:
    def test_verify_seal_by_credential(self, client, mock_session):
        mock_session.request.return_value = _mock_response({
            "valid": True,
            "checks": {"signature": True, "expiry": True},
            "issuer": "did:key:abc",
            "issued_at": "2026-01-01T00:00:00Z",
            "expires_at": "2026-06-01T00:00:00Z",
            "compliance_score": 92.5,
        })
        cred = {"type": "ForensicSeal", "proof": {}}
        result = client.verify_seal(credential=cred)
        assert isinstance(result, VerificationResult)
        assert result.valid is True
        assert result.compliance_score == 92.5
        call_args = mock_session.request.call_args
        assert call_args[0] == ("POST", "http://localhost:8000/seal/verify")

    def test_verify_seal_by_id(self, client, mock_session):
        mock_session.request.return_value = _mock_response({
            "valid": True,
            "checks": {},
            "issuer": "",
            "issued_at": "",
            "expires_at": "",
            "compliance_score": 0.0,
        })
        result = client.verify_seal(seal_id="seal-1")
        assert isinstance(result, VerificationResult)
        call_args = mock_session.request.call_args
        assert call_args[0] == ("GET", "http://localhost:8000/seal/verify/seal-1")

    def test_verify_seal_raises_without_args(self, client):
        import pytest

        with pytest.raises(ValueError, match="Provide either"):
            client.verify_seal()


class TestRunAudit:
    def test_run_audit(self, client, mock_session):
        mock_session.request.return_value = _mock_response({
            "audit_id": "aud-1",
            "demographic_parity": 0.05,
            "equalized_odds": 0.03,
            "pass": True,
        })
        result = client.run_audit(
            predictions=[{"label": "a"}],
            demographics=[
                {"group_name": "g1", "true_positives": 10, "false_positives": 1,
                 "true_negatives": 80, "false_negatives": 2},
                {"group_name": "g2", "true_positives": 9, "false_positives": 2,
                 "true_negatives": 78, "false_negatives": 3},
            ],
        )
        assert isinstance(result, AuditResult)
        assert result.audit_id == "aud-1"
        assert result.pass_status is True


class TestProxyRecognize:
    def test_proxy_recognize(self, client, mock_session):
        mock_session.request.return_value = _mock_response({
            "matches": [{"label": "person_a", "confidence": 0.95}],
            "event_ids": ["evt-10", "evt-11"],
            "latency_ms": 42.5,
        })
        result = client.proxy_recognize(b"\x89PNG...", upstream_url="http://upstream/api")
        assert isinstance(result, RecognitionResult)
        assert len(result.predictions) == 1
        assert result.predictions[0]["confidence"] == 0.95
        assert result.event_ids == ["evt-10", "evt-11"]
        assert result.latency_ms == 42.5
