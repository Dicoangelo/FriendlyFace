"""Tests for Ed25519-backed Verifiable Credentials."""

from __future__ import annotations

import json

import pytest

from friendlyface.crypto.did import Ed25519DIDKey
from friendlyface.crypto.vc import VerifiableCredential


@pytest.fixture
def issuer():
    """Deterministic issuer DID for reproducible tests."""
    return Ed25519DIDKey.from_seed(b"\x01" * 32)


@pytest.fixture
def vc(issuer):
    return VerifiableCredential(issuer)


class TestIssuance:
    def test_issue_returns_vc_structure(self, vc, issuer):
        cred = vc.issue({"action": "recognition", "bundle_id": "b123"})
        assert cred["@context"] == ["https://www.w3.org/2018/credentials/v1"]
        assert "VerifiableCredential" in cred["type"]
        assert "ForensicCredential" in cred["type"]
        assert cred["issuer"] == issuer.did
        assert "issuanceDate" in cred
        assert cred["credentialSubject"]["action"] == "recognition"
        assert cred["credentialSubject"]["bundle_id"] == "b123"

    def test_issue_with_custom_type(self, vc):
        cred = vc.issue({"role": "participant"}, credential_type="FLParticipantCredential")
        assert "FLParticipantCredential" in cred["type"]

    def test_issue_with_subject_did(self, vc):
        subject = Ed25519DIDKey.from_seed(b"\x02" * 32)
        cred = vc.issue({"role": "auditor"}, subject_did=subject.did)
        assert cred["credentialSubject"]["id"] == subject.did

    def test_proof_is_ed25519(self, vc):
        cred = vc.issue({"test": True})
        proof = cred["proof"]
        assert proof["type"] == "Ed25519Signature2020"
        assert "created" in proof
        assert "verificationMethod" in proof
        assert len(proof["proofValue"]) == 128  # 64 bytes hex-encoded


class TestVerification:
    def test_verify_valid_credential(self, vc, issuer):
        cred = vc.issue({"action": "test"})
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is True
        assert result["legacy"] is False
        assert result["credential_type"] == "ForensicCredential"

    def test_verify_tampered_claims(self, vc, issuer):
        cred = vc.issue({"action": "test"})
        cred["credentialSubject"]["action"] = "tampered"
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is False

    def test_verify_tampered_proof(self, vc, issuer):
        cred = vc.issue({"action": "test"})
        cred["proof"]["proofValue"] = "00" * 64
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is False

    def test_verify_wrong_key(self, vc):
        cred = vc.issue({"action": "test"})
        other = Ed25519DIDKey.from_seed(b"\xff" * 32)
        result = VerifiableCredential.verify(cred, other.export_public())
        assert result["valid"] is False

    def test_verify_empty_proof(self, issuer):
        cred = {"type": ["VerifiableCredential"], "proof": {"type": "Ed25519Signature2020"}}
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is False

    def test_verify_unknown_proof_type(self, issuer):
        cred = {"type": ["VerifiableCredential"], "proof": {"type": "UnknownType2024"}}
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is False


class TestLegacyCompat:
    def test_legacy_stub_proof(self, issuer):
        cred = {"type": ["VerifiableCredential", "ForensicCredential"], "proof": "stub::abc123"}
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is True
        assert result["legacy"] is True

    def test_legacy_hmac_proof(self, issuer):
        cred = {
            "type": ["VerifiableCredential", "ForensicCredential"],
            "proof": {"type": "HmacSha256Signature2024", "proofValue": "abc123"},
        }
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is True
        assert result["legacy"] is True


class TestMultipleCredentialTypes:
    def test_forensic_credential(self, vc, issuer):
        cred = vc.issue({"bundle_id": "b1"}, credential_type="ForensicCredential")
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is True
        assert result["credential_type"] == "ForensicCredential"

    def test_fl_participant_credential(self, vc, issuer):
        cred = vc.issue({"client_id": "c1"}, credential_type="FLParticipantCredential")
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is True
        assert result["credential_type"] == "FLParticipantCredential"

    def test_audit_credential(self, vc, issuer):
        cred = vc.issue({"audit_id": "a1"}, credential_type="AuditCredential")
        result = VerifiableCredential.verify(cred, issuer.export_public())
        assert result["valid"] is True
        assert result["credential_type"] == "AuditCredential"


class TestRoundTrip:
    def test_sign_serialize_verify(self, vc, issuer):
        """Round-trip: issue → JSON serialize → deserialize → verify."""
        cred = vc.issue({"action": "roundtrip", "value": 42})
        serialized = json.dumps(cred)
        deserialized = json.loads(serialized)
        result = VerifiableCredential.verify(deserialized, issuer.export_public())
        assert result["valid"] is True

    def test_multiple_sequential_credentials(self, vc, issuer):
        """Issue and verify multiple credentials from the same issuer."""
        for i in range(5):
            cred = vc.issue({"index": i})
            result = VerifiableCredential.verify(cred, issuer.export_public())
            assert result["valid"] is True, f"Credential {i} failed verification"
