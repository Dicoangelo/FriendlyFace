"""Tests for DID/VC implementation."""

from friendlyface.stubs.did import DIDKey, VerifiableCredential


class TestDIDKey:
    def test_did_format(self):
        key = DIDKey()
        assert key.did.startswith("did:key:")
        assert len(key.identifier) == 32

    def test_unique_keys(self):
        k1 = DIDKey()
        k2 = DIDKey()
        assert k1.did != k2.did

    def test_resolve_returns_document(self):
        key = DIDKey()
        doc = key.resolve()
        assert doc["id"] == key.did
        assert doc["type"] == "Ed25519VerificationKey2020"
        assert doc["status"] == "active"
        assert doc["controller"] == key.did

    def test_sign_returns_hex(self):
        key = DIDKey()
        sig = key.sign("hello")
        assert len(sig) == 64  # SHA-256 HMAC hex

    def test_sign_deterministic(self):
        key = DIDKey()
        assert key.sign("data") == key.sign("data")

    def test_verify_valid_signature(self):
        key = DIDKey()
        sig = key.sign("test data")
        assert key.verify_signature("test data", sig) is True

    def test_verify_invalid_signature(self):
        key = DIDKey()
        assert key.verify_signature("test", "wrong_signature") is False

    def test_different_keys_different_signatures(self):
        k1 = DIDKey()
        k2 = DIDKey()
        sig1 = k1.sign("same data")
        sig2 = k2.sign("same data")
        assert sig1 != sig2


class TestVerifiableCredential:
    def test_issue_returns_vc_structure(self):
        issuer = DIDKey()
        vc = VerifiableCredential(issuer_did=issuer, subject_did="did:key:subject123")
        cred = vc.issue({"role": "fl_participant", "round": 1})
        assert "VerifiableCredential" in cred["type"]
        assert "ForensicCredential" in cred["type"]
        assert cred["issuer"] == issuer.did
        assert cred["credentialSubject"]["role"] == "fl_participant"
        assert cred["proof"]["type"] == "HmacSha256Signature2024"

    def test_verify_valid_credential(self):
        issuer = DIDKey()
        vc = VerifiableCredential(issuer_did=issuer)
        cred = vc.issue({"event": "training"})
        assert vc.verify(cred) is True

    def test_verify_tampered_credential(self):
        issuer = DIDKey()
        vc = VerifiableCredential(issuer_did=issuer)
        cred = vc.issue({"event": "training"})
        cred["credentialSubject"]["event"] = "tampered"
        assert vc.verify(cred) is False

    def test_verify_missing_proof(self):
        issuer = DIDKey()
        vc = VerifiableCredential(issuer_did=issuer)
        assert vc.verify({"credentialSubject": {}}) is False

    def test_verify_legacy_stub_format(self):
        issuer = DIDKey()
        vc = VerifiableCredential(issuer_did=issuer)
        legacy = {"proof": "stub::not_implemented", "credentialSubject": {}}
        assert vc.verify(legacy) is True

    def test_custom_credential_type(self):
        issuer = DIDKey()
        vc = VerifiableCredential(issuer_did=issuer)
        cred = vc.issue({"x": 1}, credential_type="BiometricCredential")
        assert "BiometricCredential" in cred["type"]

    def test_issuance_date_present(self):
        issuer = DIDKey()
        vc = VerifiableCredential(issuer_did=issuer)
        cred = vc.issue({"test": True})
        assert "issuanceDate" in cred

    def test_w3c_context(self):
        issuer = DIDKey()
        vc = VerifiableCredential(issuer_did=issuer)
        cred = vc.issue({"test": True})
        assert "https://www.w3.org/2018/credentials/v1" in cred["@context"]


class TestBackwardCompat:
    def test_didstub_alias(self):
        from friendlyface.stubs.did import DIDStub

        assert DIDStub is DIDKey

    def test_vcstub_alias(self):
        from friendlyface.stubs.did import VCStub

        assert VCStub is VerifiableCredential
