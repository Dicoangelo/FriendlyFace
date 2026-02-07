"""Tests for real Ed25519 DID:key implementation."""

import os

from nacl.signing import VerifyKey

from friendlyface.crypto.did import Ed25519DIDKey


class TestKeyGeneration:
    def test_generates_unique_keys(self):
        k1 = Ed25519DIDKey()
        k2 = Ed25519DIDKey()
        assert k1.did != k2.did

    def test_did_starts_with_did_key_z6Mk(self):
        key = Ed25519DIDKey()
        assert key.did.startswith("did:key:z6Mk")

    def test_deterministic_seed(self):
        seed = b"\x01" * 32
        k1 = Ed25519DIDKey.from_seed(seed)
        k2 = Ed25519DIDKey.from_seed(seed)
        assert k1.did == k2.did

    def test_different_seeds_different_keys(self):
        k1 = Ed25519DIDKey.from_seed(b"\x01" * 32)
        k2 = Ed25519DIDKey.from_seed(b"\x02" * 32)
        assert k1.did != k2.did

    def test_export_public_returns_32_bytes(self):
        key = Ed25519DIDKey()
        pub = key.export_public()
        assert isinstance(pub, bytes)
        assert len(pub) == 32

    def test_export_public_matches_verify_key(self):
        seed = os.urandom(32)
        key = Ed25519DIDKey.from_seed(seed)
        exported = key.export_public()
        reconstructed = VerifyKey(exported)
        # Signing with original, verifying with reconstructed should work
        sig = key.sign(b"round-trip")
        reconstructed.verify(b"round-trip", sig)  # raises on failure


class TestSigning:
    def test_sign_returns_64_bytes(self):
        key = Ed25519DIDKey()
        sig = key.sign(b"hello")
        assert isinstance(sig, bytes)
        assert len(sig) == 64

    def test_sign_deterministic(self):
        key = Ed25519DIDKey()
        assert key.sign(b"data") == key.sign(b"data")

    def test_different_data_different_signatures(self):
        key = Ed25519DIDKey()
        assert key.sign(b"alpha") != key.sign(b"beta")

    def test_different_keys_different_signatures(self):
        k1 = Ed25519DIDKey()
        k2 = Ed25519DIDKey()
        assert k1.sign(b"same") != k2.sign(b"same")


class TestVerification:
    def test_verify_valid_signature(self):
        key = Ed25519DIDKey()
        sig = key.sign(b"forensic event")
        assert key.verify(b"forensic event", sig) is True

    def test_verify_invalid_signature(self):
        key = Ed25519DIDKey()
        assert key.verify(b"data", b"\x00" * 64) is False

    def test_verify_tampered_data(self):
        key = Ed25519DIDKey()
        sig = key.sign(b"original")
        assert key.verify(b"tampered", sig) is False

    def test_verify_wrong_key(self):
        k1 = Ed25519DIDKey()
        k2 = Ed25519DIDKey()
        sig = k1.sign(b"message")
        assert k2.verify(b"message", sig) is False

    def test_verify_empty_data(self):
        key = Ed25519DIDKey()
        sig = key.sign(b"")
        assert key.verify(b"", sig) is True

    def test_verify_large_data(self):
        key = Ed25519DIDKey()
        data = os.urandom(10_000)
        sig = key.sign(data)
        assert key.verify(data, sig) is True


class TestDIDFormat:
    def test_did_prefix(self):
        key = Ed25519DIDKey()
        assert key.did.startswith("did:key:z6Mk")

    def test_did_is_string(self):
        key = Ed25519DIDKey()
        assert isinstance(key.did, str)

    def test_did_consistent_across_calls(self):
        key = Ed25519DIDKey()
        assert key.did == key.did

    def test_did_length_reasonable(self):
        # did:key:z6Mk... — the z6Mk prefix + base58 of 32 bytes
        key = Ed25519DIDKey()
        # "did:key:" is 8 chars, then "z" + base58(0xed01 + 32 bytes)
        assert len(key.did) > 40
        assert len(key.did) < 120


class TestResolve:
    def test_resolve_returns_dict(self):
        key = Ed25519DIDKey()
        doc = key.resolve()
        assert isinstance(doc, dict)

    def test_resolve_has_context(self):
        key = Ed25519DIDKey()
        doc = key.resolve()
        assert "https://www.w3.org/ns/did/v1" in doc["@context"]
        assert "https://w3id.org/security/suites/ed25519-2020/v1" in doc["@context"]

    def test_resolve_id_matches_did(self):
        key = Ed25519DIDKey()
        doc = key.resolve()
        assert doc["id"] == key.did

    def test_resolve_verification_method(self):
        key = Ed25519DIDKey()
        doc = key.resolve()
        vm = doc["verificationMethod"]
        assert len(vm) == 1
        assert vm[0]["type"] == "Ed25519VerificationKey2020"
        assert vm[0]["controller"] == key.did
        assert vm[0]["publicKeyMultibase"].startswith("z")

    def test_resolve_authentication(self):
        key = Ed25519DIDKey()
        doc = key.resolve()
        assert len(doc["authentication"]) == 1
        assert doc["authentication"][0] == doc["verificationMethod"][0]["id"]

    def test_resolve_assertion_method(self):
        key = Ed25519DIDKey()
        doc = key.resolve()
        assert len(doc["assertionMethod"]) == 1
        assert doc["assertionMethod"][0] == doc["verificationMethod"][0]["id"]

    def test_resolve_vm_id_format(self):
        key = Ed25519DIDKey()
        doc = key.resolve()
        vm_id = doc["verificationMethod"][0]["id"]
        # VM id should be did#fragment
        assert vm_id.startswith(key.did + "#")


class TestExportImport:
    def test_exported_key_can_verify(self):
        key = Ed25519DIDKey()
        pub_bytes = key.export_public()
        sig = key.sign(b"cross-check")

        # Reconstruct VerifyKey from exported bytes and verify
        vk = VerifyKey(pub_bytes)
        vk.verify(b"cross-check", sig)  # raises BadSignatureError on failure

    def test_exported_key_rejects_bad_signature(self):
        key = Ed25519DIDKey()
        pub_bytes = key.export_public()
        vk = VerifyKey(pub_bytes)

        from nacl.exceptions import BadSignatureError
        import pytest

        with pytest.raises(BadSignatureError):
            vk.verify(b"data", b"\xff" * 64)
