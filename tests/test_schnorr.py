"""Tests for Schnorr ZK proof implementation."""

import json

from friendlyface.crypto.schnorr import (
    SchnorrProver,
    SchnorrVerifier,
    ZKBundleProver,
    ZKBundleVerifier,
    _derive_secret,
    _fiat_shamir_challenge,
    _hex_to_int,
    _P,
    _G,
    _Q,
)
from friendlyface.stubs.zk import ZKProof


# ---------------------------------------------------------------------------
# Core Schnorr protocol tests
# ---------------------------------------------------------------------------


class TestSchnorrProver:
    def test_proof_structure(self):
        prover = SchnorrProver()
        proof = prover.generate_proof(42)
        assert set(proof.keys()) == {
            "scheme",
            "commitment",
            "challenge",
            "response",
            "public_point",
        }
        assert proof["scheme"] == "schnorr-sha256"

    def test_proof_values_are_hex_strings(self):
        prover = SchnorrProver()
        proof = prover.generate_proof(99)
        for key in ("commitment", "challenge", "response", "public_point"):
            val = proof[key]
            assert isinstance(val, str)
            int(val, 16)  # must parse as hex without error

    def test_deterministic_public_point(self):
        """Same secret always yields the same public_point (g^secret mod p)."""
        prover = SchnorrProver()
        p1 = prover.generate_proof(123)
        p2 = prover.generate_proof(123)
        assert p1["public_point"] == p2["public_point"]

    def test_different_secrets_different_public_points(self):
        prover = SchnorrProver()
        p1 = prover.generate_proof(10)
        p2 = prover.generate_proof(20)
        assert p1["public_point"] != p2["public_point"]

    def test_commitment_varies_across_calls(self):
        """Each proof uses a fresh random k, so commitments differ."""
        prover = SchnorrProver()
        p1 = prover.generate_proof(42)
        p2 = prover.generate_proof(42)
        # Extremely unlikely to collide with random k
        assert p1["commitment"] != p2["commitment"]


class TestSchnorrVerifier:
    def test_valid_proof(self):
        prover = SchnorrProver()
        verifier = SchnorrVerifier()
        proof = prover.generate_proof(42)
        assert verifier.verify(proof) is True

    def test_tampered_challenge_fails(self):
        prover = SchnorrProver()
        verifier = SchnorrVerifier()
        proof = prover.generate_proof(42)
        proof["challenge"] = "00" * 32
        assert verifier.verify(proof) is False

    def test_tampered_response_fails(self):
        prover = SchnorrProver()
        verifier = SchnorrVerifier()
        proof = prover.generate_proof(42)
        proof["response"] = "ff" * 32
        assert verifier.verify(proof) is False

    def test_tampered_commitment_fails(self):
        prover = SchnorrProver()
        verifier = SchnorrVerifier()
        proof = prover.generate_proof(42)
        proof["commitment"] = "ab" * 32
        assert verifier.verify(proof) is False

    def test_tampered_public_point_fails(self):
        prover = SchnorrProver()
        verifier = SchnorrVerifier()
        proof = prover.generate_proof(42)
        proof["public_point"] = "01" * 32
        assert verifier.verify(proof) is False

    def test_missing_key_returns_false(self):
        verifier = SchnorrVerifier()
        assert verifier.verify({"scheme": "schnorr-sha256"}) is False

    def test_invalid_hex_returns_false(self):
        verifier = SchnorrVerifier()
        proof = {
            "scheme": "schnorr-sha256",
            "commitment": "not_hex",
            "challenge": "ab" * 32,
            "response": "ab" * 32,
            "public_point": "ab" * 32,
        }
        assert verifier.verify(proof) is False

    def test_various_secrets(self):
        """Verify proofs for a range of secret values."""
        prover = SchnorrProver()
        verifier = SchnorrVerifier()
        for secret in (0, 1, 2, 255, 1000, 2**128, _Q - 1):
            proof = prover.generate_proof(secret)
            assert verifier.verify(proof) is True, f"Failed for secret={secret}"


class TestFiatShamir:
    def test_challenge_derived_from_commitment_and_public_point(self):
        """Challenge must be deterministically derived from (g, r, y)."""
        prover = SchnorrProver()
        proof = prover.generate_proof(77)

        r = _hex_to_int(proof["commitment"])
        y = _hex_to_int(proof["public_point"])
        expected_c = _fiat_shamir_challenge(_G, r, y, _P)

        assert _hex_to_int(proof["challenge"]) == expected_c

    def test_different_commitments_yield_different_challenges(self):
        prover = SchnorrProver()
        p1 = prover.generate_proof(77)
        p2 = prover.generate_proof(77)
        # Same secret but different random k => different r => different c
        assert p1["challenge"] != p2["challenge"]


# ---------------------------------------------------------------------------
# Bundle prover / verifier tests
# ---------------------------------------------------------------------------


class TestZKBundleProver:
    def test_round_trip(self):
        prover = ZKBundleProver()
        verifier = ZKBundleVerifier()
        proof_str = prover.prove_bundle("bundle-1", "hash123")
        assert verifier.verify_bundle(proof_str) is True

    def test_proof_is_valid_json(self):
        prover = ZKBundleProver()
        proof_str = prover.prove_bundle("b1", "h1")
        data = json.loads(proof_str)
        assert data["scheme"] == "schnorr-sha256"

    def test_stores_proof_internally(self):
        prover = ZKBundleProver()
        proof_str = prover.prove_bundle("b1", "h1")
        assert prover.proofs["b1"] == proof_str

    def test_multiple_bundles(self):
        prover = ZKBundleProver()
        verifier = ZKBundleVerifier()
        for i in range(5):
            bid = f"bundle-{i}"
            proof_str = prover.prove_bundle(bid, f"hash-{i}")
            assert verifier.verify_bundle(proof_str) is True

    def test_same_bundle_hash_same_public_point(self):
        """Deterministic secret derivation means same hash -> same public_point."""
        prover = ZKBundleProver()
        p1 = json.loads(prover.prove_bundle("b1", "same_hash"))
        p2 = json.loads(prover.prove_bundle("b2", "same_hash"))
        assert p1["public_point"] == p2["public_point"]

    def test_different_bundle_hashes_different_public_points(self):
        prover = ZKBundleProver()
        p1 = json.loads(prover.prove_bundle("b1", "hash_a"))
        p2 = json.loads(prover.prove_bundle("b2", "hash_b"))
        assert p1["public_point"] != p2["public_point"]


class TestZKBundleVerifier:
    def test_none_returns_false(self):
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle(None) is False

    def test_empty_string_returns_false(self):
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle("") is False

    def test_invalid_json_returns_false(self):
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle("not json at all") is False

    def test_unknown_scheme_returns_false(self):
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle(json.dumps({"scheme": "unknown"})) is False

    def test_invalid_json_type_returns_false(self):
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle(123) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Backward compatibility with legacy formats
# ---------------------------------------------------------------------------


class TestLegacyCompat:
    def test_zk_stub_prefix_accepted(self):
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle("zk_stub::bundle-42::not_implemented") is True

    def test_zk_stub_various_formats(self):
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle("zk_stub::") is True
        assert verifier.verify_bundle("zk_stub::test") is True
        assert verifier.verify_bundle("zk_stub::a::b::c") is True

    def test_pedersen_sha256_from_stubs_zk(self):
        """Proofs generated by the legacy stubs/zk.py must verify."""
        zk = ZKProof()
        legacy_proof_str = zk.generate("test_bundle_hash")
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle(legacy_proof_str) is True

    def test_pedersen_sha256_tampered_fails(self):
        zk = ZKProof()
        legacy_proof_str = zk.generate("test_bundle_hash")
        data = json.loads(legacy_proof_str)
        data["bundle_hash"] = "tampered"
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle(json.dumps(data)) is False

    def test_pedersen_sha256_missing_fields_fails(self):
        verifier = ZKBundleVerifier()
        proof = json.dumps({"scheme": "pedersen-sha256", "nonce": "abc"})
        assert verifier.verify_bundle(proof) is False

    def test_pedersen_sha256_manual_proof(self):
        """Manually constructed pedersen-sha256 proof verifies correctly."""
        import hashlib

        nonce = "deadbeef" * 8
        bundle_hash = "my_bundle"
        commitment = hashlib.sha256(f"{nonce}{bundle_hash}".encode()).hexdigest()
        proof = json.dumps(
            {
                "scheme": "pedersen-sha256",
                "commitment": commitment,
                "nonce": nonce,
                "bundle_hash": bundle_hash,
            }
        )
        verifier = ZKBundleVerifier()
        assert verifier.verify_bundle(proof) is True


# ---------------------------------------------------------------------------
# Multiple sequential proofs
# ---------------------------------------------------------------------------


class TestSequentialProofs:
    def test_many_sequential_proofs(self):
        """Generate and verify many proofs in sequence."""
        prover = SchnorrProver()
        verifier = SchnorrVerifier()
        for i in range(20):
            proof = prover.generate_proof(i * 137 + 1)
            assert verifier.verify(proof) is True

    def test_bundle_sequential(self):
        prover = ZKBundleProver()
        verifier = ZKBundleVerifier()
        for i in range(10):
            proof_str = prover.prove_bundle(f"seq-{i}", f"hash-{i}")
            assert verifier.verify_bundle(proof_str) is True
        assert len(prover.proofs) == 10


# ---------------------------------------------------------------------------
# Secret derivation
# ---------------------------------------------------------------------------


class TestDeriveSecret:
    def test_deterministic(self):
        assert _derive_secret("hello") == _derive_secret("hello")

    def test_different_inputs_different_secrets(self):
        assert _derive_secret("a") != _derive_secret("b")

    def test_result_in_range(self):
        s = _derive_secret("test")
        assert 0 <= s < _Q
