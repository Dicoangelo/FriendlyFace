"""Tests for ZK proof implementation."""

import json

from friendlyface.stubs.zk import ZKBundleProver, ZKProof


class TestZKProof:
    def test_generate_returns_json(self):
        zk = ZKProof()
        proof_str = zk.generate("abc123")
        data = json.loads(proof_str)
        assert data["scheme"] == "pedersen-sha256"
        assert data["bundle_hash"] == "abc123"
        assert len(data["nonce"]) == 64  # 32 bytes hex
        assert len(data["commitment"]) == 64  # SHA-256 hex

    def test_verify_valid_proof(self):
        zk = ZKProof()
        proof_str = zk.generate("mybundle")
        assert ZKProof.verify(proof_str) is True

    def test_verify_tampered_proof_fails(self):
        zk = ZKProof()
        proof_str = zk.generate("mybundle")
        data = json.loads(proof_str)
        data["bundle_hash"] = "tampered"
        assert ZKProof.verify(json.dumps(data)) is False

    def test_verify_none_returns_false(self):
        assert ZKProof.verify(None) is False

    def test_verify_empty_string_returns_false(self):
        assert ZKProof.verify("") is False

    def test_verify_invalid_json_returns_false(self):
        assert ZKProof.verify("not json") is False

    def test_verify_legacy_stub_format(self):
        assert ZKProof.verify("zk_stub::test::not_implemented") is True

    def test_placeholder_false_after_generate(self):
        zk = ZKProof()
        zk.generate("test")
        assert zk.placeholder is False

    def test_different_bundles_different_commitments(self):
        zk1 = ZKProof()
        zk2 = ZKProof()
        p1 = json.loads(zk1.generate("bundle_a"))
        p2 = json.loads(zk2.generate("bundle_b"))
        assert p1["commitment"] != p2["commitment"]

    def test_same_bundle_different_nonces(self):
        zk1 = ZKProof()
        zk2 = ZKProof()
        p1 = json.loads(zk1.generate("same"))
        p2 = json.loads(zk2.generate("same"))
        assert p1["nonce"] != p2["nonce"]
        assert p1["commitment"] != p2["commitment"]


class TestZKBundleProver:
    def test_prove_and_verify(self):
        prover = ZKBundleProver()
        prover.prove_bundle("bundle-1", "hash123")
        assert prover.verify_bundle("bundle-1") is True

    def test_verify_unknown_bundle(self):
        prover = ZKBundleProver()
        assert prover.verify_bundle("unknown") is False

    def test_multiple_bundles(self):
        prover = ZKBundleProver()
        prover.prove_bundle("b1", "h1")
        prover.prove_bundle("b2", "h2")
        assert prover.verify_bundle("b1") is True
        assert prover.verify_bundle("b2") is True


class TestBackwardCompat:
    def test_zkproofstub_alias(self):
        from friendlyface.stubs.zk import ZKProofStub

        assert ZKProofStub is ZKProof
