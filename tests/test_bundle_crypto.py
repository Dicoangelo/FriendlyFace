"""Tests for US-030: DID + ZK integration into ForensicBundle lifecycle.

Verifies that:
- Bundle creation generates real Schnorr ZK proofs and DID-signed VCs
- Bundle verification validates ZK proofs and DID credentials
- Tampered proofs are detected
- Legacy bundles (None / zk_stub:: values) remain backward-compatible
- FF_DID_SEED env var controls deterministic key generation
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

from friendlyface.core.models import EventType, ForensicBundle
from friendlyface.core.service import ForensicService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _record_event_and_create_bundle(service: ForensicService) -> ForensicBundle:
    """Record a single event and wrap it in a forensic bundle."""
    event = await service.record_event(
        event_type=EventType.INFERENCE_RESULT,
        actor="test-actor",
        payload={"prediction": "subject-1", "confidence": 0.95},
    )
    bundle = await service.create_bundle(event_ids=[event.id])
    return bundle


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBundleZKProof:
    """ZK proof generation and verification on bundles."""

    async def test_create_bundle_generates_zk_proof(self, service: ForensicService):
        """Bundle creation should produce a non-None Schnorr ZK proof."""
        bundle = await _record_event_and_create_bundle(service)

        assert bundle.zk_proof_placeholder is not None
        # The proof should be valid JSON containing the schnorr-sha256 scheme
        proof_data = json.loads(bundle.zk_proof_placeholder)
        assert proof_data["scheme"] == "schnorr-sha256"
        assert "commitment" in proof_data
        assert "challenge" in proof_data
        assert "response" in proof_data
        assert "public_point" in proof_data

    async def test_create_bundle_generates_did_credential(self, service: ForensicService):
        """Bundle creation should produce a non-None DID Verifiable Credential."""
        bundle = await _record_event_and_create_bundle(service)

        assert bundle.did_credential_placeholder is not None
        cred = json.loads(bundle.did_credential_placeholder)
        assert "VerifiableCredential" in cred["type"]
        assert "ForensicCredential" in cred["type"]
        assert cred["issuer"].startswith("did:key:z")
        assert cred["proof"]["type"] == "Ed25519Signature2020"
        # Credential should reference the bundle
        assert cred["credentialSubject"]["bundle_id"] == str(bundle.id)
        assert cred["credentialSubject"]["bundle_hash"] == bundle.bundle_hash

    async def test_verify_bundle_returns_zk_and_did_valid(self, service: ForensicService):
        """verify_bundle() should return zk_valid=True and did_valid=True for valid bundles."""
        bundle = await _record_event_and_create_bundle(service)
        result = await service.verify_bundle(bundle.id)

        assert result["valid"] is True
        assert result["zk_valid"] is True
        assert result["did_valid"] is True
        assert result["credential_issuer"] is not None
        assert result["credential_issuer"].startswith("did:key:z")

    async def test_tampered_zk_proof_detected(self, service: ForensicService):
        """Modifying the ZK proof should cause zk_valid=False."""
        bundle = await _record_event_and_create_bundle(service)

        # Tamper with the proof: flip a character in the commitment
        proof_data = json.loads(bundle.zk_proof_placeholder)
        original_commitment = proof_data["commitment"]
        # Flip a hex digit
        tampered = original_commitment[:-1] + ("0" if original_commitment[-1] != "0" else "1")
        proof_data["commitment"] = tampered
        tampered_proof_str = json.dumps(proof_data)

        # Directly update the DB row to simulate tampering
        await service.db.db.execute(
            "UPDATE forensic_bundles SET zk_proof_placeholder = ? WHERE id = ?",
            (tampered_proof_str, str(bundle.id)),
        )
        await service.db.db.commit()

        result = await service.verify_bundle(bundle.id)
        assert result["zk_valid"] is False
        assert result["valid"] is False

    async def test_tampered_did_credential_detected(self, service: ForensicService):
        """Modifying the DID credential proof should cause did_valid=False."""
        bundle = await _record_event_and_create_bundle(service)

        # Tamper with the credential: corrupt the proofValue
        cred = json.loads(bundle.did_credential_placeholder)
        original_pv = cred["proof"]["proofValue"]
        cred["proof"]["proofValue"] = "00" * len(original_pv)  # all zeros
        tampered_cred_str = json.dumps(cred)

        await service.db.db.execute(
            "UPDATE forensic_bundles SET did_credential_placeholder = ? WHERE id = ?",
            (tampered_cred_str, str(bundle.id)),
        )
        await service.db.db.commit()

        result = await service.verify_bundle(bundle.id)
        assert result["did_valid"] is False
        assert result["valid"] is False


class TestLegacyBundleCompat:
    """Legacy bundles with None or stub values should still verify."""

    async def test_legacy_zk_proof_none(self, service: ForensicService):
        """Bundle with zk_proof_placeholder=None should verify zk_valid=True."""
        bundle = await _record_event_and_create_bundle(service)

        # Overwrite to None in DB
        await service.db.db.execute(
            "UPDATE forensic_bundles SET zk_proof_placeholder = NULL WHERE id = ?",
            (str(bundle.id),),
        )
        await service.db.db.commit()

        result = await service.verify_bundle(bundle.id)
        assert result["zk_valid"] is True

    async def test_legacy_zk_proof_stub(self, service: ForensicService):
        """Bundle with zk_stub::... value should verify zk_valid=True."""
        bundle = await _record_event_and_create_bundle(service)

        await service.db.db.execute(
            "UPDATE forensic_bundles SET zk_proof_placeholder = ? WHERE id = ?",
            ("zk_stub::legacy-proof", str(bundle.id)),
        )
        await service.db.db.commit()

        result = await service.verify_bundle(bundle.id)
        assert result["zk_valid"] is True

    async def test_legacy_did_credential_none(self, service: ForensicService):
        """Bundle with did_credential_placeholder=None should verify did_valid=True."""
        bundle = await _record_event_and_create_bundle(service)

        await service.db.db.execute(
            "UPDATE forensic_bundles SET did_credential_placeholder = NULL WHERE id = ?",
            (str(bundle.id),),
        )
        await service.db.db.commit()

        result = await service.verify_bundle(bundle.id)
        assert result["did_valid"] is True
        assert result["credential_issuer"] is None

    async def test_legacy_did_credential_stub(self, service: ForensicService):
        """Bundle with stub::... value should verify did_valid=True."""
        bundle = await _record_event_and_create_bundle(service)

        await service.db.db.execute(
            "UPDATE forensic_bundles SET did_credential_placeholder = ? WHERE id = ?",
            ("stub::legacy-vc", str(bundle.id)),
        )
        await service.db.db.commit()

        result = await service.verify_bundle(bundle.id)
        assert result["did_valid"] is True
        assert result["credential_issuer"] is None


class TestPlatformDIDKey:
    """Platform DID key loading and determinism."""

    async def test_auto_generated_key_works(self, db):
        """Without FF_DID_SEED, service should auto-generate a working key."""
        with patch.dict(os.environ, {}, clear=False):
            # Ensure FF_DID_SEED is not set
            os.environ.pop("FF_DID_SEED", None)
            svc = ForensicService(db)
            await svc.initialize()

        assert svc._platform_did is not None
        assert svc._platform_did.did.startswith("did:key:z")

        # The generated key should work for sign/verify
        data = b"test-data"
        sig = svc._platform_did.sign(data)
        assert svc._platform_did.verify(data, sig) is True

    async def test_ff_did_seed_deterministic(self, db):
        """FF_DID_SEED env var should produce a deterministic key."""
        seed_hex = "a" * 64  # 32 bytes of 0xaa
        with patch.dict(os.environ, {"FF_DID_SEED": seed_hex}):
            svc1 = ForensicService(db)
            svc2 = ForensicService(db)

        # Both services should have the same DID
        assert svc1._platform_did.did == svc2._platform_did.did
        assert svc1._platform_did.export_public() == svc2._platform_did.export_public()

    async def test_different_seeds_different_keys(self, db):
        """Different seeds should produce different keys."""
        with patch.dict(os.environ, {"FF_DID_SEED": "a" * 64}):
            svc1 = ForensicService(db)
        with patch.dict(os.environ, {"FF_DID_SEED": "b" * 64}):
            svc2 = ForensicService(db)

        assert svc1._platform_did.did != svc2._platform_did.did


class TestFieldAliases:
    """ForensicBundle field aliases for zk_proof and did_credential."""

    def test_alias_zk_proof(self):
        """zk_proof alias should set zk_proof_placeholder."""
        bundle = ForensicBundle(event_ids=[], zk_proof="test-proof")
        assert bundle.zk_proof_placeholder == "test-proof"

    def test_alias_did_credential(self):
        """did_credential alias should set did_credential_placeholder."""
        bundle = ForensicBundle(event_ids=[], did_credential="test-cred")
        assert bundle.did_credential_placeholder == "test-cred"

    def test_field_name_still_works(self):
        """Setting by field name (populate_by_name) should still work."""
        bundle = ForensicBundle(
            event_ids=[],
            zk_proof_placeholder="via-field-name",
            did_credential_placeholder="via-field-name-2",
        )
        assert bundle.zk_proof_placeholder == "via-field-name"
        assert bundle.did_credential_placeholder == "via-field-name-2"


class TestBundleHashExcludesCryptoFields:
    """Verify that compute_hash() is unaffected by ZK/DID fields."""

    def test_hash_same_regardless_of_zk_and_did(self):
        """Bundle hash should not change when zk_proof or did_credential are set."""
        bundle = ForensicBundle(event_ids=[]).seal()
        original_hash = bundle.bundle_hash
        # Set crypto fields after sealing
        bundle.zk_proof_placeholder = "some-proof"
        bundle.did_credential_placeholder = "some-cred"
        # Recompute should still match the sealed hash (crypto fields are external)
        assert bundle.verify() is True
        assert bundle.compute_hash() == original_hash
