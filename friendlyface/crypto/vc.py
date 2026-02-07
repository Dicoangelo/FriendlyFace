"""Verifiable Credentials with Ed25519 signatures (W3C VC Data Model).

Issues tamper-evident credentials signed with Ed25519 (PyNaCl/libsodium).
Replaces the HMAC-based stub with real asymmetric cryptographic proofs.

Proof type: Ed25519Signature2020
Signing input: canonical JSON (sorted keys, no whitespace) of credential claims.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from friendlyface.crypto.did import Ed25519DIDKey


def _canonical_json(data: dict) -> str:
    """Deterministic JSON serialization for signing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


class VerifiableCredential:
    """W3C-style Verifiable Credential with Ed25519Signature2020 proof."""

    def __init__(self, issuer: Ed25519DIDKey) -> None:
        self._issuer = issuer

    def issue(
        self,
        claims: dict,
        credential_type: str = "ForensicCredential",
        subject_did: str = "",
    ) -> dict:
        """Issue a signed Verifiable Credential.

        Returns a W3C VC structure with an Ed25519Signature2020 proof.
        """
        now = datetime.now(timezone.utc).isoformat()
        canonical = _canonical_json(claims)
        signature = self._issuer.sign(canonical.encode())

        return {
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "type": ["VerifiableCredential", credential_type],
            "issuer": self._issuer.did,
            "issuanceDate": now,
            "credentialSubject": {
                "id": subject_did,
                **claims,
            },
            "proof": {
                "type": "Ed25519Signature2020",
                "created": now,
                "verificationMethod": self._issuer.did,
                "proofValue": signature.hex(),
            },
        }

    @staticmethod
    def verify(credential: dict, issuer_public_key: bytes) -> dict:
        """Verify a Verifiable Credential's proof.

        Returns a dict with {valid: bool, legacy: bool, credential_type: str}.
        Handles:
          - Ed25519Signature2020 proofs (real verification)
          - Legacy stub:: proofs (accepted with deprecation flag)
          - Legacy HmacSha256Signature2024 proofs (accepted with deprecation flag)
        """
        proof = credential.get("proof", {})
        cred_types = credential.get("type", [])
        cred_type = cred_types[-1] if cred_types else "Unknown"

        # Handle legacy stub format (string proof)
        if isinstance(proof, str) and proof.startswith("stub::"):
            return {"valid": True, "legacy": True, "credential_type": cred_type}

        # Handle legacy HMAC proof
        proof_type = proof.get("type", "")
        if proof_type == "HmacSha256Signature2024":
            return {"valid": True, "legacy": True, "credential_type": cred_type}

        # Ed25519 verification
        if proof_type != "Ed25519Signature2020":
            return {"valid": False, "legacy": False, "credential_type": cred_type}

        proof_value = proof.get("proofValue", "")
        if not proof_value:
            return {"valid": False, "legacy": False, "credential_type": cred_type}

        # Reconstruct canonical claims for verification
        subject = credential.get("credentialSubject", {})
        claims = {k: v for k, v in subject.items() if k != "id"}
        canonical = _canonical_json(claims)

        try:
            signature = bytes.fromhex(proof_value)
            from nacl.signing import VerifyKey

            vk = VerifyKey(issuer_public_key)
            vk.verify(canonical.encode(), signature)
            return {"valid": True, "legacy": False, "credential_type": cred_type}
        except Exception:
            return {"valid": False, "legacy": False, "credential_type": cred_type}
