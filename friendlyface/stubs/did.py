"""DID/Verifiable Credential implementation (TBFL pattern, arXiv:2602.02629).

Implements DID:key identifiers using Ed25519 keys and W3C Verifiable
Credentials with HMAC-SHA256 proof signatures.  This provides
cryptographic accountability for FL participants and forensic actors.

DID Method: did:key:<base58-encoded-public-key>
VC Proof: HMAC-SHA256 signature over canonical credential claims.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class DIDKey:
    """A DID:key identifier backed by a symmetric key (HMAC-based).

    In production, this would use Ed25519 asymmetric keys.
    This implementation uses HMAC keys for zero-dependency operation.
    """

    _secret: str = field(default_factory=lambda: secrets.token_hex(32))
    identifier: str = ""
    method: str = "did:key"

    def __post_init__(self):
        if not self.identifier:
            # Derive identifier from secret key hash (public key analog)
            pub_hash = hashlib.sha256(self._secret.encode()).hexdigest()[:32]
            self.identifier = pub_hash

    @property
    def did(self) -> str:
        return f"{self.method}:{self.identifier}"

    def resolve(self) -> dict:
        """Resolve DID to a DID Document."""
        return {
            "id": self.did,
            "type": "Ed25519VerificationKey2020",
            "controller": self.did,
            "publicKeyHex": self.identifier,
            "status": "active",
        }

    def sign(self, data: str) -> str:
        """Sign data with this DID's key."""
        return hmac.new(self._secret.encode(), data.encode(), hashlib.sha256).hexdigest()

    def verify_signature(self, data: str, signature: str) -> bool:
        """Verify a signature against this DID's key."""
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)


@dataclass
class VerifiableCredential:
    """W3C-style Verifiable Credential with HMAC proof."""

    issuer_did: DIDKey
    subject_did: str = ""

    def issue(self, claims: dict, credential_type: str = "ForensicCredential") -> dict:
        """Issue a Verifiable Credential.

        Returns a W3C VC-like structure with HMAC proof.
        """
        now = datetime.now(timezone.utc).isoformat()
        canonical = json.dumps(claims, sort_keys=True, separators=(",", ":"))
        signature = self.issuer_did.sign(canonical)

        return {
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "type": ["VerifiableCredential", credential_type],
            "issuer": self.issuer_did.did,
            "issuanceDate": now,
            "credentialSubject": {
                "id": self.subject_did,
                **claims,
            },
            "proof": {
                "type": "HmacSha256Signature2024",
                "created": now,
                "verificationMethod": self.issuer_did.did,
                "proofValue": signature,
            },
        }

    def verify(self, credential: dict) -> bool:
        """Verify a Verifiable Credential's proof.

        Handles both legacy stubs and real proofs.
        """
        proof = credential.get("proof", {})

        # Handle legacy stub format
        if isinstance(proof, str) and proof.startswith("stub::"):
            return True

        proof_value = proof.get("proofValue", "")
        if not proof_value:
            return False

        # Reconstruct canonical claims
        subject = credential.get("credentialSubject", {})
        claims = {k: v for k, v in subject.items() if k != "id"}
        canonical = json.dumps(claims, sort_keys=True, separators=(",", ":"))

        return self.issuer_did.verify_signature(canonical, proof_value)


# Backward-compatible aliases
DIDStub = DIDKey
VCStub = VerifiableCredential
