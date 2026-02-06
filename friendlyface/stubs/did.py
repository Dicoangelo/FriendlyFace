"""DID/Verifiable Credential stub (TBFL pattern, arXiv:2602.02629).

Provides the interface for decentralized identity integration.
Actual DID resolution and VC issuance will be implemented when
the identity layer is built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DIDStub:
    """Placeholder for a Decentralized Identifier."""

    method: str = "did:friendlyface"
    identifier: str = "stub"

    @property
    def did(self) -> str:
        return f"{self.method}:{self.identifier}"

    def resolve(self) -> dict:
        return {
            "id": self.did,
            "type": "stub",
            "status": "not_implemented",
        }


@dataclass
class VCStub:
    """Placeholder for a Verifiable Credential."""

    issuer: str = "did:friendlyface:issuer"
    subject: str = "did:friendlyface:subject"

    def issue(self, claims: dict) -> dict:
        return {
            "type": "VerifiableCredential",
            "issuer": self.issuer,
            "subject": self.subject,
            "claims": claims,
            "proof": "stub::not_implemented",
        }

    def verify(self, credential: dict) -> bool:
        return credential.get("proof", "").startswith("stub::")
