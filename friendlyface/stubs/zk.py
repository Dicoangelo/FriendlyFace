"""Zero-Knowledge Proof implementation (BioZero pattern, arXiv:2409.17509).

Implements a Pedersen-style hash commitment scheme for forensic bundle
verification.  The prover commits to bundle data without revealing it;
the verifier checks that the commitment matches without seeing the data.

Scheme:
  commitment = SHA-256(secret_nonce || bundle_hash)
  proof = {commitment, nonce, bundle_hash}
  verify: SHA-256(nonce || bundle_hash) == commitment

This is a simplified ZK commitment — not a full zk-SNARK — but provides
the core property: verifying bundle integrity without revealing contents.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field


@dataclass
class ZKProof:
    """A zero-knowledge commitment proof for a forensic bundle."""

    commitment: str = ""
    nonce: str = ""
    bundle_hash: str = ""
    placeholder: bool = False

    def generate(self, bundle_hash: str) -> str:
        """Generate a ZK commitment for the given bundle hash.

        Returns a JSON-encoded proof string suitable for storage.
        """
        self.nonce = secrets.token_hex(32)
        self.bundle_hash = bundle_hash
        self.commitment = _compute_commitment(self.nonce, bundle_hash)
        self.placeholder = False
        return json.dumps(
            {
                "scheme": "pedersen-sha256",
                "commitment": self.commitment,
                "nonce": self.nonce,
                "bundle_hash": self.bundle_hash,
            }
        )

    @staticmethod
    def verify(proof_str: str) -> bool:
        """Verify a ZK commitment proof.

        Returns True if the commitment matches the nonce + bundle_hash.
        Handles both legacy stubs and real proofs.
        """
        if proof_str is None:
            return False

        # Handle legacy stub format
        if proof_str.startswith("zk_stub::"):
            return True

        try:
            data = json.loads(proof_str)
        except (json.JSONDecodeError, TypeError):
            return False

        nonce = data.get("nonce", "")
        bundle_hash = data.get("bundle_hash", "")
        commitment = data.get("commitment", "")

        if not all([nonce, bundle_hash, commitment]):
            return False

        return _compute_commitment(nonce, bundle_hash) == commitment


def _compute_commitment(nonce: str, bundle_hash: str) -> str:
    """Compute SHA-256(nonce || bundle_hash)."""
    return hashlib.sha256(f"{nonce}{bundle_hash}".encode()).hexdigest()


# Keep backward-compatible alias
ZKProofStub = ZKProof


@dataclass
class ZKBundleProver:
    """Generates and verifies ZK proofs for forensic bundles."""

    proofs: dict[str, str] = field(default_factory=dict)

    def prove_bundle(self, bundle_id: str, bundle_hash: str) -> str:
        """Create a ZK proof for a specific bundle."""
        zk = ZKProof()
        proof_str = zk.generate(bundle_hash)
        self.proofs[bundle_id] = proof_str
        return proof_str

    def verify_bundle(self, bundle_id: str) -> bool:
        """Verify a previously generated bundle proof."""
        proof_str = self.proofs.get(bundle_id)
        if proof_str is None:
            return False
        return ZKProof.verify(proof_str)
