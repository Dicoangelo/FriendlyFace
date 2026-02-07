"""Schnorr ZK proof implementation (non-interactive via Fiat-Shamir).

Implements a Schnorr identification protocol converted to non-interactive
using the Fiat-Shamir heuristic with SHA-256.  Used for forensic bundle
verification: prove knowledge of a secret derived from bundle data without
revealing the secret itself.

Protocol:
  Prover picks random k, computes r = g^k mod p
  Challenge c = SHA-256(g || r || y)  where y = g^secret mod p
  Response  s = (k - c * secret) mod q    where q = (p-1)/2
  Verifier checks: g^s * y^c == r  (mod p)

Only Python stdlib + hashlib required -- no external crypto dependencies.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Domain parameters: safe 256-bit prime p where q = (p-1)/2 is also prime.
# Generator g has order q in Z*_p (quadratic residue subgroup).
# ---------------------------------------------------------------------------

# Safe prime: p = 2*q + 1 where both p and q are prime.
# Generated via Miller-Rabin (40 rounds) and verified: g^q mod p == 1.
_P = 0xF4538F15435947859FAE0A53AC1BF6FE4019014EF6F130A72B67032DF8C59B4F
_G = 4  # Generator of the quadratic residue subgroup of order q = (p-1)/2
_Q = (_P - 1) // 2  # Order of the subgroup


def _int_to_hex(n: int) -> str:
    """Convert a non-negative integer to a zero-padded hex string."""
    # Ensure at least 64 hex chars (256 bits) for consistency
    raw = format(n, "x")
    return raw.zfill(64)


def _hex_to_int(h: str) -> int:
    """Convert a hex string to an integer."""
    return int(h, 16)


def _fiat_shamir_challenge(g: int, r: int, y: int, p: int) -> int:
    """Derive deterministic challenge via Fiat-Shamir: c = SHA256(g || r || y).

    All values are serialized as zero-padded 64-char hex strings before hashing
    to ensure canonical encoding.
    """
    payload = f"{_int_to_hex(g)}{_int_to_hex(r)}{_int_to_hex(y)}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest, 16) % _Q


# ---------------------------------------------------------------------------
# Core Schnorr prover / verifier
# ---------------------------------------------------------------------------


@dataclass
class SchnorrProver:
    """Generate non-interactive Schnorr ZK proofs."""

    p: int = _P
    g: int = _G
    q: int = _Q

    def generate_proof(self, secret: int) -> dict:
        """Produce a full non-interactive Schnorr proof for *secret*.

        Returns a dict with keys:
            scheme, commitment, challenge, response, public_point
        All numeric values are hex-encoded strings.
        """
        g, p, q = self.g, self.p, self.q

        # Public point y = g^secret mod p
        y = pow(g, secret, p)

        # Random nonce k in [1, q)
        k = secrets.randbelow(q - 1) + 1

        # Commitment r = g^k mod p
        r = pow(g, k, p)

        # Fiat-Shamir challenge
        c = _fiat_shamir_challenge(g, r, y, p)

        # Response: s = (k - c * secret) mod q
        s = (k - c * secret) % q

        return {
            "scheme": "schnorr-sha256",
            "commitment": _int_to_hex(r),
            "challenge": _int_to_hex(c),
            "response": _int_to_hex(s),
            "public_point": _int_to_hex(y),
        }


@dataclass
class SchnorrVerifier:
    """Verify non-interactive Schnorr ZK proofs."""

    p: int = _P
    g: int = _G
    q: int = _Q

    def verify(self, proof: dict) -> bool:
        """Return True iff the Schnorr proof is valid.

        Checks:
            1. g^s * y^c == r  (mod p)
            2. Challenge c was correctly derived via Fiat-Shamir
        """
        try:
            r = _hex_to_int(proof["commitment"])
            c = _hex_to_int(proof["challenge"])
            s = _hex_to_int(proof["response"])
            y = _hex_to_int(proof["public_point"])
        except (KeyError, ValueError):
            return False

        g, p = self.g, self.p

        # Re-derive the challenge to ensure Fiat-Shamir integrity
        expected_c = _fiat_shamir_challenge(g, r, y, p)
        if c != expected_c:
            return False

        # Core Schnorr check: g^s * y^c == r  (mod p)
        lhs = (pow(g, s, p) * pow(y, c, p)) % p
        return lhs == r


# ---------------------------------------------------------------------------
# Bundle-specific wrappers (backward-compatible with stubs/zk.py)
# ---------------------------------------------------------------------------


def _derive_secret(bundle_hash: str) -> int:
    """Derive a deterministic integer secret from a bundle hash string."""
    digest = hashlib.sha256(bundle_hash.encode()).hexdigest()
    return int(digest, 16) % _Q


def _compute_legacy_commitment(nonce: str, bundle_hash: str) -> str:
    """Compute SHA-256(nonce || bundle_hash) -- legacy Pedersen-SHA256 scheme."""
    return hashlib.sha256(f"{nonce}{bundle_hash}".encode()).hexdigest()


@dataclass
class ZKBundleProver:
    """Generates Schnorr ZK proofs for forensic bundles."""

    proofs: dict[str, str] = field(default_factory=dict)

    def prove_bundle(self, bundle_id: str, bundle_hash: str) -> str:
        """Create a Schnorr ZK proof for a specific bundle.

        Derives a secret from *bundle_hash* via SHA-256, generates a
        non-interactive Schnorr proof, and returns it as a JSON string.
        """
        secret = _derive_secret(bundle_hash)
        prover = SchnorrProver()
        proof = prover.generate_proof(secret)
        proof_str = json.dumps(proof)
        self.proofs[bundle_id] = proof_str
        return proof_str


@dataclass
class ZKBundleVerifier:
    """Verifies Schnorr ZK proofs for forensic bundles.

    Backward-compatible with legacy formats:
      - ``zk_stub::`` prefix  -> True
      - ``pedersen-sha256`` scheme -> verify nonce+hash commitment
    """

    def verify_bundle(self, proof_str: str) -> bool:
        """Verify a bundle proof string.

        Handles three formats:
          1. Legacy ``zk_stub::*`` prefix -- always True
          2. Legacy ``pedersen-sha256`` JSON -- SHA-256 commitment check
          3. ``schnorr-sha256`` JSON -- full Schnorr verification
        """
        if proof_str is None:
            return False

        # Legacy stub format
        if isinstance(proof_str, str) and proof_str.startswith("zk_stub::"):
            return True

        try:
            data = json.loads(proof_str)
        except (json.JSONDecodeError, TypeError):
            return False

        scheme = data.get("scheme", "")

        # Legacy Pedersen-SHA256 commitment
        if scheme == "pedersen-sha256":
            nonce = data.get("nonce", "")
            bundle_hash = data.get("bundle_hash", "")
            commitment = data.get("commitment", "")
            if not all([nonce, bundle_hash, commitment]):
                return False
            return _compute_legacy_commitment(nonce, bundle_hash) == commitment

        # Schnorr proof
        if scheme == "schnorr-sha256":
            verifier = SchnorrVerifier()
            return verifier.verify(data)

        return False
