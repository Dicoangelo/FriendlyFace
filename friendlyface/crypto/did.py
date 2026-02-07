"""Real Ed25519 DID:key implementation using PyNaCl.

Replaces the HMAC-based stub with actual asymmetric cryptography.
DID Method: did:key:z6Mk<base58-encoded-public-key>
Signatures: Ed25519 (RFC 8032) via libsodium/PyNaCl.
"""

from __future__ import annotations

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

# Base58 alphabet (Bitcoin variant) — no external dependency needed
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base58_encode(data: bytes) -> str:
    """Encode bytes to base58 (Bitcoin alphabet)."""
    # Count leading zero bytes
    n_leading = 0
    for byte in data:
        if byte == 0:
            n_leading += 1
        else:
            break

    # Convert bytes to integer
    num = int.from_bytes(data, "big")

    # Encode integer to base58
    result = bytearray()
    while num > 0:
        num, remainder = divmod(num, 58)
        result.append(_B58_ALPHABET[remainder])
    result.reverse()

    # Prepend '1' for each leading zero byte
    return ("1" * n_leading) + result.decode("ascii")


def _base58_decode(s: str) -> bytes:
    """Decode base58 string to bytes."""
    # Count leading '1' characters
    n_leading = 0
    for ch in s:
        if ch == "1":
            n_leading += 1
        else:
            break

    # Convert base58 to integer
    num = 0
    for ch in s:
        idx = _B58_ALPHABET.index(ch.encode("ascii"))
        num = num * 58 + idx

    # Convert integer to bytes
    if num == 0:
        result = b""
    else:
        result = num.to_bytes((num.bit_length() + 7) // 8, "big")

    return b"\x00" * n_leading + result


# Ed25519-pub multicodec prefix: 0xed01
_ED25519_MULTICODEC_PREFIX = b"\xed\x01"


class Ed25519DIDKey:
    """A DID:key identifier backed by a real Ed25519 keypair.

    Generates or accepts an Ed25519 signing key, derives the DID from the
    public key using the did:key multicodec method (z6Mk prefix), and
    provides sign/verify operations with real cryptographic signatures.
    """

    def __init__(self, signing_key: SigningKey | None = None) -> None:
        self._signing_key = signing_key or SigningKey.generate()
        self._verify_key: VerifyKey = self._signing_key.verify_key

    @classmethod
    def from_seed(cls, seed: bytes) -> Ed25519DIDKey:
        """Create a deterministic key from a 32-byte seed."""
        return cls(signing_key=SigningKey(seed))

    @property
    def did(self) -> str:
        """Return the did:key identifier (z6Mk-prefixed multibase/multicodec)."""
        raw = _ED25519_MULTICODEC_PREFIX + bytes(self._verify_key)
        return f"did:key:z{_base58_encode(raw)}"

    def resolve(self) -> dict:
        """Resolve DID to a W3C DID Document."""
        did = self.did
        vm_id = f"{did}#{did.split(':')[-1]}"
        return {
            "@context": [
                "https://www.w3.org/ns/did/v1",
                "https://w3id.org/security/suites/ed25519-2020/v1",
            ],
            "id": did,
            "verificationMethod": [
                {
                    "id": vm_id,
                    "type": "Ed25519VerificationKey2020",
                    "controller": did,
                    "publicKeyMultibase": f"z{_base58_encode(bytes(self._verify_key))}",
                },
            ],
            "authentication": [vm_id],
            "assertionMethod": [vm_id],
        }

    def sign(self, data: bytes) -> bytes:
        """Produce an Ed25519 signature over *data*."""
        # SignedMessage contains sig + message; .signature is just the 64-byte sig
        return self._signing_key.sign(data).signature

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify an Ed25519 *signature* over *data* against this key's public key."""
        try:
            self._verify_key.verify(data, signature)
            return True
        except BadSignatureError:
            return False

    def export_public(self) -> bytes:
        """Export the raw 32-byte Ed25519 verify (public) key."""
        return bytes(self._verify_key)
