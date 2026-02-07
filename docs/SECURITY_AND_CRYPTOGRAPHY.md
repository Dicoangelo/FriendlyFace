# FriendlyFace — Security & Cryptography

## Overview

FriendlyFace implements multiple layers of cryptographic security to ensure that forensic evidence is tamper-proof, verifiable, and attributable. This document covers every cryptographic mechanism in the system.

---

## Hash Chain (Layer 3 Foundation)

Every `ForensicEvent` is linked to its predecessor via SHA-256 hash chaining.

### How It Works

```
Event 0: previous_hash = "GENESIS"
         event_hash = SHA256(canonical_json(content) + "GENESIS")

Event 1: previous_hash = Event 0.event_hash
         event_hash = SHA256(canonical_json(content) + Event 0.event_hash)

Event N: previous_hash = Event (N-1).event_hash
         event_hash = SHA256(canonical_json(content) + Event (N-1).event_hash)
```

### Key Properties

- **Canonical JSON:** Payload is serialized with `json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)` — deterministic output regardless of key insertion order.
- **Seal mechanism:** `ForensicEvent.seal()` computes the hash and locks it. Post-sealing, any field modification makes `verify()` return False.
- **Chain verification:** `GET /chain/integrity` walks every event from 0 to N, checking that each event's `previous_hash` matches the preceding event's `event_hash` and that each event's own hash is correct.

### Tamper Detection

If any event in the chain is modified:
- Its `event_hash` no longer matches the recomputed hash → detected
- All subsequent events have an incorrect `previous_hash` → cascade detection
- The Merkle tree root changes → additional detection layer

---

## Merkle Tree

An append-only binary Merkle tree provides O(log n) inclusion proofs.

### Structure

```
                    Root
                   /    \
                H(01)   H(23)
               / \       / \
            H(0) H(1) H(2) H(3)   ← leaf hashes = event hashes
```

### Operations

- **Add leaf:** `merkle.add_leaf(event_hash)` — O(1) amortized
- **Get root:** `merkle.root_hash` — O(1)
- **Inclusion proof:** Given a leaf index, returns the sibling hashes needed to recompute the root — O(log n)

### Verification

Anyone with an event hash and a Merkle proof can verify inclusion:

```python
# Start with the leaf hash
current = event_hash

# Walk up the tree using siblings
for sibling, position in proof:
    if position == "left":
        current = SHA256(sibling + current)
    else:
        current = SHA256(current + sibling)

# Final value should equal the published root
assert current == merkle_root
```

### Startup Rebuild

On server startup, `ForensicService.initialize()` rebuilds the Merkle tree from all persisted events. The tree is maintained in memory for fast proof generation.

---

## Ed25519 DID:key (Decentralized Identifiers)

**Location:** `friendlyface/crypto/did.py`  
**Dependency:** PyNaCl ≥1.5.0

### What Are DIDs?

DIDs (Decentralized Identifiers) are self-sovereign identifiers that don't depend on a central authority. In FriendlyFace, they identify the platform and forensic actors with cryptographically verifiable identities.

### Implementation

```python
from friendlyface.crypto.did import Ed25519DIDKey

# Generate a new DID
did_key = Ed25519DIDKey()
print(did_key.did)  # "did:key:z6MkhaXg..."

# Deterministic generation (for testing)
did_key = Ed25519DIDKey.from_seed(bytes.fromhex("abcd" * 8))

# Sign data
signature = did_key.sign(b"forensic evidence")

# Verify signature
is_valid = did_key.verify(b"forensic evidence", signature)

# Export public key for sharing
public_key_bytes = did_key.export_public()

# Resolve to DID Document
did_document = did_key.resolve()
```

### DID Format

```
did:key:z6Mk<base58-encoded-ed25519-public-key>
```

### DID Document

```json
{
  "id": "did:key:z6MkhaXg...",
  "type": "Ed25519VerificationKey2020",
  "controller": "did:key:z6MkhaXg...",
  "publicKeyHex": "abc123...",
  "status": "active"
}
```

### Platform DID

The platform has a persistent DID key used for signing forensic bundles:

- If `FF_DID_SEED` is set (64 hex chars = 32 bytes), the key is deterministically derived — same seed always produces the same DID
- If `FF_DID_SEED` is unset, a random key is generated on each startup
- The platform DID signs the Verifiable Credential embedded in every forensic bundle

---

## Verifiable Credentials (VCs)

**Location:** `friendlyface/crypto/vc.py`

W3C-style Verifiable Credentials that anchor forensic bundle identity.

### Issuance

```python
from friendlyface.crypto.vc import VerifiableCredential
from friendlyface.crypto.did import Ed25519DIDKey

issuer = Ed25519DIDKey()
vc = VerifiableCredential(issuer=issuer)

credential = vc.issue(
    claims={"bundle_id": "abc123", "event_count": 42},
    credential_type="ForensicCredential",
    subject_did="did:key:z6MkSubject...",
)
```

### Verification

```python
result = VerifiableCredential.verify(
    credential=credential,
    issuer_public_key=issuer.export_public(),
)
# result = {"valid": True, "issuer": "did:key:z6Mk...", "credential_type": "ForensicCredential"}
```

### In Forensic Bundles

Every bundle created by `ForensicService.create_bundle()` automatically:
1. Issues a `ForensicCredential` VC signed by the platform DID
2. Embeds the credential in the bundle's `did_credential_placeholder` field
3. Verification checks the VC signature as part of bundle verification

---

## Schnorr ZK Proofs (Zero-Knowledge Bundle Verification)

**Location:** `friendlyface/crypto/schnorr.py`

Non-interactive Schnorr proofs using the Fiat-Shamir heuristic. These prove that a forensic bundle has a specific hash without revealing the contents.

### Why Zero-Knowledge?

External auditors need to verify bundle integrity without accessing raw biometric data. ZK proofs provide cryptographic verification while preserving privacy.

### How Schnorr Proofs Work

```
Prover has:     secret s (derived from bundle hash)
                public point P = s * G (where G is the generator)

1. Commit:      choose random r, compute R = r * G
2. Challenge:   c = SHA256(R || P || message)  [Fiat-Shamir]
3. Response:    z = r + c * s (mod order)

Verifier checks: z * G == R + c * P
```

### Implementation

```python
from friendlyface.crypto.schnorr import ZKBundleProver, ZKBundleVerifier

# Generate proof
prover = ZKBundleProver()
proof_json = prover.prove_bundle(bundle_id="abc123", bundle_hash="def456...")

# Verify proof
verifier = ZKBundleVerifier()
is_valid = verifier.verify_bundle(proof_json)
```

### Proof Structure

```json
{
  "scheme": "schnorr-sha256",
  "commitment": "hex...",
  "challenge": "hex...",
  "response": "hex...",
  "public_point": "hex...",
  "bundle_id": "abc123",
  "bundle_hash": "def456..."
}
```

### In Forensic Bundles

Every bundle created by `ForensicService.create_bundle()` automatically:
1. Generates a Schnorr proof over the bundle hash
2. Stores the proof in the bundle's `zk_proof_placeholder` field
3. Verification checks the Schnorr proof as part of bundle verification

---

## Backward Compatibility

### Legacy Stub Handling

The original implementation used HMAC-based stubs (`friendlyface/stubs/`). The verification system handles legacy formats gracefully:

- **ZK proofs:** If the proof starts with `"zk_stub::"` or is a `pedersen-sha256` format, it passes verification (legacy acceptance)
- **DID credentials:** If the credential starts with `"stub::"`, it passes verification
- **Null values:** Missing proofs or credentials also pass (for bundles created before crypto integration)

This ensures that bundles created during Phase 1 (before real crypto was implemented) remain verifiable.

---

## Authentication

### API Key Authentication

FriendlyFace uses simple API key authentication:

```python
# auth.py
async def require_api_key(request: Request):
    if not FF_API_KEYS:  # Dev mode — no auth
        return
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key not in FF_API_KEYS:
        raise HTTPException(status_code=401)
```

**Key points:**
- Keys are set via `FF_API_KEYS` environment variable (comma-separated)
- Passed via `X-API-Key` header or `api_key` query parameter
- Empty `FF_API_KEYS` = dev mode (all requests pass)
- `/health` endpoint is always public

### Security Notes

- API keys are **shared secrets** — suitable for development and internal use
- For production, consider adding OAuth 2.0 or mutual TLS
- DID-based authentication for FL participants is a future enhancement
- All API key comparisons use constant-time comparison to prevent timing attacks

---

## Data at Rest

### SQLite

- Database file at `FF_DB_PATH` (default: `friendlyface.db`)
- No built-in encryption — use filesystem-level encryption (LUKS, FileVault, etc.)
- Consent records are append-only — no `DELETE` operations, only `INSERT`

### Supabase

- Data encrypted at rest by Supabase infrastructure
- Row-level security can be configured via Supabase dashboard
- Service role key required (`SUPABASE_KEY`) — keep this secret

---

## Threat Model Summary

| Threat | Mitigation |
|--------|------------|
| Event tampering | SHA-256 hash chain + Merkle tree |
| Silent deletion | Append-only design, sequence number gaps detected |
| Bundle forgery | Schnorr ZK proofs + Ed25519 DID credentials |
| Identity impersonation | DID:key with Ed25519 signatures |
| Biased model deployment | Automatic fairness auditing with forensic logging |
| Unauthorized inference | Consent check before every prediction |
| Privacy violation | ZK proofs allow verification without data disclosure |
| Poisoned FL updates | Norm-based anomaly detection + forensic alerts |
