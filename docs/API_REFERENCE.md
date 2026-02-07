# FriendlyFace — API Reference

## Overview

FriendlyFace exposes **46+ REST endpoints** via FastAPI. The API is organized into 10 endpoint groups covering the full forensic lifecycle.

**Base URL:** `http://localhost:3849` (development)

**Authentication:** All endpoints except `/health` require an API key passed via:
- Header: `X-API-Key: <key>`
- Query param: `?api_key=<key>`
- Dev mode: Leave `FF_API_KEYS` unset to disable auth entirely

**Content Type:** `application/json` (except file uploads which use `multipart/form-data`)

---

## Health

### `GET /health`

Health check endpoint. **No authentication required.**

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 123.45
}
```

---

## Forensic Events

The core record type. Every operation across all 6 layers produces a ForensicEvent.

### `POST /events`

Record a forensic event manually.

**Request Body:**
```json
{
  "event_type": "inference_result",
  "actor": "analyst_01",
  "payload": {
    "model_id": "abc123",
    "confidence": 0.94
  }
}
```

**Response (201):**
```json
{
  "id": "uuid",
  "event_type": "inference_result",
  "timestamp": "2026-02-07T12:00:00Z",
  "actor": "analyst_01",
  "payload": { ... },
  "previous_hash": "abc...",
  "event_hash": "def...",
  "sequence_number": 42
}
```

### `GET /events`

List all forensic events in sequence order.

**Response:** Array of ForensicEvent objects.

### `GET /events/{id}`

Get a specific event by UUID.

**Response:** Single ForensicEvent object, or 404.

### `GET /events/stream`

**SSE (Server-Sent Events)** — real-time stream of forensic events as they occur.

**Usage:**
```bash
curl -N -H "X-API-Key: key1" http://localhost:3849/events/stream
```

Events are pushed as JSON payloads in standard SSE format.

---

## Merkle Tree

### `GET /merkle/root`

Get the current Merkle root hash and leaf count.

**Response:**
```json
{
  "root": "abc123...",
  "leaf_count": 42
}
```

### `GET /merkle/proof/{event_id}`

Get a Merkle inclusion proof for a specific event. This proof can be used to independently verify that the event exists in the tree without needing the full database.

**Response:**
```json
{
  "event_id": "uuid",
  "leaf_index": 5,
  "siblings": ["hash1", "hash2", "hash3"],
  "root": "abc123..."
}
```

---

## Forensic Bundles

Self-verifiable evidence packages containing events, proofs, provenance, and layer artifacts.

### `POST /bundles`

Create a forensic bundle from specified events.

**Request Body:**
```json
{
  "event_ids": ["uuid1", "uuid2", "uuid3"],
  "provenance_node_ids": ["prov1", "prov2"],
  "layer_filters": ["recognition", "bias", "explanation"]
}
```

`layer_filters` is optional. Valid values: `"recognition"`, `"fl"`, `"bias"`, `"explanation"`. When omitted, all available layers are included.

**Response (201):** ForensicBundle object with:
- Merkle root and proofs for included events
- Provenance chain
- Layer artifacts (recognition, FL, bias, explanation)
- ZK proof (Schnorr)
- DID credential (Ed25519 VC)
- Bundle hash covering all contents

### `GET /bundles/{id}`

Retrieve a bundle by UUID.

### `POST /verify/{bundle_id}`

Verify a bundle's integrity. Checks:
1. Hash chain validity for included events
2. Merkle proof verification
3. Provenance DAG connectivity
4. Layer artifact integrity
5. ZK proof validation
6. DID credential verification

**Response:**
```json
{
  "valid": true,
  "chain_valid": true,
  "merkle_valid": true,
  "provenance_valid": true,
  "zk_valid": true,
  "did_valid": true,
  "credential_issuer": "did:key:z6Mk...",
  "layer_artifacts": {
    "recognition": { "valid": true },
    "fl": { "valid": true },
    "bias": { "valid": true },
    "explanation": { "valid": true }
  }
}
```

### `GET /chain/integrity`

Verify the entire hash chain across all events.

**Response:**
```json
{
  "valid": true,
  "count": 142,
  "errors": []
}
```

---

## Provenance

### `POST /provenance`

Add a provenance node to the DAG.

**Request Body:**
```json
{
  "entity_type": "dataset",
  "entity_id": "faces_v1",
  "parents": ["uuid_of_parent_node"],
  "relations": ["derived_from"],
  "metadata": { "source": "curated_faces", "n_samples": 1000 }
}
```

### `GET /provenance/{node_id}`

Get the provenance chain for a node (the node plus all ancestors).

---

## Recognition (Layer 1)

### `POST /recognition/train`

Train a PCA+SVM pipeline on a dataset.

**Request Body:**
```json
{
  "dataset_path": "/path/to/aligned/faces",
  "output_dir": "/path/to/save/models",
  "n_components": 50,
  "labels": ["person_a", "person_a", "person_b", "person_b", ...]
}
```

**Response (201):**
```json
{
  "model_id": "uuid",
  "pca_event_id": "uuid",
  "svm_event_id": "uuid",
  "pca_provenance_id": "uuid",
  "svm_provenance_id": "uuid",
  "n_components": 50,
  "n_samples": 100,
  "n_classes": 10
}
```

### `POST /recognition/predict`

Upload an image for face recognition.

**Request:** `multipart/form-data` with `image` field  
**Query param:** `top_k` (default: 5)

**Response:**
```json
{
  "event_id": "uuid",
  "input_hash": "sha256...",
  "matches": [
    { "label": "person_a", "confidence": 0.94 },
    { "label": "person_b", "confidence": 0.87 }
  ]
}
```

### `POST /recognize`

Legacy alias for `/recognition/predict`. Same interface.

### `GET /recognition/models`

List all trained models with metadata.

### `GET /recognition/models/{id}`

Get model details with full provenance chain.

### `POST /recognition/voice/enroll`

Enroll a voice for a subject. Accepts raw PCM audio bytes.

### `POST /recognition/voice/verify`

Verify a voice against enrolled subjects.

### `POST /recognition/multimodal`

Multi-modal fusion combining face and voice recognition results with configurable weighting.

---

## Federated Learning (Layer 2)

### `POST /fl/start`

Start a federated learning simulation.

**Request Body:**
```json
{
  "n_clients": 5,
  "n_rounds": 10,
  "samples_per_client": 100,
  "poisoning_threshold": 3.0
}
```

### `POST /fl/dp-start`

Start a differential-privacy federated learning simulation.

**Additional parameters:**
```json
{
  "epsilon": 1.0,
  "delta": 1e-5,
  "clip_norm": 1.0
}
```

### `POST /fl/simulate`

Legacy alias for `/fl/start`.

### `GET /fl/rounds`

List all completed FL rounds with summary.

### `GET /fl/rounds/{sim_id}/{n}`

Get detailed round information including client contributions and security status.

### `GET /fl/rounds/{sim_id}/{n}/security`

Get poisoning detection results for a specific round.

**Response:**
```json
{
  "round": 3,
  "flagged_clients": ["client_2"],
  "threshold": 3.0,
  "client_norms": { "client_0": 1.2, "client_1": 0.8, "client_2": 5.7 }
}
```

### `GET /fl/status`

Current FL training status (idle, running, completed).

---

## Fairness (Layer 4)

### `POST /fairness/audit`

Trigger a manual bias audit on recent recognition results.

**Response:**
```json
{
  "id": "uuid",
  "event_id": "uuid",
  "demographic_parity_gap": 0.05,
  "equalized_odds_gap": 0.03,
  "groups_evaluated": ["group_a", "group_b"],
  "compliant": true,
  "details": { ... }
}
```

### `GET /fairness/audits`

List all completed bias audits.

### `GET /fairness/audits/{id}`

Get full details for a specific audit.

### `GET /fairness/status`

Get the current fairness health status.

**Response:**
```json
{
  "status": "pass",
  "last_audit_id": "uuid",
  "last_audit_time": "2026-02-07T12:00:00Z",
  "demographic_parity_gap": 0.05,
  "equalized_odds_gap": 0.03
}
```

Possible status values: `"pass"`, `"warning"`, `"fail"`

### `POST /fairness/config`

Configure auto-audit settings.

**Request Body:**
```json
{
  "auto_audit_interval": 10,
  "dp_threshold": 0.1,
  "eo_threshold": 0.1
}
```

### `GET /fairness/config`

Get current auto-audit configuration.

---

## Explainability (Layer 5)

### `POST /explainability/lime`

Generate a LIME explanation for a recognition event.

**Request Body:**
```json
{
  "event_id": "uuid_of_inference_event"
}
```

**Response:**
```json
{
  "event_id": "uuid",
  "method": "lime",
  "feature_weights": { ... },
  "top_features": [ ... ],
  "explanation_event_id": "uuid"
}
```

### `POST /explainability/shap`

Generate a KernelSHAP explanation.

### `POST /explainability/sdd`

Generate an SDD saliency map explanation.

### `GET /explainability/explanations`

List all generated explanations.

### `GET /explainability/explanations/{id}`

Get a specific explanation by ID.

### `GET /explainability/compare/{event_id}`

Compare all available explanation methods (LIME, SHAP, SDD) for a single inference event.

**Response:**
```json
{
  "event_id": "uuid",
  "explanations": {
    "lime": { ... },
    "shap": { ... },
    "sdd": { ... }
  }
}
```

---

## Consent (Layer 6)

### `POST /consent/grant`

Grant consent for a subject and purpose.

**Request Body:**
```json
{
  "subject_id": "subject_alpha",
  "purpose": "recognition",
  "actor": "admin_01"
}
```

**Response (201):**
```json
{
  "subject_id": "subject_alpha",
  "purpose": "recognition",
  "granted": true,
  "event_id": "uuid"
}
```

### `POST /consent/revoke`

Revoke consent for a subject and purpose.

**Request Body:**
```json
{
  "subject_id": "subject_alpha",
  "purpose": "recognition",
  "reason": "Subject requested deletion"
}
```

### `GET /consent/status/{subject_id}`

Get current consent status for a subject across all purposes.

### `GET /consent/history/{subject_id}`

Full consent history (all grants, revocations) for a subject. Since consent is append-only, this is a complete audit trail.

### `POST /consent/check`

Check if consent is active before performing an operation.

**Request Body:**
```json
{
  "subject_id": "subject_alpha",
  "purpose": "recognition"
}
```

**Response:**
```json
{
  "allowed": true,
  "active": true,
  "expires_at": null
}
```

---

## Governance (Layer 6)

### `GET /governance/compliance`

Get the latest compliance report.

### `POST /governance/compliance/generate`

Generate a new EU AI Act compliance report. Checks consent coverage, bias audit recency, explainability availability, and chain integrity.

---

## DID / Verifiable Credentials

### `POST /did/create`

Create a new Ed25519 DID:key identifier.

**Request Body (optional):**
```json
{
  "seed": "hex_encoded_32_byte_seed"
}
```

**Response (201):**
```json
{
  "did": "did:key:z6Mk...",
  "public_key_hex": "abc123...",
  "created_at": "2026-02-07T12:00:00Z"
}
```

### `GET /did/{did_id}/resolve`

Resolve a DID to its DID Document.

**Response:**
```json
{
  "id": "did:key:z6Mk...",
  "type": "Ed25519VerificationKey2020",
  "controller": "did:key:z6Mk...",
  "publicKeyHex": "abc123...",
  "status": "active"
}
```

### `POST /vc/issue`

Issue a Verifiable Credential.

**Request Body:**
```json
{
  "issuer_did_id": "did:key:z6Mk...",
  "subject_did": "did:key:z6Mk...",
  "claims": { "role": "forensic_examiner", "clearance": "level_3" },
  "credential_type": "ForensicCredential"
}
```

### `POST /vc/verify`

Verify a Verifiable Credential.

**Request Body:**
```json
{
  "credential": { ... },
  "issuer_public_key_hex": "abc123..."
}
```

**Response:**
```json
{
  "valid": true,
  "issuer": "did:key:z6Mk...",
  "credential_type": "ForensicCredential"
}
```

---

## ZK Proofs

### `POST /zk/prove`

Generate a Schnorr ZK proof for a forensic bundle.

**Request Body:**
```json
{
  "bundle_id": "uuid"
}
```

**Response (201):**
```json
{
  "proof": {
    "scheme": "schnorr-sha256",
    "commitment": "hex...",
    "challenge": "hex...",
    "response": "hex...",
    "public_point": "hex..."
  },
  "bundle_id": "uuid",
  "bundle_hash": "abc123..."
}
```

### `POST /zk/verify`

Verify a Schnorr ZK proof.

**Request Body:**
```json
{
  "proof": "{json_string_of_proof}"
}
```

**Response:**
```json
{
  "valid": true,
  "scheme": "schnorr-sha256"
}
```

### `GET /zk/proofs/{bundle_id}`

Get the stored ZK proof for a bundle.

---

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Meaning |
|------|---------|
| 400 | Bad request (invalid input, empty upload, etc.) |
| 401 | Unauthorized (missing or invalid API key) |
| 404 | Resource not found |
| 503 | Service unavailable (models not loaded, etc.) |

Error response format:
```json
{
  "detail": "Human-readable error message"
}
```

---

## Interactive Documentation

FastAPI auto-generates interactive docs:
- **Swagger UI:** `http://localhost:3849/docs`
- **ReDoc:** `http://localhost:3849/redoc`
