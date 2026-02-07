# FriendlyFace — Architecture Guide

## System Architecture

FriendlyFace is organized as a **6-layer forensic stack** where each layer handles a specific responsibility, and every layer emits `ForensicEvent` records into a shared, immutable chain. The layers are designed to be independent but interconnected through the forensic backbone (Layer 3).

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 6: CONSENT & GOVERNANCE                                  │
│  Consent Engine  │  Compliance Reporter  │  EU AI Act Checks    │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 5: EXPLAINABILITY (XAI)                                  │
│  LIME Saliency   │  KernelSHAP Values   │  SDD Maps  │ Compare │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: FAIRNESS & BIAS AUDITOR                               │
│  Demographic Parity  │  Equalized Odds  │  Auto-Audit Trigger   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: BLOCKCHAIN FORENSIC LAYER (Mohammed Schema)           │
│  Hash-Chained Events  │  Merkle Tree  │  Provenance DAG         │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: FEDERATED LEARNING ENGINE                             │
│  FedAvg Simulation  │  Poisoning Detection  │  DP-FL  │ Rounds  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: RECOGNITION ENGINE                                    │
│  PCA Feature Extraction │ SVM Classifier │ Voice │ Multi-Modal   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Recognition Engine

**Purpose:** Extract features from biometric inputs and produce identification predictions.

**Location:** `friendlyface/recognition/`

### Components

**PCA Feature Extraction** (`pca.py`)
- Accepts aligned grayscale face images (112×112)
- Uses scikit-learn PCA for dimensionality reduction
- Outputs serialized `.pkl` model with explained variance metadata
- Training event logged as `ForensicEvent(event_type=TRAINING_START)` with dataset hash
- Provenance node created linking to dataset source

**SVM Classifier** (`svm.py`)
- Trains an SVM on PCA-reduced feature vectors
- Produces a serialized model with class labels
- Training event logged as `ForensicEvent(event_type=TRAINING_COMPLETE)`
- Provenance node linked to PCA model (DERIVED_FROM relationship)

**Face Inference** (`inference.py`)
- Loads PCA+SVM models, runs prediction on a single image
- Returns top-K matches with confidence scores
- Each inference logged as `ForensicEvent(event_type=INFERENCE_RESULT)` with input image hash, model hashes, and match details
- Auto-triggers bias audit if configured

**Voice Biometrics** (`voice.py`)
- Extracts MFCC embeddings from raw PCM audio (numpy-only DSP, no external audio libs)
- Mel-scale triangular filterbank and type-II DCT computed from scratch
- Cosine similarity matching against enrolled reference embeddings
- Enrollment and verification both produce forensic events

**Multi-Modal Fusion** (via API)
- Combines face and voice confidence scores
- Weighted fusion with configurable alpha parameter
- Single forensic event captures both modality results

### Data Flow
```
Image bytes → PCA transform → SVM predict → top-K matches
                                    ↓
                          ForensicEvent emitted
                                    ↓
                          Provenance node linked
                                    ↓
                          Auto-audit triggered (if configured)
```

---

## Layer 2: Federated Learning Engine

**Purpose:** Train models collaboratively across distributed clients without centralizing raw data, with built-in security against poisoned updates.

**Location:** `friendlyface/fl/`

### Components

**FedAvg Simulation** (`engine.py`)
- Simulates multiple FL clients with local datasets
- Implements Federated Averaging aggregation strategy
- Each round produces a `ForensicEvent(event_type=FL_ROUND)` with:
  - Round number, client count, global model hash
  - Per-client metadata (samples, local loss)
- Provenance DAG links rounds sequentially (each round DERIVED_FROM previous)

**Poisoning Detection** (`poisoning.py`)
- Norm-based anomaly detection on client update vectors
- Updates exceeding configurable threshold are flagged
- Flagged updates logged as `ForensicEvent(event_type=SECURITY_ALERT)`
- Security results included in round provenance

**Differential Privacy FL** (via API `/fl/dp-start`)
- Adds calibrated noise to aggregated updates
- Configurable epsilon/delta privacy budget
- Privacy spend tracked per simulation and logged per round
- Fairness-aware clipping: demographic groups get proportional noise

### Data Flow
```
FL Start → Client local training → Update collection
                                        ↓
                              Poisoning detection scan
                                        ↓
                              FedAvg aggregation
                                        ↓
                              Global model hash computed
                                        ↓
                              ForensicEvent + Provenance node per round
```

---

## Layer 3: Blockchain Forensic Layer

**Purpose:** The backbone of the entire system. Provides immutable record-keeping, integrity verification, and provenance tracking for all operations across all layers.

**Location:** `friendlyface/core/`

This is the Mohammed ICDF2C 2024 schema implementation and the architectural heart of FriendlyFace.

### Components

**ForensicEvent** (`models.py`)
- The fundamental record type. Every operation across all 6 layers produces one.
- Fields: `id`, `event_type`, `timestamp`, `actor`, `payload`, `previous_hash`, `event_hash`, `sequence_number`
- Hash chaining: `event_hash = SHA256(canonical_json(content) + previous_hash)`
- The `seal()` method computes and locks the hash — once sealed, modification is detectable
- Events are strictly ordered by `sequence_number`

**Merkle Tree** (`merkle.py`)
- Append-only binary Merkle tree
- Each forensic event's hash becomes a leaf
- Tree is rebuilt from the database on startup via `ForensicService.initialize()`
- Provides inclusion proofs: given an event ID, returns the siblings needed to recompute the root
- Root hash is the single value that summarizes the integrity of all events

**Provenance DAG** (`provenance.py`)
- Directed acyclic graph tracking data lineage
- Nodes represent entities (datasets, models, inferences, explanations, bundles)
- Edges represent relationships (`DERIVED_FROM`, `GENERATED_BY`, `USED`, `ATTRIBUTED_TO`)
- Each node has its own hash for tamper detection
- Chains can be traversed to answer "what data was used to train the model that produced this inference?"

**ForensicBundle** (`models.py` + `service.py`)
- Self-verifiable output artifacts that package everything needed for an audit
- Contains: event IDs, Merkle root, Merkle proofs, provenance chain, bias audit, layer artifacts
- Bundle hash covers all included content
- Verification checks: hash chain integrity, Merkle proofs, provenance validity, ZK proof, DID credential
- Layer filters allow creating bundles with only specific layers included

**ForensicService** (`service.py`)
- The orchestrator that ties everything together
- Manages event recording, Merkle tree updates, provenance linking, bundle creation and verification
- Handles the platform DID key (loaded from `FF_DID_SEED` or auto-generated)
- Collects artifacts from recognition, FL, bias, and explanation layers when building bundles

### Integrity Verification
```
Verify hash chain:  event[0].previous_hash == "GENESIS"
                    event[n].previous_hash == event[n-1].event_hash
                    event[n].event_hash == SHA256(content + previous_hash)

Verify Merkle tree: proof siblings → recompute → matches root

Verify provenance:  each node's hash matches recomputed hash
                    no orphaned nodes in the DAG

Verify bundle:      all of the above + ZK proof + DID credential
```

---

## Layer 4: Fairness & Bias Auditor

**Purpose:** Detect and document demographic disparities in recognition results.

**Location:** `friendlyface/fairness/`

### Components

**Bias Auditor** (`auditor.py`)
- Computes two fairness metrics:
  - **Demographic parity:** Are recognition rates equal across demographic groups?
  - **Equalized odds:** Are error rates (FPR, FNR) equal across groups?
- Accepts recognition results grouped by demographic attribute
- Outputs `BiasAuditRecord` with per-group metrics and overall fairness score
- Configurable thresholds determine pass/warning/fail status
- Each audit logged as `ForensicEvent(event_type=BIAS_AUDIT)`

**Auto-Audit Trigger**
- Configurable interval: after every N inferences, automatically run a bias audit
- Configuration persisted and accessible via API
- Ensures bias checking isn't something that gets "forgotten"

### Fairness Health Status
```
GET /fairness/status returns one of:
  "pass"    — all metrics within thresholds
  "warning" — metrics approaching thresholds
  "fail"    — one or more metrics exceed thresholds
```

---

## Layer 5: Explainability (XAI)

**Purpose:** Provide multiple, corroborating explanations for every recognition decision.

**Location:** `friendlyface/explainability/`

### Components

**LIME Explanations** (`lime_explain.py`)
- Uses the real LIME library (Local Interpretable Model-agnostic Explanations)
- Perturbs input features and observes prediction changes
- Produces per-feature importance scores
- Explanation artifact includes: feature weights, top contributing features, perturbation details

**KernelSHAP Explanations** (`shap_explain.py`)
- Custom implementation of KernelSHAP (Shapley Additive Explanations)
- Computes SHAP values per feature dimension
- Produces: SHAP values, feature importance ranking, base value
- Complementary to LIME — different method, corroborating results

**SDD Saliency Maps** (`sdd_explain.py`)
- Implements saliency computation per arXiv:2505.03837
- Pixel-level contribution maps showing which facial regions drove the recognition decision
- Particularly valuable for forensic examiners who need to understand "why this face matched"

**Compare Endpoint**
- `GET /explainability/compare/{event_id}` returns all available explanations for an inference
- Side-by-side LIME vs SHAP vs SDD for any single prediction
- Each explanation method logged as `ForensicEvent(event_type=EXPLANATION_GENERATED)`

---

## Layer 6: Consent & Governance

**Purpose:** Ensure biometric data is used only with explicit permission and in compliance with regulations.

**Location:** `friendlyface/governance/`

### Components

**Consent Engine** (`consent.py`)
- Manages consent records: subject_id, purpose, granted/denied, timestamp, expiry, revocation
- **Append-only storage:** consent records are never deleted, only new records are appended
- Consent checked before any recognition inference
- Revoked consent blocks future inference and logs the block
- All consent changes logged as `ForensicEvent(event_type=CONSENT_UPDATE)`

**Compliance Reporter** (`compliance.py`)
- Generates EU AI Act readiness reports
- Checks: consent coverage, bias audit recency, explainability availability, hash chain integrity
- Reports logged as `ForensicEvent(event_type=COMPLIANCE_REPORT)`

### Consent Lifecycle
```
Grant consent → ForensicEvent logged → consent active
                                           ↓
                              Inference allowed (consent checked)
                                           ↓
Revoke consent → ForensicEvent logged → consent inactive
                                           ↓
                              Inference blocked + block logged
```

---

## Cross-Cutting Concerns

### Storage Layer

**Location:** `friendlyface/storage/`

Two interchangeable backends selected via `FF_STORAGE` environment variable:

- **SQLite** (`database.py`): Default for development and testing. Async via aiosqlite. Full schema with tables for events, provenance nodes, bundles, bias audits, and consent records.
- **Supabase** (`supabase_db.py`): Production backend. Same interface, cloud-hosted PostgreSQL.

The `Database` class provides a repository-pattern interface that both backends implement identically.

### Authentication

**Location:** `friendlyface/auth.py`

- API key-based authentication via `X-API-Key` header or `api_key` query parameter
- Keys configured via `FF_API_KEYS` environment variable (comma-separated)
- **Dev mode:** When `FF_API_KEYS` is empty/unset, auth is disabled entirely
- `/health` endpoint is always public

### Cryptographic Layer

**Location:** `friendlyface/crypto/`

- **Ed25519 DID:key** (`did.py`): Real Ed25519 keypairs via PyNaCl. DID identifiers follow `did:key:z6Mk...` format.
- **Verifiable Credentials** (`vc.py`): W3C-style VCs signed by Ed25519 keys. Used to anchor bundle identity.
- **Schnorr ZK Proofs** (`schnorr.py`): Non-interactive Schnorr proofs (Fiat-Shamir heuristic). Prove bundle integrity without revealing contents.

### Legacy Stubs

**Location:** `friendlyface/stubs/`

Original stub implementations retained for backward compatibility:
- `did.py`: HMAC-based DID stubs (superseded by `crypto/did.py`)
- `zk.py`: Placeholder ZK proofs (superseded by `crypto/schnorr.py`)

Legacy formats are still accepted during verification for migration support.

### Event Broadcasting

**Location:** `friendlyface/api/app.py`

- SSE (Server-Sent Events) streaming via `GET /events/stream`
- Real-time forensic event notifications for monitoring dashboards
- `EventBroadcaster` class manages subscriber connections

---

## Directory Structure

```
friendlyface/
├── __main__.py              # Entry point (uvicorn startup, port 3849)
├── api/
│   └── app.py               # FastAPI application (46+ endpoints)
├── auth.py                   # API key authentication middleware
├── core/
│   ├── models.py             # ForensicEvent, MerkleNode, ProvenanceNode, Bundle, BiasAuditRecord
│   ├── merkle.py             # Append-only Merkle tree implementation
│   ├── provenance.py         # Provenance DAG implementation
│   └── service.py            # ForensicService orchestrator
├── recognition/
│   ├── pca.py                # PCA dimensionality reduction (scikit-learn)
│   ├── svm.py                # SVM classifier training (scikit-learn)
│   ├── inference.py          # Face recognition inference pipeline
│   └── voice.py              # Voice biometrics (MFCC + cosine similarity)
├── fl/
│   ├── engine.py             # Federated learning simulation (FedAvg)
│   └── poisoning.py          # Norm-based poisoning detection
├── fairness/
│   └── auditor.py            # Bias auditing (demographic parity + equalized odds)
├── explainability/
│   ├── lime_explain.py       # LIME explanations (real library)
│   ├── shap_explain.py       # KernelSHAP explanations (custom)
│   └── sdd_explain.py        # SDD saliency maps
├── governance/
│   ├── consent.py            # Consent management (append-only)
│   └── compliance.py         # EU AI Act compliance reporting
├── crypto/
│   ├── did.py                # Ed25519 DID:key (PyNaCl)
│   ├── vc.py                 # Verifiable Credentials (W3C-style)
│   └── schnorr.py            # Schnorr ZK proofs (numpy)
├── storage/
│   ├── database.py           # SQLite async backend (aiosqlite)
│   └── supabase_db.py        # Supabase backend
└── stubs/
    ├── did.py                # Legacy DID stubs (backward compat)
    └── zk.py                 # Legacy ZK stubs (backward compat)

tests/
├── conftest.py               # Shared fixtures (db, service, client)
├── test_e2e_pipeline.py      # Full 6-layer lifecycle test
├── test_storage.py           # SQLite backend tests
├── test_merkle.py            # Merkle tree tests
├── test_provenance.py        # Provenance DAG tests
├── test_integrity.py         # Hash chain verification tests
├── test_pca.py               # PCA training tests
├── test_svm.py               # SVM training tests
├── test_recognition_api.py   # Recognition endpoint tests
├── test_fl.py                # FL simulation tests
├── test_fl_api.py            # FL endpoint tests
├── test_poisoning.py         # Poisoning detection tests
├── test_fairness.py          # Bias auditor unit tests
├── test_fairness_api.py      # Fairness endpoint tests
├── test_explainability.py    # LIME explanation tests
├── test_shap_explainability.py # SHAP explanation tests
├── test_consent_api.py       # Consent endpoint tests
├── test_governance.py        # Governance/compliance tests
├── test_compliance.py        # Compliance reporting tests
├── test_bundle_artifacts.py  # Multi-layer bundle tests
├── test_bundle_crypto.py     # Bundle + DID + ZK tests
├── test_ed25519_did.py       # Ed25519 DID key tests
├── test_schnorr.py           # Schnorr ZK proof tests
├── test_did.py               # Legacy DID tests
├── test_api.py               # Core API endpoint tests
└── ... (25 test files total)
```
