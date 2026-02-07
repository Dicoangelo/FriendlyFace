# FriendlyFace

[![CI](https://github.com/Dicoangelo/FriendlyFace/actions/workflows/ci.yml/badge.svg)](https://github.com/Dicoangelo/FriendlyFace/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-881%2B%20passing-brightgreen.svg)](#testing)

**Forensic-Friendly AI Facial Recognition Platform** implementing Safiia Mohammed's ICDF2C 2024 forensic-friendly schema with 2025-2026 SOTA components.

Every recognition event is hash-chained, Merkle-verified, bias-audited, explainable, and consent-tracked. Built for EU AI Act readiness and courtroom-grade forensic evidence.

---

## Architecture

```
+---------------------------------------------------------------+
|  LAYER 6: CONSENT & GOVERNANCE                                |
|  Consent Engine  |  Compliance Reporter  |  EU AI Act Checks  |
+---------------------------------------------------------------+
|  LAYER 5: EXPLAINABILITY (XAI)                                |
|  LIME Saliency   |  KernelSHAP Values   |  SDD Saliency Maps |
+---------------------------------------------------------------+
|  LAYER 4: FAIRNESS & BIAS AUDITOR                             |
|  Demographic Parity  |  Equalized Odds  |  Auto-Audit Trigger |
+---------------------------------------------------------------+
|  LAYER 3: BLOCKCHAIN FORENSIC LAYER (Mohammed Schema)         |
|  Hash-Chained Events | Merkle Tree | Provenance DAG | ZK/DID |
+---------------------------------------------------------------+
|  LAYER 2: FEDERATED LEARNING ENGINE                           |
|  FedAvg + DP-FedAvg | Poisoning Detection | Privacy Budgets  |
+---------------------------------------------------------------+
|  LAYER 1: RECOGNITION ENGINE                                  |
|  PCA + SVM  |  Voice Biometrics (MFCC)  |  Multi-Modal Fusion |
+---------------------------------------------------------------+
```

**Core principle:** Every operation across all 6 layers produces an immutable `ForensicEvent` linked into a hash chain, Merkle tree, and provenance DAG.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Dicoangelo/FriendlyFace.git
cd FriendlyFace
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the server
python3 -m friendlyface  # Starts on http://localhost:3849

# Run tests
pytest tests/ -v

# Build the dashboard
cd frontend && npm install && npm run build
```

## Dashboard

FriendlyFace includes a React 19 + Vite + TailwindCSS dashboard with 9 pages:

- **Dashboard** — Platform health overview with auto-refresh
- **Event Stream** — Live SSE forensic event feed with type filtering
- **Events Table** — Paginated event browser with detail expansion
- **Bundles** — Bundle inspector with verify + JSON-LD export
- **DID/VC** — Create DIDs, issue/verify Verifiable Credentials
- **ZK Proofs** — Generate and verify Schnorr ZK proofs
- **FL Simulations** — Run FedAvg + DP-FedAvg simulations
- **Bias Audits** — Fairness reports and compliance status
- **Consent** — Grant, check, revoke consent records

The dashboard is served as static files from FastAPI — single deployment.

## API Reference

**62+ endpoints** across 12 domains. All available at both `/` and `/api/v1/` prefix. All endpoints require API key auth (header `X-API-Key`) except `/health` and `/metrics`.

> **Pagination:** List endpoints (`/events`, `/fairness/audits`, `/explainability/explanations`) accept `limit` (default 50, max 500) and `offset` query params. Responses use `{items, total, limit, offset}` envelope.
>
> **Rate limiting:** Global default `100/min/IP`. Sensitive endpoints: `/recognition/train` 5/min, `/did/create` 10/min, `/zk/prove` 20/min. Returns `429` with `Retry-After` header. Disable with `FF_RATE_LIMIT=none`.

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (public) |
| GET | `/api/version` | API version info |
| GET | `/metrics` | Prometheus metrics (public) |

### Forensic Events

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/events` | Record a forensic event |
| GET | `/events` | List all events |
| GET | `/events/{id}` | Get event by ID |
| GET | `/events/stream` | SSE real-time event stream |

### Merkle Tree

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/merkle/root` | Current Merkle root + leaf count |
| GET | `/merkle/proof/{event_id}` | Merkle inclusion proof for event |

### Forensic Bundles

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/bundles` | Create forensic bundle |
| GET | `/bundles/{id}` | Get bundle by ID |
| POST | `/verify/{bundle_id}` | Verify bundle integrity |
| GET | `/bundles/{id}/export` | Export as JSON-LD |
| POST | `/bundles/import` | Import JSON-LD bundle |
| GET | `/chain/integrity` | Verify entire hash chain |

### Provenance

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/provenance` | Add provenance node |
| GET | `/provenance/{node_id}` | Get provenance chain |

### DID / Verifiable Credentials

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/did/create` | Create Ed25519 DID:key |
| GET | `/did/{did_id}/resolve` | Resolve DID to DID Document |
| POST | `/vc/issue` | Issue signed Verifiable Credential |
| POST | `/vc/verify` | Verify credential signature |

### ZK Proofs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/zk/prove` | Generate Schnorr ZK proof for bundle |
| POST | `/zk/verify` | Verify ZK proof |
| GET | `/zk/proofs/{bundle_id}` | Get stored proof |

### Recognition (Layer 1)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/recognition/train` | Train PCA+SVM on dataset |
| POST | `/recognition/predict` | Upload image for recognition |
| GET | `/recognition/models` | List trained models |
| GET | `/recognition/models/{id}` | Model details + provenance chain |
| POST | `/voice/enroll` | Enroll voice biometric |
| POST | `/voice/verify` | Verify voice identity |
| POST | `/fusion/verify` | Multi-modal face + voice fusion |

### Federated Learning (Layer 2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/fl/start` | Start FL simulation (FedAvg) |
| POST | `/fl/dp-start` | Start DP-FedAvg with privacy budget |
| GET | `/fl/status` | Current FL training status |
| GET | `/fl/rounds` | List completed rounds |
| GET | `/fl/rounds/{sim}/{n}` | Round details + client contributions |
| GET | `/fl/rounds/{sim}/{n}/security` | Poisoning detection results |

### Fairness (Layer 4)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/fairness/audit` | Trigger manual bias audit |
| GET | `/fairness/audits` | List completed audits |
| GET | `/fairness/audits/{id}` | Full audit details |
| GET | `/fairness/status` | Fairness health (pass/warning/fail) |
| POST | `/fairness/config` | Configure auto-audit interval |
| GET | `/fairness/config` | Current auto-audit config |

### Explainability (Layer 5)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/explainability/lime` | Generate LIME explanation |
| POST | `/explainability/shap` | Generate SHAP explanation |
| POST | `/explainability/sdd` | Generate SDD saliency map |
| GET | `/explainability/explanations` | List all explanations |
| GET | `/explainability/explanations/{id}` | Get explanation by ID |
| GET | `/explainability/compare/{event_id}` | Compare LIME vs SHAP |

### Consent (Layer 6)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/consent/grant` | Grant consent for subject+purpose |
| POST | `/consent/revoke` | Revoke consent |
| GET | `/consent/status/{subject_id}` | Current consent status |
| GET | `/consent/history/{subject_id}` | Full consent history |
| POST | `/consent/check` | Verify consent before inference |

### Governance (Layer 6)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/governance/compliance` | Latest compliance report |
| POST | `/governance/compliance/generate` | Generate new report |

### Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/dashboard` | Audit dashboard with compliance summary |

## Authentication

Set API keys via environment variable:

```bash
export FF_API_KEYS="key1,key2,key3"
```

Pass via header or query parameter:

```bash
curl -H "X-API-Key: key1" http://localhost:3849/events
curl http://localhost:3849/events?api_key=key1
```

When `FF_API_KEYS` is unset, auth is disabled (dev mode).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FF_STORAGE` | `sqlite` | Storage backend (`sqlite` or `supabase`) |
| `FF_DB_PATH` | `friendlyface.db` | SQLite database path |
| `FF_API_KEYS` | *(empty)* | Comma-separated API keys |
| `FF_DID_SEED` | *(auto-gen)* | Deterministic Ed25519 DID key seed |
| `FF_LOG_FORMAT` | `text` | Set to `json` for structured JSON logs |
| `FF_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `FF_SERVE_FRONTEND` | `true` | Serve React dashboard static files |
| `FF_HOST` | `0.0.0.0` | Server bind host |
| `FF_PORT` | `8000` | Server bind port |
| `FF_RATE_LIMIT` | `100/minute` | Default rate limit (`none` to disable) |
| `FF_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase service role key |

> All `FF_*` variables are validated at startup via Pydantic `BaseSettings` (`friendlyface/config.py`). Invalid values cause a fast failure with a clear error message.

## Testing

```bash
# Full test suite (881+ tests)
pytest tests/ -v

# Specific layer
pytest tests/test_recognition_api.py tests/test_pca.py tests/test_svm.py -v  # Layer 1
pytest tests/test_fl.py tests/test_fl_api.py tests/test_poisoning.py -v      # Layer 2
pytest tests/test_integrity.py tests/test_merkle.py tests/test_provenance.py -v  # Layer 3
pytest tests/test_fairness.py tests/test_fairness_api.py -v                   # Layer 4
pytest tests/test_explainability.py tests/test_shap_explainability.py -v      # Layer 5
pytest tests/test_consent_api.py tests/test_governance.py tests/test_compliance.py -v  # Layer 6

# Cryptographic layer
pytest tests/test_ed25519_did.py tests/test_ed25519_vc.py tests/test_schnorr.py -v
pytest tests/test_bundle_crypto.py tests/test_bundle_export.py -v

# End-to-end pipeline
pytest tests/test_e2e_pipeline.py -v

# Lint + format
ruff check friendlyface/ tests/
ruff format --check friendlyface/ tests/
```

## Deployment

### Docker

```bash
docker build -t friendlyface .
docker run -p 8000:8000 -e FF_API_KEYS=mykey friendlyface
```

### Railway

```bash
railway init
railway up
```

### Fly.io

```bash
fly launch --config fly.toml
fly secrets set FF_API_KEYS=mykey
fly deploy
```

## Project Structure

```
friendlyface/
├── api/
│   ├── app.py              # FastAPI application (62+ endpoints, /api/v1/ versioned)
│   └── sse.py              # Server-Sent Events broadcaster
├── auth.py                  # API key authentication
├── core/
│   ├── models.py            # ForensicEvent, MerkleTree, ProvenanceNode, Bundle
│   ├── merkle.py            # Append-only Merkle tree
│   ├── provenance.py        # Provenance DAG
│   └── service.py           # ForensicService orchestrator
├── crypto/
│   ├── did.py               # Ed25519 DID:key (PyNaCl/libsodium)
│   ├── vc.py                # Verifiable Credentials with Ed25519 signatures
│   └── schnorr.py           # Schnorr ZK proofs (Fiat-Shamir, numpy-only)
├── recognition/
│   ├── pca.py               # PCA dimensionality reduction
│   ├── svm.py               # SVM classifier training
│   ├── inference.py         # Face recognition inference
│   └── voice.py             # Voice biometrics (MFCC extraction)
├── fl/
│   ├── engine.py            # Federated learning (FedAvg + DP-FedAvg)
│   └── poisoning.py         # Norm-based poisoning detection
├── fairness/
│   └── auditor.py           # Bias auditing (demographic parity + equalized odds)
├── explainability/
│   ├── lime_explain.py      # LIME explanations
│   ├── shap_explain.py      # KernelSHAP explanations
│   └── sdd.py               # SDD saliency maps (7-region decomposition)
├── governance/
│   ├── consent.py           # Consent management (append-only)
│   └── compliance.py        # EU AI Act compliance reporting
├── config.py                # Pydantic BaseSettings (all FF_* env vars)
├── exceptions.py            # Custom exception hierarchy + error middleware
├── storage/
│   ├── database.py          # SQLite async backend (indexed, paginated)
│   └── supabase_db.py       # Supabase backend
├── logging_config.py        # Structured JSON logging
└── frontend/                # React 19 + Vite + TailwindCSS dashboard
    ├── src/components/       # ErrorBoundary, Toast, ConfirmDialog
    ├── src/pages/            # 9 dashboard pages
    └── dist/                 # Built static files (served by FastAPI)
```

## Research Lineage

Built on Safiia Mohammed's forensic-friendly AI framework (University of Windsor, ICDF2C 2024) with SOTA 2025-2026 components:

| Paper | Integration |
|-------|-------------|
| Mohammed, ICDF2C 2024 | Hash-chained events, provenance DAG, forensic bundles |
| BioZero (arXiv:2409.17509) | Merkle tree verification, Schnorr ZK proofs |
| TBFL (arXiv:2602.02629) | Ed25519 DID:key identity, Verifiable Credentials |
| FedFDP (arXiv:2402.16028) | DP-FedAvg with calibrated Gaussian noise |
| SDD (arXiv:2505.03837) | Spatial-directional saliency decomposition |
| EU AI Act (arXiv:2512.13907) | Compliance reporting framework |

## License

MIT
