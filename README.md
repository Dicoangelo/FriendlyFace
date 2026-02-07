# FriendlyFace

[![CI](https://github.com/Dicoangelo/FriendlyFace/actions/workflows/ci.yml/badge.svg)](https://github.com/Dicoangelo/FriendlyFace/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-560%20passing-brightgreen.svg)](#testing)

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
|  LIME Saliency   |  KernelSHAP Values   |  Compare Endpoint  |
+---------------------------------------------------------------+
|  LAYER 4: FAIRNESS & BIAS AUDITOR                             |
|  Demographic Parity  |  Equalized Odds  |  Auto-Audit Trigger |
+---------------------------------------------------------------+
|  LAYER 3: BLOCKCHAIN FORENSIC LAYER (Mohammed Schema)         |
|  Hash-Chained Events  |  Merkle Tree  |  Provenance DAG      |
+---------------------------------------------------------------+
|  LAYER 2: FEDERATED LEARNING ENGINE                           |
|  FedAvg Simulation  |  Poisoning Detection  |  Round Logging  |
+---------------------------------------------------------------+
|  LAYER 1: RECOGNITION ENGINE                                  |
|  PCA Feature Extraction  |  SVM Classifier  |  Inference API  |
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
```

## API Reference

**46 endpoints** across 8 domains. All endpoints require API key auth (header `X-API-Key`) except `/health`.

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (public) |

### Forensic Events

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/events` | Record a forensic event |
| GET | `/events` | List all events |
| GET | `/events/{id}` | Get event by ID |

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
| GET | `/chain/integrity` | Verify entire hash chain |

### Provenance

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/provenance` | Add provenance node |
| GET | `/provenance/{node_id}` | Get provenance chain |

### Recognition (Layer 1)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/recognition/train` | Train PCA+SVM on dataset |
| POST | `/recognition/predict` | Upload image for recognition |
| GET | `/recognition/models` | List trained models |
| GET | `/recognition/models/{id}` | Model details + provenance chain |

### Federated Learning (Layer 2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/fl/start` | Start FL simulation |
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
| `FF_HOST` | `0.0.0.0` | Server bind host |
| `FF_PORT` | `8000` | Server bind port |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase service role key |

## Testing

```bash
# Full test suite (560 tests)
pytest tests/ -v

# Specific layer
pytest tests/test_recognition_api.py tests/test_pca.py tests/test_svm.py -v  # Layer 1
pytest tests/test_fl.py tests/test_fl_api.py tests/test_poisoning.py -v      # Layer 2
pytest tests/test_integrity.py tests/test_merkle.py tests/test_provenance.py -v  # Layer 3
pytest tests/test_fairness.py tests/test_fairness_api.py -v                   # Layer 4
pytest tests/test_explainability.py tests/test_shap_explainability.py -v      # Layer 5
pytest tests/test_consent_api.py tests/test_governance.py tests/test_compliance.py -v  # Layer 6

# End-to-end pipeline
pytest tests/test_e2e_pipeline.py -v

# Lint
ruff check friendlyface/ tests/
```

## Deployment

### Docker

```bash
docker build -t friendlyface .
docker run -p 8000:8000 -e FF_API_KEYS=mykey friendlyface
```

### Fly.io

```bash
fly launch --config fly.toml
fly secrets set FF_API_KEYS=mykey
fly deploy
```

### Railway

```bash
railway init
railway up
```

## Project Structure

```
friendlyface/
├── api/
│   └── app.py              # FastAPI application (46 endpoints)
├── auth.py                  # API key authentication
├── core/
│   ├── models.py            # ForensicEvent, MerkleTree, ProvenanceNode, Bundle
│   ├── merkle.py            # Append-only Merkle tree
│   ├── provenance.py        # Provenance DAG
│   └── service.py           # ForensicService orchestrator
├── recognition/
│   ├── pca.py               # PCA dimensionality reduction
│   ├── svm.py               # SVM classifier training
│   └── inference.py         # Face recognition inference
├── fl/
│   ├── engine.py            # Federated learning simulation (FedAvg)
│   └── poisoning.py         # Norm-based poisoning detection
├── fairness/
│   └── auditor.py           # Bias auditing (demographic parity + equalized odds)
├── explainability/
│   ├── lime_explain.py      # LIME explanations
│   └── shap_explain.py      # KernelSHAP explanations
├── governance/
│   ├── consent.py           # Consent management (append-only)
│   └── compliance.py        # EU AI Act compliance reporting
├── storage/
│   ├── database.py          # SQLite async backend
│   └── supabase_db.py       # Supabase backend
└── stubs/
    ├── did.py               # DID/VC stubs (Phase 2)
    └── zk.py                # ZK proof stubs (Phase 2)
```

## Research Lineage

Built on Safiia Mohammed's forensic-friendly AI framework (University of Windsor, ICDF2C 2024) with SOTA 2025-2026 components:

| Paper | Integration |
|-------|-------------|
| Mohammed, ICDF2C 2024 | Hash-chained events, provenance DAG, forensic bundles |
| BioZero (arXiv:2409.17509) | Merkle tree verification (ZK proofs in Phase 2) |
| TBFL (arXiv:2602.02629) | DID/VC identity (Phase 2) |
| FedFDP (arXiv:2402.16028) | Fairness-aware FL (Phase 2) |
| SDD (arXiv:2505.03837) | Explainable saliency (Phase 2) |
| EU AI Act (arXiv:2512.13907) | Compliance reporting framework |

## License

MIT
