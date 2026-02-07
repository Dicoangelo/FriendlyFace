# CLAUDE.md

## Overview

FriendlyFace is a forensic-friendly AI facial recognition platform implementing Safiia Mohammed's ICDF2C 2024 schema. 6-layer architecture: Recognition, FL, Blockchain Forensic, Fairness, Explainability, Consent & Governance.

## Commands

```bash
# Dev server (port 3849, auto-reload)
source .venv/bin/activate
python3 -m friendlyface

# Tests (865+ passing, ~29s)
pytest tests/ -v
pytest tests/test_e2e_pipeline.py -v  # Full pipeline test

# Lint + format
ruff check friendlyface/ tests/
ruff format --check friendlyface/ tests/

# Frontend
cd frontend && npm run build && npm run lint

# Docker
docker build -t friendlyface .
docker run -p 8000:8000 friendlyface
```

## Architecture

**Core pattern:** Every operation produces a `ForensicEvent` → hash-chained → Merkle tree → provenance DAG.

```
friendlyface/
├── api/app.py           # 62+ FastAPI endpoints + /api/v1/ versioned routes + OpenAPI tags
├── api/sse.py           # Server-Sent Events broadcaster
├── auth.py              # API key auth (X-API-Key header) + audit logging
├── config.py            # Pydantic BaseSettings (all FF_* env vars, validated at startup)
├── exceptions.py        # Custom exception hierarchy + centralized error middleware
├── core/                # ForensicEvent, MerkleTree, ProvenanceNode, Bundle
├── crypto/              # Ed25519 DID:key (PyNaCl), Schnorr ZK proofs, VCs
├── recognition/         # PCA + SVM (scikit-learn) + voice biometrics (MFCC)
├── fl/                  # FedAvg + DP-FedAvg + poisoning detection
├── fairness/            # Demographic parity + equalized odds + auto-audit
├── explainability/      # LIME (real lib) + KernelSHAP + SDD saliency
├── governance/          # Consent (append-only) + EU AI Act compliance
├── storage/             # SQLite (default) or Supabase (FF_STORAGE=supabase)
├── logging_config.py    # Structured JSON logging (FF_LOG_FORMAT=json)
└── frontend/            # React 19 + Vite + TailwindCSS dashboard (9 pages)
    └── src/components/  # ErrorBoundary, Toast, ConfirmDialog
```

## Key Patterns

- **Hash chaining:** Each `ForensicEvent.event_hash = SHA256(content + previous_hash)`
- **Merkle tree:** Append-only, rebuilt from DB on startup via `ForensicService.initialize()`
- **Provenance DAG:** Nodes link training → model → inference → explanation → bundle
- **Storage switching:** `FF_STORAGE=sqlite|supabase` env var
- **Auth bypass:** Empty `FF_API_KEYS` = dev mode (no auth)
- **Crypto:** Ed25519 DID:key + Schnorr ZK proofs auto-generated on bundle creation
- **SSE:** Real-time forensic event stream at `/events/stream`
- **API versioning:** All routes available at `/api/v1/` prefix (backward compatible)
- **Pagination:** List endpoints use `limit`/`offset` params, return `{items, total, limit, offset}`
- **Rate limiting:** slowapi, global 100/min/IP. Sensitive: `/recognition/train` 5/min, `/did/create` 10/min, `/zk/prove` 20/min
- **Config:** Pydantic `BaseSettings` in `config.py` validates all `FF_*` vars at startup (fail-fast)
- **Exceptions:** Custom hierarchy in `exceptions.py` → centralized error middleware → structured JSON errors with `request_id`
- **Metrics:** Prometheus at `/metrics` via `prometheus-fastapi-instrumentator`
- **Audit logging:** Auth failures + consent grant/revoke logged with `event_category=audit`
- **Test fixtures:** `conftest.py` provides `db`, `service`, `client` fixtures with state reset

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `FF_STORAGE` | `sqlite` | Backend selection |
| `FF_DB_PATH` | `friendlyface.db` | SQLite path |
| `FF_API_KEYS` | *(empty=dev)* | Comma-separated keys |
| `FF_DID_SEED` | *(auto-gen)* | Deterministic DID key seed |
| `FF_LOG_FORMAT` | `text` | `json` for structured logs |
| `FF_LOG_LEVEL` | `INFO` | Log level |
| `FF_SERVE_FRONTEND` | `true` | Serve built React frontend |
| `FF_RATE_LIMIT` | `100/minute` | Default rate limit (`none` to disable) |
| `FF_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase service role key |

## Test Structure

40+ test files, 865+ tests. Key test files:
- `test_e2e_pipeline.py` — Full forensic lifecycle
- `test_ed25519_did.py` — Ed25519 DID:key with PyNaCl
- `test_ed25519_vc.py` — Verifiable Credentials with Ed25519
- `test_schnorr.py` — Schnorr ZK proofs + legacy compat
- `test_bundle_crypto.py` — Bundle creation with crypto wiring
- `test_bundle_export.py` — JSON-LD export/import round-trip
- `test_api.py` — API integration tests (all endpoints)
- `test_api_versioning.py` — /api/v1/ prefix tests
- `test_logging.py` — Structured JSON logging tests

## Deployment

- **Railway:** `railway.toml` (Dockerfile builder, 120s healthcheck)
- **Fly.io:** `fly.toml` (iad region, 256MB, SQLite on mounted volume)
- **Docker:** Multi-stage build, non-root user (ffuser:1000)

## Dependencies

Core: FastAPI, uvicorn, pydantic, pydantic-settings, aiosqlite, scikit-learn, numpy, Pillow, lime, pynacl, supabase, slowapi, prometheus-fastapi-instrumentator
Dev: pytest, pytest-asyncio, httpx, ruff

## GitHub

- Repo: Dicoangelo/FriendlyFace (public)
- CI: GitHub Actions (test py3.11+3.12, ruff lint)
