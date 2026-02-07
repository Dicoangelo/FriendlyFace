# CLAUDE.md

## Overview

FriendlyFace is a forensic-friendly AI facial recognition platform implementing Safiia Mohammed's ICDF2C 2024 schema. 6-layer architecture: Recognition, FL, Blockchain Forensic, Fairness, Explainability, Consent & Governance.

## Commands

```bash
# Dev server (port 3849, auto-reload)
source .venv/bin/activate
python3 -m friendlyface

# Tests (560 passing, ~29s)
pytest tests/ -v
pytest tests/test_e2e_pipeline.py -v  # Full pipeline test

# Lint
ruff check friendlyface/ tests/
ruff format --check friendlyface/ tests/

# Docker
docker build -t friendlyface .
docker run -p 8000:8000 friendlyface
```

## Architecture

**Core pattern:** Every operation produces a `ForensicEvent` → hash-chained → Merkle tree → provenance DAG.

```
friendlyface/
├── api/app.py           # 46 FastAPI endpoints
├── auth.py              # API key auth (X-API-Key header)
├── core/                # ForensicEvent, MerkleTree, ProvenanceNode, Bundle
├── recognition/         # PCA + SVM (scikit-learn)
├── fl/                  # FedAvg simulation + poisoning detection
├── fairness/            # Demographic parity + equalized odds
├── explainability/      # LIME (real lib) + KernelSHAP (custom)
├── governance/          # Consent (append-only) + EU AI Act compliance
├── storage/             # SQLite (default) or Supabase (FF_STORAGE=supabase)
└── stubs/               # DID + ZK placeholders (Phase 2)
```

## Key Patterns

- **Hash chaining:** Each `ForensicEvent.event_hash = SHA256(content + previous_hash)`
- **Merkle tree:** Append-only, rebuilt from DB on startup via `ForensicService.initialize()`
- **Provenance DAG:** Nodes link training → model → inference → explanation → bundle
- **Storage switching:** `FF_STORAGE=sqlite|supabase` env var in `api/app.py:_create_database()`
- **Auth bypass:** Empty `FF_API_KEYS` = dev mode (no auth)
- **Test fixtures:** `conftest.py` provides `db`, `service`, `client` fixtures with state reset

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `FF_STORAGE` | `sqlite` | Backend selection |
| `FF_DB_PATH` | `friendlyface.db` | SQLite path |
| `FF_API_KEYS` | *(empty=dev)* | Comma-separated keys |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase service role key |

## Test Structure

25 test files, 560 tests. Key test files:
- `test_e2e_pipeline.py` — Full forensic lifecycle (consent → train → recognize → explain → audit → bundle → verify)
- `test_supabase.py` — Supabase adapter with mocked client
- `test_auth.py` — API key auth enforcement

## Deployment

- **Fly.io:** `fly.toml` (iad region, 256MB, SQLite on mounted volume)
- **Railway:** `railway.toml` (Dockerfile builder, 120s healthcheck)
- **Render:** `render.yaml` (free tier, Docker)
- **Docker:** Multi-stage build, non-root user (ffuser:1000)

## Dependencies

Core: FastAPI, uvicorn, pydantic, aiosqlite, scikit-learn, numpy, Pillow, lime, supabase
Dev: pytest, pytest-asyncio, httpx, ruff

## GitHub

- Repo: Dicoangelo/FriendlyFace (public)
- CI: GitHub Actions (test py3.11+3.12, ruff lint)
